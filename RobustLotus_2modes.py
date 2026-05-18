#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RobustLotus  –  Unified Pipeline
==================================

Two modes, one shared processing chain:

  ┌─ HEALING ──────────────────────────────────────────────────────────────────┐
  │  Input  : OBJ mesh                                                         │
  │  Step 1 : OBJ → voxel grid → ray-cast support analysis → temp PLY         │
  │  Step 2 : temp PLY → LotusMesh (voxelise → shell → decimate → generalise) │
  │  Output : OBJ mesh  /  CityJSON (LOD2 solid)                               │
  └────────────────────────────────────────────────────────────────────────────┘

  ┌─ RECONSTRUCTION ────────────────────────────────────────────────────────────┐
  │  Input  : PLY point cloud                                                   │
  │  Step 1 : PLY → LotusMesh (voxelise → shell → decimate → generalise)       │
  │  Output : OBJ mesh  /  CityJSON (LOD2 solid)                                │
  └────────────────────────────────────────────────────────────────────────────┘

All mesh parameters (voxel size, decimation, smoothing, Taubin) and the output
format/EPSG settings are shared between both modes.
The only difference is the input: an OBJ mesh (Healing) vs. a PLY point cloud
(Reconstruction).  In Healing mode, an intermediate PLY is written next to the
input file and fed straight into the LotusMesh stage.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard / third-party imports
# ──────────────────────────────────────────────────────────────────────────────
import json
import os
import queue
import threading
import uuid
from functools import reduce

import numpy as np
import open3d as o3d
import open3d.core as o3c
import trimesh
from numba import cuda
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox

# healing_lib is optional – instant_grid_estimate is currently unused
try:
    from healing_lib import *
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 ─ CUDA KERNEL  (module-level, required by Numba)
# ══════════════════════════════════════════════════════════════════════════════

@cuda.jit
def filter_chunk_cuda_kernel(vox_chunk, sphere_centers, radius2, keep_out):
    """
    CUDA kernel: each thread checks one voxel against every sphere centre.
    Marks the voxel as "discard" (0) if any sphere encloses it.
    """
    idx = cuda.grid(1)
    if idx < vox_chunk.shape[0]:
        vx, vy, vz = vox_chunk[idx, 0], vox_chunk[idx, 1], vox_chunk[idx, 2]
        n_sph   = sphere_centers.shape[0]
        is_kept = 1
        for j in range(n_sph):
            dx = vx - sphere_centers[j, 0]
            dy = vy - sphere_centers[j, 1]
            dz = vz - sphere_centers[j, 2]
            if dx*dx + dy*dy + dz*dz <= radius2:
                is_kept = 0
                break
        keep_out[idx] = is_kept


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 ─ LotusMesh CLASS  (shared by both modes)
# ══════════════════════════════════════════════════════════════════════════════

class LotusMesh:
    """
    Full point-cloud → watertight mesh pipeline.

    Parameters
    ----------
    ply_path          : str   – input PLY file
    voxel_size        : float – voxel edge length (metres)
    decimate_percent  : float – target face ratio after decimation (0 < x ≤ 1)
    smooth_iterations : int   – Taubin smoothing iterations
    taubin_lambda     : float
    taubin_mu         : float
    log_fn            : callable(str)
    """

    def __init__(self, ply_path, voxel_size=1.0,
                 decimate_percent=0.25, smooth_iterations=10,
                 taubin_lambda=0.20, taubin_mu=-0.22,
                 log_fn=print):
        self.ply_path          = ply_path
        self.voxel_size        = voxel_size
        self.decimate_percent  = decimate_percent
        self.smooth_iterations = smooth_iterations
        self.taubin_lambda     = taubin_lambda
        self.taubin_mu         = taubin_mu
        self.log               = log_fn

        self.pcd_in               = None
        self.points_rotated       = None
        self.voxel_centers        = None
        self.remaining_voxel_grid = None
        self.voxel_mesh           = None

        self.load_ply()

    # ── I/O ───────────────────────────────────────────────────────────────────

    def load_ply(self):
        if not os.path.exists(self.ply_path):
            raise FileNotFoundError(f"File not found: {self.ply_path}")
        self.log(f"Loading {self.ply_path}...")
        self.pcd_in         = o3d.io.read_point_cloud(self.ply_path)
        self.points_rotated = np.asarray(self.pcd_in.points)
        self.log(f"  {len(self.points_rotated):,} points loaded.")

    # ── Voxelisation ──────────────────────────────────────────────────────────

    def generate_and_filter_voxels(self):
        rotated = self.points_rotated
        vs      = self.voxel_size
        min_b   = rotated.min(axis=0)
        max_b   = rotated.max(axis=0)

        x = np.arange(min_b[0], max_b[0], vs)
        y = np.arange(min_b[1], max_b[1], vs)
        z = np.arange(min_b[2], max_b[2], vs)
        xx, yy, zz         = np.meshgrid(x, y, z, indexing="ij")
        self.voxel_centers = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

        self.log(f"Filtering {len(self.voxel_centers):,} candidate voxels (CUDA)...")
        mask             = self._filter_voxels_cuda(self.voxel_centers, rotated, vs * 1.5)
        remaining_voxels = self.voxel_centers[~mask]

        pcd_tmp        = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(remaining_voxels)
        self.remaining_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd_tmp, voxel_size=vs)
        self.log(f"  {len(remaining_voxels):,} voxels remaining after filtering.")

    def _filter_voxels_cuda(self, voxel_centers, sphere_centers, radius,
                             chunk_size=100_000):
        n_voxels        = voxel_centers.shape[0]
        keep_mask       = np.ones(n_voxels, dtype=np.bool_)
        radius2         = radius * radius
        d_sphere        = cuda.to_device(np.ascontiguousarray(sphere_centers))
        threads_per_blk = 128

        for i in range(0, n_voxels, chunk_size):
            end     = min(i + chunk_size, n_voxels)
            chunk   = np.ascontiguousarray(voxel_centers[i:end])
            d_chunk = cuda.to_device(chunk)
            d_keep  = cuda.device_array(chunk.shape[0], dtype=np.int8)
            blk     = (chunk.shape[0] + threads_per_blk - 1) // threads_per_blk
            filter_chunk_cuda_kernel[blk, threads_per_blk](
                d_chunk, d_sphere, radius2, d_keep)
            keep_mask[i:end] = d_keep.copy_to_host().astype(np.bool_)
        return keep_mask

    # ── Triangulation ─────────────────────────────────────────────────────────

    def triangulate(self):
        self.log("Triangulating voxels...")
        grid       = self.remaining_voxel_grid
        vs         = grid.voxel_size
        voxel_mesh = o3d.geometry.TriangleMesh()

        for voxel in grid.get_voxels():
            box    = o3d.geometry.TriangleMesh.create_box(vs, vs, vs)
            center = grid.get_voxel_center_coordinate(voxel.grid_index)
            box.translate(center - (vs / 2.0), relative=False)
            n_verts = len(voxel_mesh.vertices)
            voxel_mesh.vertices.extend(box.vertices)
            for tri in box.triangles:
                voxel_mesh.triangles.append(
                    [tri[0] + n_verts, tri[1] + n_verts, tri[2] + n_verts])

        voxel_mesh.merge_close_vertices(1e-6)
        voxel_mesh.compute_vertex_normals()
        voxel_mesh.paint_uniform_color([0.5, 0.5, 0.5])
        self.voxel_mesh = voxel_mesh

    # ── Ray intersection helpers ───────────────────────────────────────────────

    @staticmethod
    def o3d_to_trimesh(o3d_mesh):
        leg = o3d_mesh.to_legacy() if isinstance(
            o3d_mesh, o3d.t.geometry.TriangleMesh) else o3d_mesh
        return trimesh.Trimesh(
            vertices=np.asarray(leg.vertices),
            faces=np.asarray(leg.triangles))

    @staticmethod
    def triangle_normals_to_rays(mesh_in, offset_distance):
        mesh = (mesh_in.to_legacy() if isinstance(mesh_in, o3d.t.geometry.TriangleMesh)
                else mesh_in)
        mesh.compute_triangle_normals()
        verts = np.asarray(mesh.vertices)
        tris  = np.asarray(mesh.triangles)
        norms = np.asarray(mesh.triangle_normals)
        cents = verts[tris].mean(axis=1)
        origs = cents + norms * offset_distance
        return np.hstack((origs, norms))

    @staticmethod
    def extend_normals_cardinal_axes(normals_array):
        N   = normals_array.shape[0]
        ext = np.zeros((N, 18))
        ext[:, :6] = normals_array
        nx, ny, nz = normals_array[:,3], normals_array[:,4], normals_array[:,5]
        cx = (np.abs(nx)==1)[:,np.newaxis]
        cy = (np.abs(ny)==1)[:,np.newaxis]
        cz = (np.abs(nz)==1)[:,np.newaxis]
        ext[:,6:] = np.select(
            [cx, cy, cz],
            [np.array([0,1,0,0,-1,0,0,0,1,0,0,-1]),
             np.array([1,0,0,-1,0,0,0,0,1,0,0,-1]),
             np.array([1,0,0,-1,0,0,0,1,0,0,-1,0])],
            default=np.zeros(12))
        return ext

    def perform_ray_intersection(self):
        self.log("Performing ray-intersection shell analysis...")
        voxel_t = o3d.t.geometry.TriangleMesh.from_legacy(self.voxel_mesh)
        normals  = self.triangle_normals_to_rays(
            self.voxel_mesh, offset_distance=self.voxel_size * 0.25)
        ext      = self.extend_normals_cardinal_axes(normals)
        mesh_    = self.o3d_to_trimesh(voxel_t)

        rays = [
            (ext[:,:3], ext[:,3:6]),
            (ext[:,:3], ext[:,6:9]),
            (ext[:,:3], ext[:,9:12]),
            (ext[:,:3], ext[:,12:15]),
            (ext[:,:3], ext[:,15:]),
        ]
        ray_sets = []
        for origins, dirs in rays:
            _, ir, _ = mesh_.ray.intersects_id(
                origins, dirs, return_locations=True, multiple_hits=False)
            ray_sets.append(ir)
        return reduce(np.intersect1d, ray_sets)

    # ── Generalisation ────────────────────────────────────────────────────────

    def generalize_surface(self, mesh):
        iters = self.smooth_iterations
        self.log(f"Generalising surface (Taubin ×{iters} + plane snap)...")
        o3d_m           = o3d.geometry.TriangleMesh()
        o3d_m.vertices  = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_m.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_m = o3d_m.filter_smooth_taubin(
            number_of_iterations=iters,
            lambda_filter=self.taubin_lambda,
            mu=self.taubin_mu)

        verts    = np.asarray(o3d_m.vertices).copy()
        faces    = np.asarray(o3d_m.triangles)
        smoothed = trimesh.Trimesh(vertices=verts, faces=faces)

        v_to_faces = [set() for _ in range(len(verts))]
        for fi, f in enumerate(faces):
            for v in f:
                v_to_faces[v].add(fi)

        for facet in smoothed.facets:
            if len(facet) < 2:
                continue
            facet_set      = set(facet.tolist())
            facet_vert_idx = np.unique(faces[facet])
            pts            = verts[facet_vert_idx]
            centroid       = pts.mean(axis=0)
            _, _, Vt       = np.linalg.svd(pts - centroid, full_matrices=False)
            plane_n        = Vt[-1]
            for vi in facet_vert_idx:
                if v_to_faces[vi].issubset(facet_set):
                    p = verts[vi]
                    verts[vi] = p - np.dot(p - centroid, plane_n) * plane_n

        out = trimesh.Trimesh(vertices=verts, faces=faces)
        trimesh.repair.fix_winding(out)
        self.log(f"  Generalisation done | watertight: {out.is_watertight}")
        return out

    # ── Decimation ────────────────────────────────────────────────────────────

    def decimate(self, mesh):
        pct = self.decimate_percent
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_inversion(mesh)
            trimesh.repair.fix_winding(mesh)

        n_before     = len(mesh.faces)
        target_faces = max(int(n_before * pct), 4)
        self.log(f"Decimating: {n_before:,} → {target_faces:,} ({pct*100:.0f}%)")

        o3d_m           = o3d.geometry.TriangleMesh()
        o3d_m.vertices  = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_m.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_m.compute_vertex_normals()
        o3d_s = o3d_m.simplify_quadric_decimation(target_faces)
        out   = trimesh.Trimesh(
            vertices=np.asarray(o3d_s.vertices),
            faces=np.asarray(o3d_s.triangles))

        if not out.is_watertight:
            trimesh.repair.fill_holes(out)
            trimesh.repair.fix_inversion(out)
            trimesh.repair.fix_winding(out)

        self.log(f"  Decimation done | faces: {len(out.faces):,} | "
                 f"watertight: {out.is_watertight}")
        return out

    # ── Master pipeline ───────────────────────────────────────────────────────

    def execute(self):
        """Run the full pipeline and return the final Trimesh."""
        self.generate_and_filter_voxels()
        self.triangulate()

        self.voxel_mesh.compute_triangle_normals()
        internal = self.perform_ray_intersection()
        self.log(f"  Internal faces identified: {len(internal):,}")
        self.voxel_mesh.remove_triangles_by_index(internal)
        self.voxel_mesh.remove_unreferenced_vertices()

        self.log("Converting voxel mesh to Trimesh...")
        tm = self.o3d_to_trimesh(self.voxel_mesh)
        tm.merge_vertices(merge_tex=True, merge_norm=True)
        tm.update_faces(tm.nondegenerate_faces())
        tm.update_faces(tm.unique_faces())
        trimesh.repair.fix_inversion(tm)
        trimesh.repair.fix_winding(tm)
        trimesh.repair.fill_holes(tm)

        tm = self.decimate(tm)
        tm = self.generalize_surface(tm)
        tm = self.decimate(tm)

        self.log("-" * 40)
        self.log("FINAL MESH STATUS:")
        self.log(f"  Vertices  : {len(tm.vertices):,}")
        self.log(f"  Faces     : {len(tm.faces):,}")
        self.log(f"  Watertight: {tm.is_watertight}")
        if tm.is_watertight:
            self.log(f"  Volume    : {tm.volume:.4f}")
        self.log("-" * 40)
        return tm


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 ─ HEALING STAGE  (OBJ → ray-cast → temp PLY)
# ══════════════════════════════════════════════════════════════════════════════

def run_healing_stage(obj_path, voxel_size, log_fn=print):
    """
    Execute the Healing pre-stage (untitled4.py logic).

    Loads an OBJ mesh, voxelises it with the shared voxel_size, casts rays in
    +Z then +X/+Y, and writes the surviving support points to a temporary PLY
    file next to the input.  That PLY is then consumed by LotusMesh.

    Parameters
    ----------
    obj_path   : str   – input OBJ file
    voxel_size : float – shared voxel edge length (metres)
    log_fn     : callable(str)

    Returns
    -------
    str  – path to the intermediate PLY point cloud
    """
    # 1. Load OBJ via trimesh and convert to Open3D, translate to origin
    log_fn(f"[Healing] Loading OBJ: {obj_path}")
    t_mesh = trimesh.load(obj_path, force='mesh')

    mesh           = o3d.geometry.TriangleMesh()
    translated_v   = t_mesh.vertices - t_mesh.vertices.min(axis=0)
    mesh.vertices  = o3d.utility.Vector3dVector(translated_v)
    mesh.triangles = o3d.utility.Vector3iVector(t_mesh.faces)
    mesh.compute_vertex_normals()
    log_fn(f"[Healing]   {len(t_mesh.vertices):,} vertices, "
           f"{len(t_mesh.faces):,} faces loaded.")

    # 2. Voxelise and build a box-mesh (identical approach to LotusMesh)
    vs         = voxel_size
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=vs)
    voxels     = voxel_grid.get_voxels()
    origin     = voxel_grid.origin
    log_fn(f"[Healing] Voxelising → {len(voxels)} voxels (size={vs} m).")

    combined = o3d.geometry.TriangleMesh()
    for v in voxels:
        cube      = o3d.geometry.TriangleMesh.create_box(vs, vs, vs)
        world_pos = origin + v.grid_index.astype(float) * vs
        cube.translate(world_pos)
        combined += cube

    combined.merge_close_vertices(1e-6)
    combined.compute_vertex_normals()

    # 3. Ray-cast: +Z pass, then +X or +Y pass
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(combined)
    scene  = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    aabb  = combined.get_axis_aligned_bounding_box()
    min_b = aabb.min_bound
    max_b = aabb.max_bound

    xs = np.arange(min_b[0], max_b[0], vs)
    ys = np.arange(min_b[1], max_b[1], vs)
    zs = np.arange(min_b[2], max_b[2], vs)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    points     = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    n_points   = len(points)
    log_fn(f"[Healing] Ray-casting +Z over {n_points:,} grid points...")

    dirs_z     = np.tile([0, 0, 1], (n_points, 1))
    hit_z      = scene.cast_rays(
        o3c.Tensor(np.concatenate([points, dirs_z], axis=1),
                   dtype=o3c.float32))['t_hit'].isfinite().numpy()
    candidates = points[hit_z]
    log_fn(f"[Healing]   +Z pass: {len(candidates):,} / {n_points:,}")

    if len(candidates) == 0:
        raise RuntimeError(
            "No candidates after +Z pass. Check mesh normals or orientation.")

    n_cand = len(candidates)
    dirs_x = np.tile([1, 0, 0], (n_cand, 1))
    dirs_y = np.tile([0, 1, 0], (n_cand, 1))

    hit_x = scene.cast_rays(
        o3c.Tensor(np.concatenate([candidates, dirs_x], axis=1),
                   dtype=o3c.float32))['t_hit'].isfinite().numpy()
    hit_y = scene.cast_rays(
        o3c.Tensor(np.concatenate([candidates, dirs_y], axis=1),
                   dtype=o3c.float32))['t_hit'].isfinite().numpy()

    valid_points = candidates[hit_x | hit_y]
    log_fn(f"[Healing]   +X or +Y pass: {len(valid_points):,} / {n_cand:,}")

    if len(valid_points) == 0:
        raise RuntimeError("No support points survived the +X/+Y pass.")

    # 4. Write intermediate PLY next to the input OBJ
    temp_ply   = os.path.splitext(obj_path)[0] + "_healing_temp.ply"
    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.paint_uniform_color([0.2, 0.8, 0.2])
    o3d.io.write_point_cloud(temp_ply, pcd, write_ascii=True)
    log_fn(f"[Healing] Intermediate PLY → {temp_ply} ({len(valid_points):,} pts)")

    return temp_ply


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 ─ OBJ → CityJSON CONVERTER
# ══════════════════════════════════════════════════════════════════════════════

def _load_obj_direct(filepath):
    verts, faces = [], []
    with open(filepath, 'r') as fh:
        for line in fh:
            if line.startswith('v '):
                p = line.strip().split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('f '):
                p    = line.strip().split()[1:]
                face = [int(x.split('/')[0]) - 1 for x in p]
                if len(face) >= 3:
                    faces.append(face)
    return np.array(verts), faces


def _generate_cityjson_id():
    return f"GUID_{uuid.uuid4()}"


def _refine_ground_vs_ceiling(candidates, semantics_list,
                               gap_threshold=0.5, log_fn=print):
    if not candidates:
        return
    dtype    = [('index', int), ('z', float)]
    data     = np.array(candidates, dtype=dtype)
    sorted_d = np.sort(data, order='z')
    z_vals   = sorted_d['z']
    z_diffs  = np.diff(z_vals)
    if len(z_diffs) == 0:
        return
    max_gap_i = np.argmax(z_diffs)
    max_gap   = z_diffs[max_gap_i]
    if max_gap > gap_threshold:
        split_z  = (z_vals[max_gap_i] + z_vals[max_gap_i + 1]) / 2.0
        ceil_idx = sorted_d[sorted_d['z'] > split_z]['index']
        for idx in ceil_idx:
            semantics_list[idx] = "OuterCeiling"
        log_fn(f"   [Clustering] Gap {max_gap:.2f}m at Z={split_z:.2f}; "
               f"{len(ceil_idx)} faces → OuterCeiling.")
    else:
        log_fn(f"   [Clustering] No significant gap (max {max_gap:.2f}m); "
               "all kept as GroundSurface.")


def _build_face_adjacency(faces):
    edge_to_faces = {}
    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            edge = tuple(sorted((face[i], face[(i + 1) % n])))
            edge_to_faces.setdefault(edge, []).append(fi)
    adjacency = {i: set() for i in range(len(faces))}
    for nbrs in edge_to_faces.values():
        for a in nbrs:
            for b in nbrs:
                if a != b:
                    adjacency[a].add(b)
    return adjacency


def _reclassify_surrounded_by_roof(faces, semantics_list, log_fn=print):
    TARGETS   = {"WallSurface", "OuterCeiling"}
    ROOF      = "RoofSurface"
    adjacency = _build_face_adjacency(faces)
    original  = list(semantics_list)

    changed = 0
    for fi, st in enumerate(original):
        if st not in TARGETS:
            continue
        nbrs = adjacency[fi]
        if not nbrs:
            continue
        if sum(1 for nb in nbrs if original[nb] == ROOF) >= 2:
            semantics_list[fi] = ROOF
            changed += 1
    log_fn(f"   [Neighbour] {changed} faces → RoofSurface.")

    original   = list(semantics_list)
    floor_chg  = 0
    ground_chg = 0
    for fi, st in enumerate(original):
        nbrs = adjacency[fi]
        if not nbrs:
            continue
        wc = sum(1 for nb in nbrs if original[nb] == "WallSurface")
        if st == "OuterFloor" and wc >= 2:
            semantics_list[fi] = "WallSurface"; floor_chg  += 1
        elif st == "GroundSurface" and wc >= 2:
            semantics_list[fi] = "WallSurface"; ground_chg += 1
    log_fn(f"   [Neighbour] {floor_chg} OuterFloor + "
           f"{ground_chg} GroundSurface → WallSurface.")


def obj_to_cityjson(obj_path, output_path, epsg_code="7415", log_fn=print):
    """Convert an OBJ file to a CityJSON v2.0 file with LOD2 semantics."""
    log_fn("Loading OBJ for CityJSON conversion...")
    verts, faces = _load_obj_direct(obj_path)
    if len(verts) == 0:
        log_fn("Error: No vertices found in OBJ.")
        return

    log_fn(f"  {len(faces):,} faces loaded.")
    min_v  = np.min(verts, axis=0)
    max_v  = np.max(verts, axis=0)
    extent = [float(min_v[0]), float(min_v[1]), float(min_v[2]),
              float(max_v[0]), float(max_v[1]), float(max_v[2])]

    threshold           = np.sin(np.radians(5))
    surfaces_geometry   = []
    surfaces_semantics  = []
    downward_candidates = []

    for face in faces:
        p0, p1, p2 = verts[face[0]], verts[face[1]], verts[face[2]]
        n           = np.cross(p1 - p0, p2 - p0)
        ln          = np.linalg.norm(n)
        if ln < 1e-6:
            continue
        n_unit = n / ln
        cz     = (p0[2] + p1[2] + p2[2]) / 3.0

        if abs(n_unit[2]) <= threshold:
            st = "WallSurface"
        elif n_unit[2] > threshold:
            st = "RoofSurface"
        else:
            st = "GroundSurface"
            downward_candidates.append((len(surfaces_semantics), cz))

        surfaces_geometry.append([face])
        surfaces_semantics.append(st)

    log_fn(f"Analysing {len(downward_candidates)} downward surfaces...")
    _refine_ground_vs_ceiling(downward_candidates, surfaces_semantics, log_fn=log_fn)
    log_fn("Running neighbour-based reclassification...")
    _reclassify_surrounded_by_roof(faces, surfaces_semantics, log_fn=log_fn)

    unique_types    = sorted(set(surfaces_semantics))
    type_map        = {t: i for i, t in enumerate(unique_types)}
    semantic_values = [type_map[t] for t in surfaces_semantics]
    building_id     = _generate_cityjson_id()
    height          = float(max_v[2] - min_v[2])
    epsg_uri        = f"https://www.opengis.net/def/crs/EPSG/0/{epsg_code}"

    cityjson = {
        "type"   : "CityJSON",
        "version": "2.0",
        "metadata": {
            "geographicalExtent": extent,
            "referenceSystem"   : epsg_uri,
        },
        "vertices": verts.tolist(),
        "CityObjects": {
            building_id: {
                "type": "Building",
                "attributes": {
                    "measuredHeight"    : round(height, 3),
                    "roofType"          : "1000",
                    "storeysAboveGround": 1,
                },
                "geometry": [{
                    "type"      : "Solid",
                    "lod"       : "2",
                    "boundaries": [surfaces_geometry],
                    "semantics" : {
                        "surfaces": [{"type": t} for t in unique_types],
                        "values"  : [semantic_values],
                    },
                }],
            }
        },
    }

    with open(output_path, 'w') as fh:
        json.dump(cityjson, fh, separators=(',', ':'))
    log_fn(f"CityJSON written → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 ─ GUI
# ══════════════════════════════════════════════════════════════════════════════

class PipelineGUI(tk.Tk):
    """
    Tkinter front-end for both RobustLotus pipelines.

    The mode selector switches only the INPUT row between an OBJ browser
    (Healing) and a PLY browser (Reconstruction).  Every other control —
    voxel size, mesh parameters, output format, EPSG, and output path — is
    shared and always visible, because both modes feed into the same
    LotusMesh → OBJ / CityJSON chain.
    """

    BG       = "#1e1e2e"
    FG       = "#cdd6f4"
    ACCENT   = "#89b4fa"
    ENTRY_BG = "#313244"
    BTN_BG   = "#45475a"
    BTN_FG   = "#cdd6f4"
    LOG_BG   = "#11111b"
    LOG_FG   = "#a6e3a1"
    FONT     = ("Segoe UI", 10)
    FONT_B   = ("Segoe UI", 10, "bold")
    FONT_H   = ("Segoe UI", 11, "bold")

    def __init__(self):
        super().__init__()
        self.title("RobustLotus Pipeline")
        self.configure(bg=self.BG)
        self.resizable(True, True)
        self.minsize(700, 860)

        self._log_queue = queue.Queue()
        self._build_ui()
        self._poll_log()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=12, pady=5)

        # Header
        tk.Label(self, text="  RobustLotus  –  Unified Pipeline",
                 font=("Segoe UI", 14, "bold"),
                 bg=self.BG, fg=self.ACCENT, anchor="w"
                 ).pack(fill="x", padx=12, pady=(14, 4))
        ttk.Separator(self).pack(fill="x", padx=12, pady=4)

        # ── Mode selector ─────────────────────────────────────────────────────
        self._section("Pipeline Mode")
        mode_frame = tk.Frame(self, bg=self.BG)
        mode_frame.pack(fill="x", **pad)

        self._mode_var = tk.StringVar(value="healing")
        modes = [
            ("healing",
             "Healing",
             "OBJ  →  ray-cast points  →  LotusMesh  →  OBJ / CityJSON"),
            ("reconstruction",
             "Reconstruction",
             "PLY point cloud  →  LotusMesh  →  OBJ / CityJSON"),
        ]
        for val, label, tip in modes:
            col = tk.Frame(mode_frame, bg=self.BG)
            col.pack(side="left", padx=(0, 28))
            tk.Radiobutton(col, text=label, variable=self._mode_var, value=val,
                           command=self._on_mode_change,
                           bg=self.BG, fg=self.FG, selectcolor=self.ENTRY_BG,
                           activebackground=self.BG, font=self.FONT_B
                           ).pack(anchor="w")
            tk.Label(col, text=tip, font=("Segoe UI", 8),
                     bg=self.BG, fg="#6c7086").pack(anchor="w")

        ttk.Separator(self).pack(fill="x", padx=12, pady=6)

        # ── Input row (label + path swap on mode change) ───────────────────────
        self._section("Input File")
        self._input_label_var = tk.StringVar(value="Mesh File (.obj)")
        tk.Label(self, textvariable=self._input_label_var,
                 font=self.FONT, bg=self.BG, fg=self.FG, anchor="w"
                 ).pack(fill="x", padx=16)

        input_row = tk.Frame(self, bg=self.BG)
        input_row.pack(fill="x", **pad)
        self._input_var = tk.StringVar()
        tk.Entry(input_row, textvariable=self._input_var, bg=self.ENTRY_BG,
                 fg=self.FG, font=self.FONT, insertbackground=self.FG,
                 relief="flat").pack(side="left", fill="x", expand=True)
        tk.Button(input_row, text="Browse…", command=self._browse_input,
                  bg=self.ACCENT, fg="#1e1e2e", font=self.FONT_B,
                  relief="flat", cursor="hand2"
                  ).pack(side="left", padx=(6, 0))

        ttk.Separator(self).pack(fill="x", padx=12, pady=6)

        # ── Shared mesh parameters ────────────────────────────────────────────
        self._section("Mesh Parameters")
        params_frame = tk.Frame(self, bg=self.BG)
        params_frame.pack(fill="x", padx=12, pady=2)

        self._voxel_var    = tk.DoubleVar(value=1.0)
        self._decimate_var = tk.DoubleVar(value=25.0)
        self._smooth_var   = tk.IntVar(value=10)
        self._lambda_var   = tk.DoubleVar(value=0.20)
        self._mu_var       = tk.DoubleVar(value=-0.22)

        self._param_row(params_frame, 0, "Voxel Size (m):",
                        self._voxel_var,    0.05,  10.0,  0.05)
        self._param_row(params_frame, 1, "Decimation Target (%):",
                        self._decimate_var, 1,     100,   1)
        self._param_row(params_frame, 2, "Smooth Iterations:",
                        self._smooth_var,   1,     100,   1)
        self._param_row(params_frame, 3, "Taubin λ (shrink):",
                        self._lambda_var,   0.01,  0.99,  0.01)
        self._param_row(params_frame, 4, "Taubin μ (anti-shrink):",
                        self._mu_var,      -0.99, -0.01,  0.01)

        ttk.Separator(self).pack(fill="x", padx=12, pady=6)

        # ── Shared output settings ────────────────────────────────────────────
        self._section("Output Settings")

        fmt_frame = tk.Frame(self, bg=self.BG)
        fmt_frame.pack(fill="x", **pad)
        tk.Label(fmt_frame, text="Output Format:", font=self.FONT_B,
                 bg=self.BG, fg=self.FG).pack(side="left")
        self._fmt_var = tk.StringVar(value="cityjson")
        for val, txt in [("obj", "OBJ only"), ("cityjson", "CityJSON (LOD2)")]:
            tk.Radiobutton(fmt_frame, text=txt, variable=self._fmt_var,
                           value=val, command=self._on_fmt_change,
                           bg=self.BG, fg=self.FG, selectcolor=self.ENTRY_BG,
                           activebackground=self.BG, font=self.FONT
                           ).pack(side="left", padx=10)

        # EPSG (visible for CityJSON only)
        self._epsg_frame = tk.Frame(self, bg=self.BG)
        self._epsg_frame.pack(fill="x", **pad)
        tk.Label(self._epsg_frame, text="CRS EPSG Code:", font=self.FONT_B,
                 bg=self.BG, fg=self.FG).pack(side="left")
        self._epsg_var = tk.StringVar(value="7415")
        tk.Entry(self._epsg_frame, textvariable=self._epsg_var, width=10,
                 bg=self.ENTRY_BG, fg=self.FG, font=self.FONT,
                 insertbackground=self.FG, relief="flat"
                 ).pack(side="left", padx=6)
        tk.Label(self._epsg_frame,
                 text="(e.g. 7415 = RD New + NAP, 4326 = WGS84)",
                 font=("Segoe UI", 9), bg=self.BG, fg="#6c7086"
                 ).pack(side="left")

        # Output path
        out_row = tk.Frame(self, bg=self.BG)
        out_row.pack(fill="x", **pad)
        tk.Label(out_row, text="Output Path:", font=self.FONT_B,
                 bg=self.BG, fg=self.FG).pack(side="left")
        self._out_var = tk.StringVar()
        tk.Entry(out_row, textvariable=self._out_var, bg=self.ENTRY_BG,
                 fg=self.FG, font=self.FONT, insertbackground=self.FG,
                 relief="flat").pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(out_row, text="Browse…", command=self._browse_out,
                  bg=self.BTN_BG, fg=self.BTN_FG, font=self.FONT,
                  relief="flat", cursor="hand2"
                  ).pack(side="left")

        ttk.Separator(self).pack(fill="x", padx=12, pady=6)

        # ── Run button ────────────────────────────────────────────────────────
        self._run_btn = tk.Button(
            self, text="▶  RUN PIPELINE",
            command=self._run,
            bg=self.ACCENT, fg="#1e1e2e",
            font=("Segoe UI", 12, "bold"),
            relief="flat", cursor="hand2",
            padx=20, pady=8)
        self._run_btn.pack(pady=8)

        # ── Log console ───────────────────────────────────────────────────────
        ttk.Separator(self).pack(fill="x", padx=12, pady=4)
        tk.Label(self, text="Pipeline Log", font=self.FONT_B,
                 bg=self.BG, fg=self.FG, anchor="w"
                 ).pack(fill="x", padx=14)
        self._log_box = scrolledtext.ScrolledText(
            self, bg=self.LOG_BG, fg=self.LOG_FG,
            font=("Consolas", 9), relief="flat",
            state="disabled", wrap="word")
        self._log_box.pack(fill="both", expand=True, padx=12, pady=(2, 12))

    # ── Layout helpers ────────────────────────────────────────────────────────

    def _section(self, title):
        tk.Label(self, text=title, font=self.FONT_H,
                 bg=self.BG, fg=self.ACCENT, anchor="w"
                 ).pack(fill="x", padx=14, pady=(6, 0))

    def _param_row(self, parent, row, label, var, from_, to, increment):
        tk.Label(parent, text=label, font=self.FONT, bg=self.BG, fg=self.FG,
                 anchor="w").grid(row=row, column=0, sticky="w", pady=3)
        tk.Spinbox(parent, textvariable=var,
                   from_=from_, to=to, increment=increment,
                   bg=self.ENTRY_BG, fg=self.FG, font=self.FONT,
                   buttonbackground=self.BTN_BG, relief="flat",
                   width=8, insertbackground=self.FG
                   ).grid(row=row, column=1, sticky="w", padx=(8, 24), pady=3)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_mode_change(self):
        """Swap input label and clear stale paths when switching modes."""
        if self._mode_var.get() == "healing":
            self._input_label_var.set("Mesh File (.obj)")
        else:
            self._input_label_var.set("Point Cloud (.ply)")
        self._input_var.set("")
        self._out_var.set("")

    def _on_fmt_change(self):
        vis = self._fmt_var.get() == "cityjson"
        for child in self._epsg_frame.winfo_children():
            child.configure(state="normal" if vis else "disabled")
        inp = self._input_var.get().strip()
        if inp:
            self._auto_output_path(inp)

    def _browse_input(self):
        if self._mode_var.get() == "healing":
            p = filedialog.askopenfilename(
                title="Select input OBJ mesh",
                filetypes=[("OBJ mesh", "*.obj"), ("All files", "*.*")])
        else:
            p = filedialog.askopenfilename(
                title="Select input PLY point cloud",
                filetypes=[("Point Cloud", "*.ply"), ("All files", "*.*")])
        if p:
            self._input_var.set(p)
            self._auto_output_path(p)

    def _browse_out(self):
        fmt = self._fmt_var.get()
        if fmt == "cityjson":
            ft, defext = ([("CityJSON", "*.city.json *.json"),
                            ("All files", "*.*")], ".city.json")
        else:
            ft, defext = ([("OBJ mesh", "*.obj"), ("All files", "*.*")], ".obj")
        p = filedialog.asksaveasfilename(
            title="Save output as…", filetypes=ft, defaultextension=defext)
        if p:
            self._out_var.set(p)

    def _auto_output_path(self, input_path):
        """Derive an output path from the input stem and chosen format."""
        base = os.path.splitext(input_path)[0]
        for compound in (".city.json", ".json"):
            if base.lower().endswith(compound):
                base = base[: -len(compound)]
                break
        ext = ".city.json" if self._fmt_var.get() == "cityjson" else ".obj"
        self._out_var.set(base + ext)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg):
        self._log_queue.put(str(msg))

    def _poll_log(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._log_box.configure(state="normal")
                self._log_box.insert("end", msg + "\n")
                self._log_box.see("end")
                self._log_box.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(80, self._poll_log)

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        input_path = self._input_var.get().strip()
        out_path   = self._out_var.get().strip()

        if not input_path:
            messagebox.showerror("Missing input", "Please select an input file.")
            return
        if not os.path.isfile(input_path):
            messagebox.showerror("File not found", f"Cannot find:\n{input_path}")
            return
        if not out_path:
            messagebox.showerror("Missing output", "Please specify an output path.")
            return

        mode          = self._mode_var.get()
        fmt           = self._fmt_var.get()
        epsg          = self._epsg_var.get().strip() or "7415"
        voxel_size    = float(self._voxel_var.get())
        decimate_pc   = float(self._decimate_var.get()) / 100.0
        smooth_iter   = int(self._smooth_var.get())
        taubin_lambda = float(self._lambda_var.get())
        taubin_mu     = float(self._mu_var.get())

        self._run_btn.configure(state="disabled", text="⏳ Running…")
        threading.Thread(
            target=self._pipeline_thread,
            args=(mode, input_path, out_path, fmt, epsg,
                  voxel_size, decimate_pc, smooth_iter,
                  taubin_lambda, taubin_mu),
            daemon=True).start()

    def _pipeline_thread(self, mode, input_path, out_path, fmt, epsg,
                         voxel_size, decimate_pc, smooth_iter,
                         taubin_lambda, taubin_mu):
        try:
            label = "HEALING" if mode == "healing" else "RECONSTRUCTION"
            self._log("=" * 52)
            self._log(f"{label} PIPELINE STARTED")
            self._log(f"  Input        : {input_path}")
            self._log(f"  Output       : {out_path}")
            self._log(f"  Format       : {fmt.upper()}")
            self._log(f"  Voxel size   : {voxel_size} m")
            self._log(f"  Decimation   : {decimate_pc*100:.0f}%")
            self._log(f"  Smooth iters : {smooth_iter}")
            self._log(f"  Taubin λ     : {taubin_lambda:.3f}")
            self._log(f"  Taubin μ     : {taubin_mu:.3f}")
            if fmt == "cityjson":
                self._log(f"  EPSG         : {epsg}")
            self._log("=" * 52)

            # ── Stage 1 (Healing only): OBJ → ray-cast → intermediate PLY ────
            if mode == "healing":
                ply_path = run_healing_stage(
                    input_path, voxel_size, log_fn=self._log)
            else:
                ply_path = input_path       # PLY supplied directly

            # ── Stage 2: LotusMesh  (shared between both modes) ───────────────
            processor  = LotusMesh(
                ply_path,
                voxel_size        = voxel_size,
                decimate_percent  = decimate_pc,
                smooth_iterations = smooth_iter,
                taubin_lambda     = taubin_lambda,
                taubin_mu         = taubin_mu,
                log_fn            = self._log,
            )
            final_mesh = processor.execute()

            # ── Stage 3: Export OBJ ───────────────────────────────────────────
            if fmt == "obj":
                obj_path = out_path
            else:
                base = out_path
                for compound in (".city.json", ".json", ".obj"):
                    if base.lower().endswith(compound):
                        base = base[: -len(compound)]
                        break
                obj_path = base + "_mesh.obj"

            self._log(f"Exporting OBJ → {obj_path}")
            final_mesh.export(obj_path, file_type="obj")

            # ── Stage 4: CityJSON (optional) ──────────────────────────────────
            if fmt == "cityjson":
                obj_to_cityjson(obj_path, out_path,
                                epsg_code=epsg, log_fn=self._log)

            self._log("=" * 52)
            self._log("✅ PIPELINE COMPLETE")
            self._log(f"   Output saved to: {out_path}")
            self._log("=" * 52)

            self.after(0, lambda: messagebox.showinfo(
                "Done", f"Pipeline finished.\n\nOutput:\n{out_path}"))

        except Exception as exc:
            import traceback
            self._log("\n❌ ERROR:")
            self._log(traceback.format_exc())
            self.after(0, lambda e=exc: messagebox.showerror(
                "Pipeline Error", str(e)))

        finally:
            self.after(0, lambda: self._run_btn.configure(
                state="normal", text="▶  RUN PIPELINE"))


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = PipelineGUI()
    app.mainloop()