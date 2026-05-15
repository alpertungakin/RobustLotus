#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 23:46:58 2025

@author: alper
"""

import open3d as o3d
import numpy as np
import numba as nb
from numba import jit, prange, cuda
import trimesh
from functools import reduce
import os

# --- Global Numba Function (Must stay at module level for JIT) ---
@cuda.jit
def filter_chunk_cuda_kernel(vox_chunk, sphere_centers, radius2, keep_out):
    """
    CUDA Kernel to filter voxels.
    Each thread processes one voxel and checks it against all sphere centers.
    """
    # Calculate the unique thread index
    idx = cuda.grid(1)

    # Boundary check: Ensure we don't access out of bounds
    if idx < vox_chunk.shape[0]:
        # Read voxel coordinates (Global Memory)
        vx = vox_chunk[idx, 0]
        vy = vox_chunk[idx, 1]
        vz = vox_chunk[idx, 2]

        n_sph = sphere_centers.shape[0]
        is_kept = 1 # Assume true initially
        
        # Loop through all sphere centers
        for j in range(n_sph):
            sx = sphere_centers[j, 0]
            sy = sphere_centers[j, 1]
            sz = sphere_centers[j, 2]

            dx = vx - sx
            dy = vy - sy
            dz = vz - sz
            
            d2 = dx*dx + dy*dy + dz*dz

            # Optimization: We don't need min_d2. 
            # If ANY point is within radius, we discard the voxel.
            if d2 <= radius2:
                is_kept = 0
                break # Early exit
        
        # Write result to global memory
        keep_out[idx] = is_kept

# --- Main Class Structure ---

class LotusMesh:
    def __init__(self, ply_path, voxel_size=0.1):
        self.voxel_size = voxel_size
        self.ply_path = ply_path
        
        # State variables
        self.pcd_in = None
        self.points_rotated = None # Using this to match 'rotated' variable in snippet
        self.voxel_centers = None
        self.remaining_voxel_grid = None
        self.voxel_mesh = None
        
        # Load data immediately
        self.load_ply()

    def load_ply(self):
        """Reads the PLY file as requested."""
        if not os.path.exists(self.ply_path):
            raise FileNotFoundError(f"File not found: {self.ply_path}")
        
        print(f"Loading {self.ply_path}...")
        self.pcd_in = o3d.io.read_point_cloud(self.ply_path)
        # Assuming the data in the PLY is already the 'rotated' data you wanted
        # If your previous script rotated it manually, that logic would go here.
        # For now, we treat the input points as 'rotated'.
        self.points_rotated = np.asarray(self.pcd_in.points)

    def generate_and_filter_voxels(self):
        # Step 1: Compute bounding box
        rotated = self.points_rotated
        voxel_size = self.voxel_size

        min_bound = rotated.min(axis=0)
        max_bound = rotated.max(axis=0)

        # Step 2: Fill bounding box with voxels
        x = np.arange(min_bound[0], max_bound[0], voxel_size)
        y = np.arange(min_bound[1], max_bound[1], voxel_size)
        z = np.arange(min_bound[2], max_bound[2], voxel_size)

        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        self.voxel_centers = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

        # Step 3: Filter voxels
        sphere_radius = voxel_size * 1.5
        sphere_centers = rotated

        print("Filtering voxels (JIT)...")
        mask = self._filter_voxels_outside_spheres_jit(self.voxel_centers, sphere_centers, sphere_radius)
        remaining_voxels = self.voxel_centers[~mask]

        # Create VoxelGrid object
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(remaining_voxels)
        self.remaining_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd_temp,
            voxel_size=voxel_size
        )
        
    def _filter_voxels_outside_spheres_jit(self, voxel_centers, sphere_centers, radius, chunk_size=None):
            n_voxels = voxel_centers.shape[0]
            keep_mask = np.ones(n_voxels, dtype=np.bool_)
            radius2 = radius * radius
    
            # 1. Ensure sphere_centers is contiguous too, just in case
            d_sphere_centers = cuda.to_device(np.ascontiguousarray(sphere_centers))
    
            threads_per_block = 128
            if chunk_size is None: 
                chunk_size = 100000 
    
            print(f"Running CUDA Kernel on {n_voxels} voxels...")
            
            for i in range(0, n_voxels, chunk_size):
                end = min(i + chunk_size, n_voxels)
                
                # Slice the host array
                vox_chunk = voxel_centers[i:end]
                
                # FIX: Force contiguous memory layout before transfer
                d_vox_chunk = cuda.to_device(np.ascontiguousarray(vox_chunk))
                
                # Allocate output array on device
                d_keep_chunk = cuda.device_array(vox_chunk.shape[0], dtype=np.int8)
                
                blocks_per_grid = (vox_chunk.shape[0] + (threads_per_block - 1)) // threads_per_block
                
                filter_chunk_cuda_kernel[blocks_per_grid, threads_per_block](
                    d_vox_chunk, 
                    d_sphere_centers, 
                    radius2, 
                    d_keep_chunk
                )
                
                keep_mask[i:end] = d_keep_chunk.copy_to_host().astype(np.bool_)
    
            return keep_mask

    def triangulate(self):
        print("Triangulating voxels...")
        # Exact logic from your 'triangulateVoxels' function
        remaining_voxel_grid = self.remaining_voxel_grid
        
        voxel_mesh = o3d.geometry.TriangleMesh()
        
        # Get all voxels from the grid
        voxels = remaining_voxel_grid.get_voxels()
        voxel_size = remaining_voxel_grid.voxel_size
        
        # For each voxel, create a box and add it to the mesh
        for voxel in voxels:
            # Create a box primitive
            box = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
            
            # Get the center coordinate of the voxel
            center = remaining_voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
            
            # Position the box at the voxel's center
            box.translate(center - (voxel_size / 2.0), relative=False)
            
            # Add the box's triangles to the main mesh
            # Get current triangle and vertex counts to correctly offset indices
            num_vertices = len(voxel_mesh.vertices)
            
            # Add the new box's vertices to the mesh
            voxel_mesh.vertices.extend(box.vertices)
            
            # Add the new box's triangles, offsetting the vertex indices
            for triangle in box.triangles:
                new_triangle = [
                    triangle[0] + num_vertices,
                    triangle[1] + num_vertices,
                    triangle[2] + num_vertices,
                ]
                voxel_mesh.triangles.append(new_triangle)
                
        # Clean up the mesh and set a consistent color
        voxel_mesh.merge_close_vertices(1e-6)
        voxel_mesh.compute_vertex_normals()
        voxel_mesh.paint_uniform_color([0.5, 0.5, 0.5])

        self.voxel_mesh = voxel_mesh
        return voxel_mesh

    # --- Static / Utility Methods from snippet ---

    @staticmethod
    def o3d_to_trimesh(o3d_mesh):
        if isinstance(o3d_mesh, o3d.t.geometry.TriangleMesh):
            o3d_mesh_legacy = o3d_mesh.to_legacy()
        elif isinstance(o3d_mesh, o3d.geometry.TriangleMesh):
            o3d_mesh_legacy = o3d_mesh
        else:
            raise TypeError(f"Beklenmeyen Open3D mesh tipi: {type(o3d_mesh)}")

        try:
            vertices = np.asarray(o3d_mesh_legacy.vertices)
            faces = np.asarray(o3d_mesh_legacy.triangles)
            if len(vertices) == 0:
                pass
            return trimesh.Trimesh(vertices=vertices, faces=faces)
        except Exception as e:
            print(f"HATA: Legacy mesh Trimesh'e dönüştürülürken hata: {e}")
            raise e

    @staticmethod
    def triangle_normals_to_rays(mesh_in, offset_distance):
        if isinstance(mesh_in, o3d.t.geometry.TriangleMesh):
            mesh = mesh_in.to_legacy()
        elif isinstance(mesh_in, o3d.geometry.TriangleMesh):
            mesh = mesh_in
        else:
            raise TypeError(f"Beklenmeyen mesh tipi: {type(mesh_in)}")

        mesh.compute_triangle_normals()
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.triangle_normals) 
        
        tri_vertices = vertices[triangles]
        ray_centers = np.mean(tri_vertices, axis=1) 
        ray_origins = ray_centers + (normals * offset_distance) 
        ray_directions = normals 
        rays_array = np.hstack((ray_origins, ray_directions))
        return rays_array

    @staticmethod
    def extend_normals_cardinal_axes(normals_array):
        N = normals_array.shape[0]
        extended_array = np.zeros((N, 18))
        extended_array[:, :6] = normals_array
        
        nx = normals_array[:, 3]
        ny = normals_array[:, 4]
        nz = normals_array[:, 5]

        cond_x_axis = (np.abs(nx) == 1)[:, np.newaxis]
        cond_y_axis = (np.abs(ny) == 1)[:, np.newaxis]
        cond_z_axis = (np.abs(nz) == 1)[:, np.newaxis]
        
        choice_x = np.array([0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1])
        choice_y = np.array([1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, -1])
        choice_z = np.array([1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0])
            
        conditions = [cond_x_axis, cond_y_axis, cond_z_axis]
        choices = [choice_x, choice_y, choice_z]
        
        new_data_block = np.select(conditions, choices, default=np.zeros(12))
        extended_array[:, 6:] = new_data_block
        return extended_array

    def perform_ray_intersection(self):
        print("Performing ray intersection analysis...")
        
        voxel_cpu = o3d.t.geometry.TriangleMesh.from_legacy(self.voxel_mesh)
        
        normals = self.triangle_normals_to_rays(self.voxel_mesh, offset_distance=self.voxel_size*0.25)
        extent_normals = self.extend_normals_cardinal_axes(normals)

        mesh_ = self.o3d_to_trimesh(voxel_cpu)
        
        # 5 direction intersection logic
        index_tri_0, index_ray_0, hit_locations_0 = mesh_.ray.intersects_id(extent_normals[:,:3], extent_normals[:,3:6], return_locations=True, multiple_hits=False)
        index_tri_1, index_ray_1, hit_locations_1 = mesh_.ray.intersects_id(extent_normals[:,:3], extent_normals[:,6:9], return_locations=True, multiple_hits=False)
        index_tri_2, index_ray_2, hit_locations_2 = mesh_.ray.intersects_id(extent_normals[:,:3], extent_normals[:,9:12], return_locations=True, multiple_hits=False)
        index_tri_3, index_ray_3, hit_locations_3 = mesh_.ray.intersects_id(extent_normals[:,:3], extent_normals[:,12:15], return_locations=True, multiple_hits=False)
        index_tri_4, index_ray_4, hit_locations_4 = mesh_.ray.intersects_id(extent_normals[:,:3], extent_normals[:,15:], return_locations=True, multiple_hits=False)

        # Put all arrays into a list
        all_arrays = [index_ray_0, index_ray_1, index_ray_2, index_ray_3, index_ray_4]

        # Iteratively apply np.intersect1d to the list
        common_elements_array = reduce(np.intersect1d, all_arrays)
        
        return common_elements_array

    def execute(self):
        """Runs the sequential flow and returns the final Trimesh object."""
        # 1. Generate & Filter
        self.generate_and_filter_voxels()

        # 2. Triangulate
        self.triangulate()

        # 3. Ray Intersection (Shell Calculation)
        self.voxel_mesh.compute_triangle_normals()
        final_indices = self.perform_ray_intersection()
        print(f"Indices determined to be internal (common elements): {len(final_indices)}")

        # 4. Remove Internal Faces
        self.voxel_mesh.remove_triangles_by_index(final_indices)
        self.voxel_mesh.remove_unreferenced_vertices()

        # 5. Final Conversion
        print("Converting to Trimesh for final output...")
        final_trimesh = self.o3d_to_trimesh(self.voxel_mesh)

        # --- REPAIR BLOCK ---
        final_trimesh.merge_vertices(merge_tex=True, merge_norm=True)
        final_trimesh.update_faces(final_trimesh.nondegenerate_faces())
        final_trimesh.update_faces(final_trimesh.unique_faces())
        trimesh.repair.fix_inversion(final_trimesh)
        trimesh.repair.fix_winding(final_trimesh)
        trimesh.repair.fill_holes(final_trimesh)


        # 6. Decimate
        final_trimesh = self.decimate(final_trimesh)

        # 7. Surface Generalization
        final_trimesh = self.generalize_surface(final_trimesh, smooth_iterations=10)
        final_trimesh = self.decimate(final_trimesh)

        # Verbose Status
        is_watertight = final_trimesh.is_watertight
        print("-" * 30)
        print(f"FINAL MESH STATUS:")
        print(f"  Vertices:   {len(final_trimesh.vertices)}")
        print(f"  Faces:      {len(final_trimesh.faces)}")
        print(f"  Watertight: {is_watertight}")
        print(f"  Volume:     {final_trimesh.volume:.4f}" if is_watertight else "  Volume:     N/A")
        print("-" * 30)
        return final_trimesh

    def generalize_surface(self, mesh, smooth_iterations=10):
        """
        Two-pass surface generalization to eliminate shark teeth on walls
        and stair stepping on slanted roofs.

        Pass 1 — Taubin smoothing:
            Alternating positive/negative Laplacian steps suppress high-frequency
            noise (shark teeth, stair-step quantization) without the volume
            shrinkage of plain Laplacian smoothing.

        Pass 2 — Interior plane snapping:
            For each connected coplanar region (mesh.facets), fits a least-squares
            plane via SVD to the smoothed geometry and projects interior vertices
            back onto it. Interior means every adjacent face of that vertex belongs
            to the same coplanar region — boundary vertices (shared with a
            neighbouring wall or roof facet) are never moved, which preserves
            sharp feature edges and keeps the mesh watertight.
        """
        # --- Pass 1: Taubin Smoothing ---
        print(f"Generalizing surface (Taubin ×{smooth_iterations} + plane snap)...")
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices  = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh = o3d_mesh.filter_smooth_taubin(
            number_of_iterations=smooth_iterations,
            lambda_filter=0.2,
            mu=-0.22          # standard Taubin mu: prevents shrinkage
        )

        verts = np.asarray(o3d_mesh.vertices).copy()
        faces = np.asarray(o3d_mesh.triangles)
        smoothed = trimesh.Trimesh(vertices=verts, faces=faces)

        # --- Pass 2: Interior Plane Snapping ---
        # Precompute vertex → set of adjacent face indices (needed for interior check)
        v_to_faces = [set() for _ in range(len(verts))]
        for fi, f in enumerate(faces):
            for v in f:
                v_to_faces[v].add(fi)

        for facet in smoothed.facets:
            if len(facet) < 2:
                continue
            facet_set     = set(facet.tolist())
            facet_vert_idx = np.unique(faces[facet])
            pts           = verts[facet_vert_idx]

            # Fit plane: eigenvector of smallest singular value = plane normal
            centroid = pts.mean(axis=0)
            _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
            plane_n  = Vt[-1]

            # Snap only interior vertices to eliminate in-plane noise.
            # Boundary vertices stay put to keep shared edges exact.
            for vi in facet_vert_idx:
                if v_to_faces[vi].issubset(facet_set):
                    p = verts[vi]
                    verts[vi] = p - np.dot(p - centroid, plane_n) * plane_n

        out = trimesh.Trimesh(vertices=verts, faces=faces)
        trimesh.repair.fix_winding(out)

        print("-" * 30)
        print(f"GENERALIZATION STATUS:")
        print(f"  Smooth iterations: {smooth_iterations}")
        print(f"  Watertight:        {out.is_watertight}")
        print("-" * 30)

        return out

    def decimate(self, mesh, percent_target=0.25):
        """
        Reduce face count using Open3D's quadric decimation.
        Repair passes run before and after to guard watertightness.
        """
        # Pre-decimation repair
        if not mesh.is_watertight:
            print("Pre-decimation repair: filling holes and fixing winding...")
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_inversion(mesh)
            trimesh.repair.fix_winding(mesh)

        faces_before = len(mesh.faces)
        target_faces = max(int(faces_before * percent_target), 4)
        print(f"Decimating: {faces_before:,} → target {target_faces:,} ({percent_target*100:.0f}%)")

        # Convert to O3D, decimate, convert back
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices  = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()
        o3d_simplified = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )
        out = trimesh.Trimesh(
            vertices=np.asarray(o3d_simplified.vertices),
            faces=np.asarray(o3d_simplified.triangles),
        )

        # Post-decimation repair
        if not out.is_watertight:
            print("Post-decimation repair: fixing artifacts...")
            trimesh.repair.fill_holes(out)
            trimesh.repair.fix_inversion(out)
            trimesh.repair.fix_winding(out)

        print("-" * 30)
        print(f"DECIMATION STATUS:")
        print(f"  Faces before: {faces_before:,}")
        print(f"  Faces after:  {len(out.faces):,}")
        print(f"  Reduction:    {100 * (1 - len(out.faces) / faces_before):.1f}%")
        print(f"  Watertight:   {out.is_watertight}")
        print("-" * 30)

        return out

# --- Usage ---
if __name__ == "__main__":
    # Ensure this matches your actual file name
    PLY_FILE = "bag_0518100001637360_points.ply"
    VOXEL_SIZE = 1
    
    if os.path.exists(PLY_FILE):
        processor = LotusMesh(PLY_FILE, voxel_size=VOXEL_SIZE)
        
        # Execute Pipeline
        final_mesh = processor.execute()
        
        # Export or Visualize
        output_filename = "bag_0518100001637360_points.obj"
        final_mesh.export(output_filename)
        print(f"Mesh saved to {output_filename}")
        
        # Optional: Visualize using Trimesh's native viewer
        print("Opening Open3D Viewer...")
        final_o3d = processor.voxel_mesh
        final_o3d.compute_vertex_normals()
        final_o3d.paint_uniform_color([0.7, 0.7, 0.7]) # nice grey
        o3d.visualization.draw_geometries([final_o3d], mesh_show_wireframe=True)
        
    else:
        print(f"Error: {PLY_FILE} not found. Please check the path.")