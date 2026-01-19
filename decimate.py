import trimesh
import open3d as o3d
import numpy as np

def trimesh_to_open3d(t_mesh):
    """Helper to convert Trimesh object to Open3D TriangleMesh"""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(t_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(t_mesh.faces)
    # Compute normals for better decimation heuristic
    o3d_mesh.compute_vertex_normals() 
    return o3d_mesh

def open3d_to_trimesh(o3d_mesh):
    """Helper to convert Open3D TriangleMesh back to Trimesh"""
    # Open3D stores data in vector objects, we need to cast to numpy arrays
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def decimate(input_path, output_path, percent_target):
    print(f"--- Processing: {input_path} ---")
    
    # 1. Load with Trimesh (Auto-cleans basic duplication)
    mesh = trimesh.load(input_path)
    
    # Validation Check 1
    if not mesh.is_watertight:
        print(">> Input Repair: Filling holes and fixing winding...")
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_winding(mesh)

    original_faces = len(mesh.faces)
    target_faces = int(original_faces * percent_target)
    target_faces = max(target_faces, 4) # Safety floor

    print(f"Original Faces: {original_faces}")
    print(f"Target Faces:   {target_faces} ({percent_target*100}%)")

    # 2. Convert to Open3D for Decimation
    # Open3D's implementation is native and fast
    o3d_mesh = trimesh_to_open3d(mesh)
    
    # Simplify (Quadric Decimation)
    # preserve_border=True prevents the mesh from shrinking edges and creating holes
    o3d_simplified = o3d_mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_faces
    )

    # 3. Convert back to Trimesh for Final Repairs
    final_mesh = open3d_to_trimesh(o3d_simplified)
    
    # Validation Check 2 (Post-Decimation)
    # Decimation can sometimes flip a tiny face or create a micro-hole
    if not final_mesh.is_watertight:
        print(">> Post-Decimation Repair: Fixing artifacts...")
        trimesh.repair.fill_holes(final_mesh)
        trimesh.repair.fix_inversion(final_mesh)
        trimesh.repair.fix_winding(final_mesh)

    # Final Stats
    print(f"Final Faces:    {len(final_mesh.faces)}")
    print(f"Is Watertight?  {final_mesh.is_watertight}")
    
    # 4. Save
    final_mesh.export(output_path)
    print(f"--- Saved to {output_path} ---")

# Usage
decimate("final_building_mesh_hipped.obj", "decimated.obj", 0.10)