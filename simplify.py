import trimesh
import numpy as np

class MeshSimplifier:
    def __init__(self, file_path):
        self.original_mesh = self._load_mesh(file_path)
        self.simplified_mesh = None

    def _load_mesh(self, path):
        loaded = trimesh.load(path, force='mesh')
        if isinstance(loaded, trimesh.Scene):
            if len(loaded.geometry) == 0:
                raise ValueError("Input OBJ appears to be empty.")
            return trimesh.util.concatenate(tuple(loaded.geometry.values()))
        return loaded

    def clean_mesh(self):
        """
        Pre-process to fix disconnected vertices and duplicate faces.
        """
        print("Cleaning mesh (merging vertices, removing duplicates)...")
        self.original_mesh.merge_vertices()
        
        # FIX: New syntax replacing deprecated remove_duplicate_faces
        self.original_mesh.update_faces(self.original_mesh.unique_faces())
        
        # FIX: New syntax replacing deprecated remove_degenerate_faces
        self.original_mesh.update_faces(self.original_mesh.nondegenerate_faces())

    def simplify_coplanar(self, angle_deg=2.0):
        print(f"Identifying coplanar facets (Tolerance: {angle_deg}°)...")
        
        # FIX: Use the graph module function, not the property
        tolerance_rad = np.radians(angle_deg)
        facets = trimesh.graph.facets(self.original_mesh, tolerance=tolerance_rad)
        
        print(f"Found {len(facets)} coplanar regions. Retriangulating...")

        new_vertices = []
        new_faces = []
        
        # Process each facet
        for i, face_indices in enumerate(facets):
            try:
                # 1. Project this cluster of faces to 2D
                # This finds the outer boundary (and holes) of the coplanar cluster
                path_2d = trimesh.path.polygons.projected(
                    self.original_mesh, 
                    faces=face_indices
                )
                
                # 2. Triangulate the resulting polygon
                # This creates a clean set of large triangles filling the shape
                triangulated_facet = path_2d.triangulate()
                
                # 3. Add to our new lists (offsetting indices)
                current_vertex_count = len(new_vertices) * len(triangulated_facet.vertices) 
                # Note: fast stacking is better, but let's do safe appending first
                
                # Careful with vertex indexing here. 
                # We need to append vertices and shift face indices.
                if len(new_vertices) == 0:
                    current_offset = 0
                else:
                    # Calculate total vertices currently in the list
                    current_offset = sum(len(v) for v in new_vertices)

                new_vertices.append(triangulated_facet.vertices)
                new_faces.append(triangulated_facet.faces + current_offset)
                
            except Exception as e:
                # Fallback: Keep original faces if complex path fails
                # (e.g. self-intersecting geometry)
                continue

        if not new_vertices:
            print("Simplification failed (no valid geometry created).")
            return

        # Stack into final arrays
        final_vertices = np.vstack(new_vertices)
        final_faces = np.vstack(new_faces)

        # Create new mesh
        self.simplified_mesh = trimesh.Trimesh(vertices=final_vertices, faces=final_faces)
        self.simplified_mesh.fix_normals()
        
        print(f"Done. Reduced from {len(self.original_mesh.faces)} to {len(self.simplified_mesh.faces)} faces.")

    def save(self, output_path):
        if self.simplified_mesh is None:
            self.original_mesh.export(output_path)
        else:
            self.simplified_mesh.export(output_path)
            print(f"Saved to {output_path}")

# --- Usage ---
if __name__ == "__main__":
    # Update these paths to your actual files
    input_file = "final_building_mesh_hipped.obj"  
    output_file = "clean_mesh.obj"
    
    processor = MeshSimplifier(input_file)
    processor.clean_mesh()
    
    # 5 degrees tolerance
    processor.simplify_coplanar(angle_deg=5.0)
    
    processor.save(output_file)