import json
import numpy as np
import uuid

def load_obj_direct(filepath):
    """
    Reads OBJ v and f tags directly. 
    Does NOT merge vertices or faces.
    """
    verts = []
    faces = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                # Store (x, y, z)
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                # OBJ is 1-based; convert to 0-based index.
                # We take the first part of 'v/vt/vn' (vertex index).
                face = [int(p.split('/')[0]) - 1 for p in parts]
                
                # Only keep valid faces (triangles or polys)
                if len(face) >= 3:
                    faces.append(face)
                    
    return np.array(verts), faces

def generate_cityjson_id():
    """Generates a UUID based ID."""
    return f"GUID_{uuid.uuid4()}"

def obj_to_cityjson(obj_path, output_path):
    print("Loading OBJ (Direct Mode)...")
    verts, faces = load_obj_direct(obj_path)
    
    if len(verts) == 0:
        print("Error: No vertices found.")
        return

    print(f"Processing {len(faces)} faces...")

    # 1. Calculate Bounding Box Center
    # This is more robust than the mean of vertices for "Inside" checks
    min_v = np.min(verts, axis=0)
    max_v = np.max(verts, axis=0)
    center = (min_v + max_v) / 2.0
    
    extent = [min_v[0], min_v[1], min_v[2], max_v[0], max_v[1], max_v[2]]

    surfaces_geometry = [] 
    surfaces_semantics = []
    
    for face in faces:
        p0 = verts[face[0]]
        p1 = verts[face[1]]
        p2 = verts[face[2]]
        
        # --- ORIENTATION ASSURANCE ---
        
        # 1. Calculate tentative normal from current winding
        n = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(n)
        
        # Skip degenerate triangles (zero area)
        if ln < 1e-6:
            continue 
            
        n_unit = n / ln
        
        # 2. Check orientation relative to Bounding Box Center
        # Vector from center to face centroid
        face_centroid = (p0 + p1 + p2) / 3.0
        view_vec = face_centroid - center
        
        # If dot product is negative, normal points INWARD. 
        # We must flip the face to make it CCW (Outward).
        if np.dot(n_unit, view_vec) < 0:
            face = face[::-1] # Flip the list of indices
            n_unit = -n_unit  # Flip normal for semantic check below
        
        # --- END ORIENTATION ASSURANCE ---

        # 3. Semantic Classification (using the corrected normal)
        # Z component > 0.5 is Roof, < -0.5 is Ground, else Wall
        if n_unit[2] > 0.5: 
            s_type = "RoofSurface"
        elif n_unit[2] < -0.5: 
            s_type = "GroundSurface"
        else: 
            s_type = "WallSurface"
            
        # 4. Add to lists
        surfaces_geometry.append([face]) 
        surfaces_semantics.append(s_type)

    # Prepare Semantic Object
    unique_types = sorted(list(set(surfaces_semantics)))
    type_map = {t: i for i, t in enumerate(unique_types)}
    semantic_values = [type_map[t] for t in surfaces_semantics]

    # Structure for "Solid" geometry
    solid_boundaries = [surfaces_geometry]
    solid_semantics_values = [semantic_values]

    # Building Metadata
    building_id = generate_cityjson_id()
    height = float(max_v[2] - min_v[2])

    cityjson = {
        "type": "CityJSON",
        "version": "2.0",
        "metadata": {
            "geographicalExtent": extent,
            "referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/7415"
        },
        "vertices": verts.tolist(),
        "CityObjects": {
            building_id: {
                "type": "Building",
                "attributes": {
                    "measuredHeight": round(height, 3),
                    "roofType": "1000",
                    "storeysAboveGround": 1
                },
                "geometry": [{
                    "type": "Solid",
                    "lod": "2", 
                    "boundaries": solid_boundaries,
                    "semantics": {
                        "surfaces": [{"type": t} for t in unique_types],
                        "values": solid_semantics_values
                    }
                }]
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(cityjson, f, separators=(',', ':'))
    
    print(f"Generated {output_path}")
    print(f" - Valid Surfaces: {len(surfaces_geometry)}")

# Usage
obj_to_cityjson("final_building_mesh_hipped.obj", "output_oriented.city.json")