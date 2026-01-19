import json
import numpy as np
import uuid
import math

def load_obj_direct(filepath):
    """Reads OBJ v and f tags directly."""
    verts = []
    faces = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                face = [int(p.split('/')[0]) - 1 for p in parts]
                if len(face) >= 3:
                    faces.append(face)
                    
    return np.array(verts), faces

def generate_cityjson_id():
    return f"GUID_{uuid.uuid4()}"

def refine_ground_vs_ceiling(candidates, semantics_list, gap_threshold=0.5):
    """
    Clusters downward surfaces into Ground (bottom) and OuterCeiling (top)
    based on the largest vertical gap.
    
    :param candidates: List of tuples (original_index, z_centroid)
    :param semantics_list: The master list of semantics to update in-place
    :param gap_threshold: Minimum vertical distance (m) to consider a split
    """
    if not candidates:
        return

    # Extract Z values and indices
    dtype = [('index', int), ('z', float)]
    data = np.array(candidates, dtype=dtype)
    
    # Sort by Z to find gaps
    sorted_data = np.sort(data, order='z')
    z_values = sorted_data['z']
    
    # Calculate differences between consecutive sorted Zs
    z_diffs = np.diff(z_values)
    
    if len(z_diffs) == 0:
        return # Only one surface, stays GroundSurface
    
    # Find the largest gap
    max_gap_idx = np.argmax(z_diffs)
    max_gap = z_diffs[max_gap_idx]
    
    # Logic:
    # If the largest gap is significant (> 1m), we assume everything BELOW 
    # that gap is Ground, and everything ABOVE is Ceiling/Overhang.
    if max_gap > gap_threshold:
        # The split point is between index max_gap_idx and max_gap_idx+1
        split_z_value = (z_values[max_gap_idx] + z_values[max_gap_idx+1]) / 2.0
        
        # Indices in the sorted array that are ABOVE the split
        # We look at the original indices stored in the structured array
        ceiling_indices = sorted_data[sorted_data['z'] > split_z_value]['index']
        
        # Update the semantics list
        for idx in ceiling_indices:
            semantics_list[idx] = "OuterCeiling"
            
        print(f"   [Clustering] Detected gap of {max_gap:.2f}m at Z={split_z_value:.2f}.")
        print(f"   - Reclassified {len(ceiling_indices)} surfaces as OuterCeiling.")
    else:
        print(f"   [Clustering] No significant vertical gap detected (Max gap: {max_gap:.2f}m). Keeping all as Ground.")

def obj_to_cityjson(obj_path, output_path):
    print("Loading OBJ (Direct Mode)...")
    verts, faces = load_obj_direct(obj_path)
    
    if len(verts) == 0:
        print("Error: No vertices found.")
        return

    print(f"Processing {len(faces)} faces...")

    min_v = np.min(verts, axis=0)
    max_v = np.max(verts, axis=0)
    
    # Metadata extent
    extent = [min_v[0], min_v[1], min_v[2], max_v[0], max_v[1], max_v[2]]

    surfaces_geometry = [] 
    surfaces_semantics = []
    
    # Store indices of downward faces for post-processing: (list_index, z_centroid)
    downward_candidates = []

    # Threshold for "Vertical" walls (approx 85 degrees)
    threshold = np.sin(np.radians(5))

    for face in faces:
        p0 = verts[face[0]]
        p1 = verts[face[1]]
        p2 = verts[face[2]]
        
        # Normal Calculation
        n = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(n)
        
        if ln < 1e-6:
            continue 
            
        n_unit = n / ln
        
        # Calculate Face Centroid Z for clustering
        face_centroid_z = (p0[2] + p1[2] + p2[2]) / 3.0
        
        # Classification
        s_type = "WallSurface" # Default fallback
        
        if abs(n_unit[2]) <= threshold: 
            # It's a Wall (Normal is roughly horizontal)
            s_type = "WallSurface"
        elif n_unit[2] > threshold: 
            # Upward facing
            s_type = "RoofSurface"
        elif n_unit[2] < 0: # Strongly downward (negative Z)
            # Temporarily assign GroundSurface; will refine later
            s_type = "GroundSurface"
            
            # Store index and Z for clustering
            current_index = len(surfaces_semantics)
            downward_candidates.append((current_index, face_centroid_z))
            
        # 4. Add to lists
        surfaces_geometry.append([face]) 
        surfaces_semantics.append(s_type)

    # --- POST PROCESSING CLUSTERING ---
    print(f" Analyzing {len(downward_candidates)} downward surfaces for Ground/Ceiling split...")
    refine_ground_vs_ceiling(downward_candidates, surfaces_semantics)

    # Prepare Semantic Object
    unique_types = sorted(list(set(surfaces_semantics)))
    type_map = {t: i for i, t in enumerate(unique_types)}
    semantic_values = [type_map[t] for t in surfaces_semantics]

    solid_boundaries = [surfaces_geometry]
    solid_semantics_values = [semantic_values]

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

# Usage
obj_to_cityjson("decimated.obj", "output_oriented.city.json")