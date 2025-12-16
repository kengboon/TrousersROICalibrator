import numpy as np
import torch

class TrousersROICalibrator:
    def __init__(self, model, transforms=None, score_thres: float = 0.85, device="cuda"):
        self.model = model
        self.transforms = transforms
        self.score_thres = score_thres
        self.device = device
        self.learned_cfgs = {}

        # MAPPING: Image Labels (1-14) -> Code Indices (0-13)
        # 4:L_Hip, 5:L_Knee, 6:L_Hem, 7:L_InHem, 8:L_InKnee, 9:Crotch
        # 14:R_Hip, 13:R_Knee, 12:R_Hem, 11:R_InHem, 10:R_InKnee
        
        # Define the 'Mesh' of triangles covering the pants
        # Each tuple is (Point A, Point B, Point C)
        self.MESH_TOPOLOGY = {
            # --- LEFT LEG ---
            # Thigh Area (Split into 2 triangles)
            "L_Thigh_Upper": [3, 8, 4],  # L_Hip(4), Crotch(9), L_Knee(5)
            "L_Thigh_Lower": [8, 7, 4],  # Crotch(9), L_InKnee(8), L_Knee(5)
            # Calf Area
            "L_Calf_Upper":  [4, 7, 5],  # L_Knee(5), L_InKnee(8), L_Hem(6)
            "L_Calf_Lower":  [7, 6, 5],  # L_InKnee(8), L_InHem(7), L_Hem(6)

            # --- RIGHT LEG ---
            # Thigh Area
            "R_Thigh_Upper": [13, 8, 12], # R_Hip(14), Crotch(9), R_Knee(13)
            "R_Thigh_Lower": [8, 9, 12],  # Crotch(9), R_InKnee(10), R_Knee(13)
            # Calf Area
            "R_Calf_Upper":  [12, 9, 11], # R_Knee(13), R_InKnee(10), R_Hem(12)
            "R_Calf_Lower":  [9, 10, 11], # R_InKnee(10), R_InHem(11), R_Hem(12)
            
            # --- WAIST (Optional, for pockets) ---
            "Waist_Left":    [0, 1, 3],   # L_Waist(1), M_Waist(2), L_Hip(4)
            "Waist_Right":   [1, 2, 13],  # M_Waist(2), R_Waist(3), R_Hip(14)
        }

    def _get_keypoints(self, image, do_transforms: bool = True):
        # (Same model inference code as before)
        if do_transforms and self.transforms is not None:
            image = self.transforms(image)
        image = image.unsqueeze(0).to(device=self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)[0]

        filter_mask = outputs["scores"] >= self.score_thres
        if torch.sum(filter_mask) == 0:
            return None

        # Assume taking the best detection
        keypoints = outputs["keypoints"][filter_mask].cpu().numpy()[0]
        # keypoints shape: (14, 3) -> [x, y, visibility]
        return keypoints

    def _barycentric_coords(self, p, a, b, c):
        """
        Calculates weights (u, v, w) for point p inside triangle abc.
        p = u*a + v*b + w*c
        """
        v0, v1, v2 = b - a, c - a, p - a
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        
        # Avoid division by zero for degenerate triangles
        if abs(denom) < 1e-5: return -1, -1, -1 

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u, v, w

    def _cartesian_coords(self, u, v, w, a, b, c):
        """ Reconstructs p from weights and vertices """
        return u * a + v * b + w * c

    def learn_rois(self, ref_image, user_drawn_boxes, ref_keypoints=None, reset_learned=True):
        if reset_learned:
            self.learned_cfgs = {}
        if ref_keypoints is None:
            ref_keypoints = self._get_keypoints(ref_image)
            if ref_keypoints is None: raise ValueError("No keypoints detected.")

        print(f"Calibrating {len(user_drawn_boxes)} ROIs...")

        for roi_name, box in user_drawn_boxes.items():
            x1, y1, x2, y2 = box
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            box_dims = np.array([x2 - x1, y2 - y1])

            best_tri_name = None
            best_coords = None
            
            # 1. Find which triangle contains the ROI center
            # If multiple (overlapping) or none, pick the one with "most positive" weights or closest centroid
            best_score = -float('inf') # Score = sum of negative weights (closer to 0 is better if outside)

            for tri_name, indices in self.MESH_TOPOLOGY.items():
                pts = ref_keypoints[indices, :2] # Get X,Y of the 3 points
                u, v, w = self._barycentric_coords(center, pts[0], pts[1], pts[2])
                
                # Check if point is strictly inside (all > 0)
                is_inside = (u >= 0) and (v >= 0) and (w >= 0)
                
                if is_inside:
                    best_tri_name = tri_name
                    best_coords = (u, v, w)
                    break # Found a perfect match
                else:
                    # If outside, track the "least outside" triangle (closest)
                    # A heuristic: max(min(u,v,w)) tries to find the one we are "least far" from
                    score = min(u, v, w) 
                    if score > best_score:
                        best_score = score
                        best_tri_name = tri_name
                        best_coords = (u, v, w)

            # 2. Store Config
            # We also store the box size relative to the triangle's "size" (e.g. area or edge len)
            # But for simplicity, we'll store raw w/h and scale it by the triangle's scale factor during prediction
            
            # Calculate Reference Scale (Square root of Triangle Area)
            pts = ref_keypoints[self.MESH_TOPOLOGY[best_tri_name], :2]
            ref_area = 0.5 * np.abs(np.cross(pts[1]-pts[0], pts[2]-pts[0]))
            ref_scale = np.sqrt(ref_area) if ref_area > 0 else 1.0

            self.learned_cfgs[roi_name] = {
                "triangle": best_tri_name,
                "bary_weights": best_coords,
                "rel_dims": box_dims / ref_scale, # Store dimensions normalized to triangle size
            }
            print(f" - Learned '{roi_name}': attached to {best_tri_name}")

    def predict_rois(self, target_image):
        if not self.learned_cfgs: return {}
        
        tgt_keypoints = self._get_keypoints(target_image)
        if tgt_keypoints is None: return {}

        results = {}

        for roi_name, cfg in self.learned_cfgs.items():
            tri_name = cfg["triangle"]
            u, v, w = cfg["bary_weights"]
            indices = self.MESH_TOPOLOGY[tri_name]
            
            # Get the new positions of the 3 anchor points
            # Ensure they are visible! If invisible, inference might be shaky.
            pts = tgt_keypoints[indices, :2] 
            
            # 1. Reconstruct Center
            new_center = self._cartesian_coords(u, v, w, pts[0], pts[1], pts[2])
            
            # 2. Reconstruct Scale
            # Calculate new triangle area to determine scale factor
            new_area = 0.5 * np.abs(np.cross(pts[1]-pts[0], pts[2]-pts[0]))
            new_scale = np.sqrt(new_area) if new_area > 0 else 1.0
            
            new_w, new_h = cfg["rel_dims"] * new_scale
            
            # 3. Build Box
            nx1 = int(new_center[0] - new_w / 2)
            ny1 = int(new_center[1] - new_h / 2)
            nx2 = int(new_center[0] + new_w / 2)
            ny2 = int(new_center[1] + new_h / 2)
            
            results[roi_name] = [nx1, ny1, nx2, ny2]

        return results

def fix_trouser_orientation(kpts):
    """
    Robustly corrects trouser keypoints using geometric consistency.
    Ensures all horizontal vectors align with the waist direction (1->3).
    
    Args:
        kpts (np.array): Shape (14, 2) or (14, 3). 
                         Indices 0-13 correspond to Labels 1-14.
    """
    kpts = kpts.copy()
    
    # --- 1. Master Direction ---
    # We trust the Waist (Label 1 -> Label 3) to define the object's orientation.
    # If the person turns around (back view), this vector flips, 
    # and all other vectors must flip to match it.
    p1 = kpts[0, :2] # Label 1
    p3 = kpts[2, :2] # Label 3
    waist_vector = p3 - p1
    
    # --- 2. Check Inter-Leg (Left Leg vs Right Leg) ---
    # These checks ensure the Left Leg is actually to the left of the Right Leg.
    # Pairs: (Left_Point, Right_Point)
    inter_leg_pairs = [
        (3, 13),   # Hips:   4 (L) <-> 14 (R)
        (4, 12),   # Knees:  5 (L) <-> 13 (R)
        (7, 9),    # Inner:  8 (L) <-> 10 (R)
        (5, 11),   # Cuffs:  6 (L) <-> 12 (R)
        (6, 10)    # Inner:  7 (L) <-> 11 (R)
    ]
    
    for (idx_a, idx_b) in inter_leg_pairs:
        kpts = check_and_swap(kpts, idx_a, idx_b, waist_vector)

    # --- 3. Check Intra-Leg (Leg Width / Twist) ---
    # These checks ensure the legs aren't "twisted" (Inner vs Outer seam).
    # Logic: The vector must always flow "Left-to-Right" (relative to the object).
    intra_leg_pairs = [
        # Left Leg: Outer(L) -> Inner(L)
        (4, 7),    # Label 5 -> 8 (Knee Width)
        (5, 6),    # Label 6 -> 7 (Cuff Width)
        
        # Right Leg: Inner(R) -> Outer(R)
        (9, 12),   # Label 10 -> 13 (Knee Width)
        (10, 11)   # Label 11 -> 12 (Cuff Width)
    ]
    
    for (idx_a, idx_b) in intra_leg_pairs:
        kpts = check_and_swap(kpts, idx_a, idx_b, waist_vector)
            
    return kpts

def check_and_swap(kpts, idx_a, idx_b, ref_vector):
    """
    Swaps kpts[idx_a] and kpts[idx_b] if the vector A->B 
    points in the opposite direction of the reference vector.
    """
    p_a = kpts[idx_a, :2]
    p_b = kpts[idx_b, :2]
    
    pair_vector = p_b - p_a
    
    # Dot product > 0: Aligned (Correct)
    # Dot product < 0: Opposed (Swapped)
    if np.dot(ref_vector, pair_vector) < 0:
        temp = kpts[idx_a].copy()
        kpts[idx_a] = kpts[idx_b]
        kpts[idx_b] = temp

    return kpts