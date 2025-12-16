import numpy as np
import torch

class TrousersROICalibrator:
    def __init__(self, model, transforms=None, score_thres: float=0.85, device="cuda"):
        self.model = model
        self.transforms = transforms
        self.score_thres = score_thres
        self.device = device
        self.learned_cfgs = {}

    def _get_keypoints(self, image, do_transforms: bool=True):
        if do_transforms and self.transforms is not None:
            image = self.transforms(image)
        image = image.unsqueeze(0).to(device=self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)[0]

        # Filter low confidence instances
        filter = outputs["scores"] >= self.score_thres
        boxes = outputs["boxes"][filter].cpu().numpy()
        keypoints = outputs["keypoints"][filter].cpu().numpy()
        keypoints = np.array([fix_trouser_orientation(kypts) for kypts in keypoints])

        return {
            "boxes": boxes,
            "keypoints": keypoints
        }

    def _get_limb_anchors(self, keypoints, box_center_x):
        """
        Dynamically finds the 'Top' and 'Bottom' anchors relevant to a specific ROI.
        It splits the jeans into Left/Right legs based on X-coordinates.
        """
        # Filter only visible points
        visible_kps = keypoints[keypoints[:, :, 2] > 0.5]
        if len(visible_kps) < 5:
            raise ValueError("Not enough visible keypoints to determine shape.")

        # 1. Determine Vertical span (Fabric Top/Bottom)
        fabric_top = np.min(visible_kps[:, 1])
        fabric_bottom = np.max(visible_kps[:, 1])
        
        # 2. Determine Horizontal span (Left Leg vs Right Leg)
        # We use the median X to split the pants into Left and Right zones
        fabric_center_x = np.median(visible_kps[:, 0])
        
        # Check which 'side' the ROI box belongs to
        if box_center_x < fabric_center_x:
            # ROI is on the Left side of the image
            relevant_kps = visible_kps[visible_kps[:, 0] < fabric_center_x]
            side = "left"
        else:
            # ROI is on the Right side
            relevant_kps = visible_kps[visible_kps[:, 0] >= fabric_center_x]
            side = "right"

        # Refine Top/Bottom for this specific leg/side
        if len(relevant_kps) > 2:
            limb_top = np.min(relevant_kps[:, 1])
            limb_bottom = np.max(relevant_kps[:, 1])
            limb_left = np.min(relevant_kps[:, 0])
            limb_right = np.max(relevant_kps[:, 0])
        else:
            # Fallback to global if leg detection is sparse
            limb_top, limb_bottom = fabric_top, fabric_bottom
            limb_left = np.min(visible_kps[:, 0])
            limb_right = np.max(visible_kps[:, 0])

        return {
            "top": limb_top,
            "bottom": limb_bottom,
            "left": limb_left,
            "right": limb_right,
            "height": limb_bottom - limb_top,
            "width": limb_right - limb_left,
            "side": side
        }

    def learn_rois(self, ref_image, user_drawn_boxes, ref_keypoints=None, reset_learned=True):
        """
        Step 1: Calibration
        ref_image       : The 'Golden Sample' image
        user_drawn_boxes: Dict {'knee': [x1, y1, x2, y2], 'pocket': ...}
        """
        if reset_learned:
            self.learned_cfgs = {}

        if ref_keypoints is None:
            ref_keypoints = self._get_keypoints(ref_image)["keypoints"]
        
        print(f"Calibrating {len(user_drawn_boxes)} ROIs...")
        
        for roi_name, box in user_drawn_boxes.items():
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            
            # 1. Find the relevant anchors (Left Leg? Right Leg?)
            anchors = self._get_limb_anchors(ref_keypoints, cx)
            
            # 2. Calculate Ratios (Normalize box relative to the limb)
            # Example: A knee might start at 50% of leg height and end at 70%
            relative_y1 = (y1 - anchors['top']) / anchors['height']
            relative_y2 = (y2 - anchors['top']) / anchors['height']
            
            # For Width, we usually want it centered on the leg, but let's learn the offset too
            relative_x1 = (x1 - anchors['left']) / anchors['width']
            relative_x2 = (x2 - anchors['left']) / anchors['width']
            
            self.learned_cfgs[roi_name] = {
                "rel_y1": relative_y1,
                "rel_y2": relative_y2,
                "rel_x1": relative_x1,
                "rel_x2": relative_x2,
                "side": anchors['side'] # Remembers if this was a "left leg" ROI
            }
            print(f" - Learned '{roi_name}' on {anchors['side']} side: Y-range [{relative_y1:.2f}-{relative_y2:.2f}]")

    def predict_rois(self, target_image):
        """
        Step 2: Inference on new images
        Returns: Dict of projected boxes {'knee': [x1, y1, x2, y2]}
        """
        if not self.learned_cfgs:
            print("Warning: No ROIs learned yet. Call learn_rois() first.")
            return {}

        keypoints = self._get_keypoints(target_image)["keypoints"]
        results = {}

        # Pre-calculate anchors for both sides to save time
        # We assume the new image has a similar center split
        visible_kps = keypoints[keypoints[:, :, 2] > 0.5]
        fabric_center_x = np.median(visible_kps[:, 0])
        
        # Get anchors for virtual "Left" and "Right" zones on the new image
        # Note: We pass a dummy X to force the function to give us left/right params
        left_anchors = self._get_limb_anchors(keypoints, fabric_center_x - 10)
        right_anchors = self._get_limb_anchors(keypoints, fabric_center_x + 10)

        for roi_name, config in self.learned_cfgs.items():
            # 1. Select the correct anchors based on what we learned
            anchors = left_anchors if config['side'] == 'left' else right_anchors
            
            # 2. Re-project (Denormalize)
            new_y1 = anchors['top'] + (config['rel_y1'] * anchors['height'])
            new_y2 = anchors['top'] + (config['rel_y2'] * anchors['height'])
            new_x1 = anchors['left'] + (config['rel_x1'] * anchors['width'])
            new_x2 = anchors['left'] + (config['rel_x2'] * anchors['width'])
            
            # Clamp to image bounds (optional)
            results[roi_name] = [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]
            
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