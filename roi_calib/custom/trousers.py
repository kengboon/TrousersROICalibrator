import numpy as np

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