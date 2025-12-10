from typing import List
import json
import os

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm # Optional: for progress bar

class DeepFashion2Dataset(Dataset):
    def __init__(
            self,
            root_dir,
            category_ids: None | List[int]=None,
            exclude_occulded: bool=False
        ):
        """
        root_dir: Path to the 'train' or 'validation' folder
        transforms: albumentations or torchvision transforms
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "image")
        self.anno_dir = os.path.join(root_dir, "annos")
        self.category_ids = category_ids
        self.exclude_occulded = exclude_occulded
        
        # Pre-scan for valid jeans images to avoid checking during training
        self.valid_files = []
        # --- OPTIMIZATION START ---
        # We pre-scan files to ensure we ONLY train on images with Jeans.
        print("Scanning dataset for valid categories...")
        
        all_jsons = [f for f in os.listdir(self.anno_dir) if f.endswith(".json")]
        # Take all categories
        if category_ids is None:
            self.valid_files = all_jsons
            return
        
        # Use tqdm to show progress because opening thousands of JSONs takes time
        for json_file in tqdm(all_jsons):
            json_path = os.path.join(self.anno_dir, json_file)
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Check if ANY item in this image is in target categories
                has_cat = False
                for key in data:
                    if key.startswith('item'):
                        if data[key]['category_id'] in category_ids:
                            has_cat = True
                            break # Found one, this file is valid
                
                if has_cat:
                    self.valid_files.append(json_file)
                    
            except Exception as e:
                print(f"Error reading {json_file}: {e}")

        print(f"Found {len(self.valid_files)} valid images out of {len(all_jsons)} total files.")
        # --- OPTIMIZATION END ---

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        # 1. Load Annotation
        json_file = self.valid_files[idx]
        json_path = os.path.join(self.anno_dir, json_file)
        
        with open(json_path, 'r') as f:
            anno_data = json.load(f)
            
        # 2. Load Image
        img_name = json_file.replace(".json", ".jpg")
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV is BGR

        # 3. Parse Objects
        boxes = []
        keypoints = []
        labels = []
        
        # DeepFashion2 JSONs have keys like 'item1', 'item2'...
        for key in anno_data:
            if not key.startswith('item'):
                continue

            item = anno_data[key]

            if (self.category_ids is not None) and (item['category_id'] not in self.category_ids):
                continue

            # Bounding Box [x1, y1, x2, y2]
            bbox = item['bounding_box']
            boxes.append(bbox)
            
            # Landmarks: specific format [x, y, v, x, y, v, ...]
            # DeepFashion2 has 294 total landmarks.
            # Trousers utilize a subset, but we keep the 294 shape for model compatibility.
            raw_lm = item['landmarks']
            kps = []
            for i in range(0, len(raw_lm), 3):
                x = raw_lm[i]
                y = raw_lm[i+1]
                v = raw_lm[i+2]

                if self.exclude_occulded:
                    # v=1 (occluded), v=2 (visible). Keypoint RCNN expects 1 for visible.
                    # We map: 0->0, 1->0 (treat occluded as invisible for simpler training), 2->1
                    vis = 1 if v == 2 else 0
                else:
                    vis = 1 if v > 0 else 0
                kps.append([x, y, vis])
            
            keypoints.append(kps)

            if self.category_ids is not None:
                # Class starts from 1 (0 is always background)
                labels.append(self.category_ids.index(item['category_id']) + 1)
            else:
                labels.append(item["category_id"])

        # 4. Convert to Tensors
        target = {}
        
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
            target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # Handle images with no target (shouldn't happen due to filter, but safe to have)
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["keypoints"] = torch.zeros((0, 294, 3), dtype=torch.float32)
            target["area"] = torch.as_tensor([], dtype=torch.float32)
            target["iscrowd"] = torch.as_tensor([], dtype=torch.int64)

        # Apply transforms (Basic ToTensor)
        image = image.transpose((2, 0, 1)) # HWC -> CHW
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float() / 255.0

        return image, target