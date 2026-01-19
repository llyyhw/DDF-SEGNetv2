import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class SegDataset(Dataset):
    def __init__(self, root, split="train"):
        self.split = split
        self.img_dir = os.path.join(root, split, "img")
        self.label_dir = os.path.join(root, split, "label")

        self.img_files = sorted([f for f in os.listdir(self.img_dir) 
                               if f.endswith(('.jpg', '.png', '.JPG', '.jpeg'))])

        self.label_files = []
        for img_file in self.img_files:
            label_file = os.path.splitext(img_file)[0] + '.png'
            label_path = os.path.join(self.label_dir, label_file)
            
            if os.path.exists(label_path):
                self.label_files.append(label_file)
            else:
                print(f"Warning: Label file {label_file} not found, skipping image {img_file}")
        
        assert len(self.img_files) == len(self.label_files), \
            f"Image and label count mismatch: {len(self.img_files)} vs {len(self.label_files)}"
        
        print(f"✅ {split} set: {len(self.img_files)} samples")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        label_file = self.label_files[idx]
        
        img_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.join(self.label_dir, label_file)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = label.astype(np.uint8)
        
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        label_tensor = torch.from_numpy(label).long()
        
        return img_tensor, label_tensor