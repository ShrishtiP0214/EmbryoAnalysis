import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
import pandas as pd
import random
import re
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Utility function to find common embryo IDs
def get_common_embryo_ids(base_paths):
    """
    Returns a sorted list of folder names (embryo IDs)
    that appear in *all* the given directories.
    """
    sets_of_ids = []
    for path in base_paths:
        subfolders = [
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]
        sets_of_ids.append(set(subfolders))

    common_ids = set.intersection(*sets_of_ids)
    return sorted(list(common_ids))

# Utility function to extract frame number from filename
def extract_frame_number(filename):
    match = re.search(r'RUN(\d+)', filename)
    return int(match.group(1)) if match else None

# Custom Dataset for t4 phase
class EmbryoT4Dataset(Dataset):
    def __init__(self, base_paths, phase_csv_dir, embryo_ids=None, transform=None):
        if len(base_paths) != 6:
            raise ValueError("Exactly 6 focal-plane directories are required.")
        
        self.base_paths = base_paths
        self.phase_csv_dir = phase_csv_dir
        self.transform = transform
        
        # If embryo_ids not provided, compute common IDs across all base paths
        if embryo_ids is None:
            embryo_ids = get_common_embryo_ids(base_paths)
        
        # Identify embryos with t4 phase
        t4_embryos = []
        for eid in embryo_ids:
            csv_path = os.path.join(phase_csv_dir, f"{eid}_phases.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, header=None, names=['phase', 'start_frame', 'end_frame'])
                t4_row = df[df['phase'] == 't4']
                if not t4_row.empty and t4_row['start_frame'].iloc[0] <= t4_row['end_frame'].iloc[0]:
                    t4_embryos.append((eid, t4_row['start_frame'].iloc[0], t4_row['end_frame'].iloc[0]))
        
        self.t4_embryos = t4_embryos  # Use all embryos with t4 phase
        
        # Map embryos to their frames and filenames
        self.embryo_to_frames = {}
        self.embryo_to_frame_files = {}
        for eid in [eid for eid, _, _ in self.t4_embryos]:
            subfolder = os.path.join(base_paths[0], eid)
            image_files = sorted([f for f in os.listdir(subfolder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            frames = []
            frame_files = {}
            for f in image_files:
                frame = extract_frame_number(f)
                if frame is not None:
                    frames.append(frame)
                    frame_files[frame] = f
            self.embryo_to_frames[eid] = sorted(frames)
            self.embryo_to_frame_files[eid] = frame_files
        
        # Create samples: 2 frames per embryo from t4 phase
        self.samples = []
        for eid, start, end in self.t4_embryos:
            available_t4_frames = [f for f in self.embryo_to_frames[eid] if start <= f <= end]
            selected_frames = random.sample(available_t4_frames, min(2, len(available_t4_frames)))
            for frame in selected_frames:
                self.samples.append((eid, frame))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        embryo_id, frame = self.samples[idx]
        filename = self.embryo_to_frame_files[embryo_id][frame]
        
        # Load images from all focal planes
        focal_images = []
        for path in self.base_paths:
            img_path = os.path.join(path, embryo_id, filename)
            image = Image.open(img_path).convert('L')
            image = np.array(image)  # Convert to numpy for albumentations
            focal_images.append(image)
        
        # Apply augmentations to all images simultaneously
        augmented = [self.transform(image=img)['image'] for img in focal_images]
        
        # Input tensor: stack all focal images
        input_tensor = torch.cat(augmented, dim=0)
        # Mask tensor: use the third focal plane as the mask (following original logic)
        mask_tensor = augmented[2]
        
        return input_tensor, 
    
from embryo_t4_dataset import EmbryoT4Dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

base_paths = [
    r"C:\Projects\Embryo\Dataset\embryo_dataset_F15",
    r"C:\Projects\Embryo\Dataset\embryo_dataset_F-15",
    r"C:\Projects\Embryo\Dataset\embryo_dataset_F30",
    r"C:\Projects\Embryo\Dataset\embryo_dataset_F-30",
    r"C:\Projects\Embryo\Dataset\embryo_dataset_F45",
    r"C:\Projects\Embryo\Dataset\embryo_dataset_F-45"
]
phase_csv_dir = r"C:\Projects\Embryo\Dataset\embryo_dataset_annotations"

transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.RandomBrightnessContrast(),
    A.Resize(256, 256),
    ToTensorV2()
])

dataset = EmbryoT4Dataset(
    base_paths=base_paths,
    phase_csv_dir=phase_csv_dir,
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for input_tensor, mask_tensor in dataloader:
    print(f"Input tensor shape: {input_tensor.shape}, Mask tensor shape: {mask_tensor.shape}")