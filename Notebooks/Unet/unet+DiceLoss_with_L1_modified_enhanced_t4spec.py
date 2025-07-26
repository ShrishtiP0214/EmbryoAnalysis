import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageFile
import numpy as np
import pandas as pd
import random
import re
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set a fixed seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Utility function to find common embryo IDs
def get_common_embryo_ids(base_paths):
    sets_of_ids = []
    for path in base_paths:
        subfolders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        sets_of_ids.append(set(subfolders))
    common_ids = set.intersection(*sets_of_ids)
    return sorted(list(common_ids))

# U-Net building blocks
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(UNet, self).__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        output = self.conv10(c9)
        return torch.sigmoid(output)

# Custom Dataset with Augmentation for t4 phase
class EmbryoT4Dataset(Dataset):
    def __init__(self, base_paths, phase_csv_dir, embryo_ids, transform=None, num_t4_embryos=150, num_other_embryos=50):
        if len(base_paths) != 6:
            raise ValueError("Exactly 6 focal-plane directories are required.")
        
        self.base_paths = base_paths
        self.phase_csv_dir = phase_csv_dir
        self.transform = transform
        
        t4_embryos = []
        for eid in embryo_ids:
            csv_path = os.path.join(phase_csv_dir, f"{eid}_phases.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, header=None, names=['phase', 'start_frame', 'end_frame'])
                t4_row = df[df['phase'] == 't4']
                if not t4_row.empty and t4_row['start_frame'].iloc[0] <= t4_row['end_frame'].iloc[0]:
                    t4_embryos.append((eid, t4_row['start_frame'].iloc[0], t4_row['end_frame'].iloc[0]))
        
        self.t4_embryos = random.sample(t4_embryos, min(num_t4_embryos, len(t4_embryos)))
        t4_ids = set(eid for eid, _, _ in self.t4_embryos)
        other_ids = [eid for eid in embryo_ids if eid not in t4_ids]
        self.other_embryos = random.sample(other_ids, min(num_other_embryos, len(other_ids)))
        
        self.embryo_to_frames = {}
        self.embryo_to_frame_files = {}
        for eid in [eid for eid, _, _ in self.t4_embryos] + self.other_embryos:
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
        
        self.samples = []
        for eid, start, end in self.t4_embryos:
            available_t4_frames = [f for f in self.embryo_to_frames[eid] if start <= f <= end]
            selected_frames = random.sample(available_t4_frames, min(2, len(available_t4_frames)))
            for frame in selected_frames:
                self.samples.append((eid, frame))
        for eid in self.other_embryos:
            available_frames = self.embryo_to_frames[eid]
            selected_frames = random.sample(available_frames, min(2, len(available_frames)))
            for frame in selected_frames:
                self.samples.append((eid, frame))
        
        if len(self.samples) == 0:
            raise ValueError("No samples found in the dataset. Check if t4 phase data exists and matches image frames.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        embryo_id, frame = self.samples[idx]
        filename = self.embryo_to_frame_files[embryo_id][frame]
        
        focal_images = []
        for path in self.base_paths:
            img_path = os.path.join(path, embryo_id, filename)
            image = Image.open(img_path).convert('L')
            image = np.array(image)
            focal_images.append(image)
        
        augmented = [self.transform(image=img)['image'] for img in focal_images]
        
        input_tensor = torch.cat(augmented, dim=0)
        target = augmented[2]  # Using third focal plane as dummy target
        return input_tensor, target

def extract_frame_number(filename):
    match = re.search(r'RUN(\\d+)', filename)
    return int(match.group(1)) if match else None

# Loss Functions
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice_score

def gradient_loss(output, inputs):
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32).to(output.device)
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32).to(output.device)
    
    grad_output_x = torch.abs(F.conv2d(output, sobel_x, padding=1))
    grad_output_y = torch.abs(F.conv2d(output, sobel_y, padding=1))
    grad_output = torch.sqrt(grad_output_x**2 + grad_output_y**2)

    max_grad = None
    for i in range(inputs.size(1)):
        input_channel = inputs[:, i:i+1, :, :]
        grad_x = torch.abs(F.conv2d(input_channel, sobel_x, padding=1))
        grad_y = torch.abs(F.conv2d(input_channel, sobel_y, padding=1))
        grad = torch.sqrt(grad_x**2 + grad_y**2)
        if max_grad is None:
            max_grad = grad
        else:
            max_grad = torch.max(max_grad, grad)

    loss = torch.mean(torch.clamp(max_grad - grad_output, min=0))
    return loss

def combined_loss(output, target, inputs, lambda_grad=0.1):
    dice_loss_fn = DiceLoss()
    l1_loss_fn = nn.L1Loss()
    dice = dice_loss_fn(output, target)
    l1 = l1_loss_fn(output, target)
    grad_loss = gradient_loss(output, inputs)
    return dice + l1 + lambda_grad * grad_loss

# Training Function with Tuned Hyperparameters
def train_model(model, train_loader, val_loader, num_epochs=50, device='cpu', lr=0.0001, batch_size=4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = combined_loss(outputs, targets, inputs, lambda_grad=0.1)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        train_loss = running_train_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = combined_loss(outputs, targets, inputs, lambda_grad=0.1)
                running_val_loss += loss.item() * inputs.size(0)
        val_loss = running_val_loss / len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'embryo_unet_t4.pth')
            print(f"  [*] Model saved at epoch {epoch+1}")

# Testing Function (Unchanged)
def test_single_embryo(model, image_paths, transform, device='cpu'):
    model.eval()
    focal_tensors = []
    for path in image_paths:
        img = Image.open(path).convert('L')
        img_tensor = transform(image=np.array(img))['image']
        focal_tensors.append(img_tensor)
    
    input_tensor = torch.cat(focal_tensors, dim=0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    output_image = output.squeeze(0).cpu()
    fused_pil = transforms.ToPILImage()(output_image)
    return fused_pil

# Main Training Function
def main_train(seed=42):
    set_seed(seed)

    base_paths = [
        r"C:\Projects\Embryo\Dataset\embryo_dataset_F15",
        r"C:\Projects\Embryo\Dataset\embryo_dataset_F-15",
        r"C:\Projects\Embryo\Dataset\embryo_dataset_F30",
        r"C:\Projects\Embryo\Dataset\embryo_dataset_F-30",
        r"C:\Projects\Embryo\Dataset\embryo_dataset_F45",
        r"C:\Projects\Embryo\Dataset\embryo_dataset_F-45"
    ]
    phase_csv_dir = r"C:\Projects\Embryo\Dataset\embryo_dataset_annotations"
    
    embryo_ids = get_common_embryo_ids(base_paths)
    print(f"Found {len(embryo_ids)} embryo IDs: {embryo_ids[:5]} ...")
    
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
        embryo_ids=embryo_ids,
        transform=transform,
        num_t4_embryos=50,  # Modified to select 50 t4 embryos
        num_other_embryos=0  # Modified to exclude non-t4 embryos
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = UNet(in_channels=6, out_channels=1).to(device)
    
    print("Starting training...")
    train_model(model, train_loader, val_loader, num_epochs=50, device=device, lr=0.0001, batch_size=4)
    print("Training complete. Best model saved as 'embryo_unet_t4.pth'.")

# Main Testing Function (Updated Transform)
def main_test():
    transform = A.Compose([
        A.Resize(256, 256),
        ToTensorV2()
    ])
    
    test_image_paths = [
        r"C:\Projects\Embryo\Dataset\embryo_dataset_F15\AB91-1\D2013.01.29_S0719_I132_WELL1_RUN150.jpeg",
        r"C:\Projects\Embryo\Dataset\embryo_dataset_F30\AB91-1\D2013.01.29_S0719_I132_WELL1_RUN150.jpeg",
        r"C:\Projects\Embryo\Dataset\embryo_dataset_F45\AB91-1\D2013.01.29_S0719_I132_WELL1_RUN150.jpeg",
        r"C:\Projects\Embryo\Dataset\embryo_dataset_F-30\AB91-1\D2013.01.29_S0719_I132_WELL1_RUN150.jpeg",
        r"C:\Projects\Embryo\Dataset\embryo_dataset_F-15\AB91-1\D2013.01.29_S0719_I132_WELL1_RUN150.jpeg",
        r"C:\Projects\Embryo\Dataset\embryo_dataset_F-45\AB91-1\D2013.01.29_S0719_I132_WELL1_RUN150.jpeg"
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = UNet(in_channels=6, out_channels=1).to(device)
    model_path = 'embryo_unet_t4.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained model weights.")
    else:
        raise FileNotFoundError("Trained model 'embryo_unet_t4.pth' not found. Please train the model first.")
    
    fused_image = test_single_embryo(model, test_image_paths, transform, device)
    
    fused_image.save("fused_output.jpg")
    print("Fused image saved as 'fused_output.jpg'.")

if __name__ == "__main__":
    main_train(seed=42)
    # main_test()