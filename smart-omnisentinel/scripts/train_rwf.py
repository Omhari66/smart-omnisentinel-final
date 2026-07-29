import os
import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
import numpy as np
from pathlib import Path
from tqdm import tqdm

class RWFDataset(Dataset):
    def __init__(self, root_dir, split="train", num_frames=16, resize=(112, 112)):
        """
        Args:
            root_dir: Path to RWF-2000 (e.g. data/Violence Fight Detection dataset/RWF-2000)
            split: "train" or "val"
        """
        self.root_dir = Path(root_dir) / split
        self.num_frames = num_frames
        self.resize = resize
        
        self.video_paths = []
        self.labels = []
        
        # Fight = 1, NonFight = 0
        fight_dir = self.root_dir / "Fight"
        nonfight_dir = self.root_dir / "NonFight"
        
        if fight_dir.exists():
            for f in glob.glob(str(fight_dir / "*.avi")) + glob.glob(str(fight_dir / "*.mp4")):
                self.video_paths.append(f)
                self.labels.append(1)
                
        if nonfight_dir.exists():
            for f in glob.glob(str(nonfight_dir / "*.avi")) + glob.glob(str(nonfight_dir / "*.mp4")):
                self.video_paths.append(f)
                self.labels.append(0)

        # ── Production Feedback Loop ──────────────────────────────────────────
        # These folders are auto-populated by the SmartOmniSentinel review pipeline.
        # Every time a guard marks an incident as "False Positive" or "Confirmed Fight",
        # the clip is automatically copied here. The model learns from real deployment data.
        production_root = Path(__file__).parent.parent / "ml_training"

        false_positive_dir = production_root / "false_positives"
        confirmed_fights_dir = production_root / "confirmed_fights"

        fp_count = 0
        if false_positive_dir.exists():
            for f in glob.glob(str(false_positive_dir / "*.mp4")):
                self.video_paths.append(f)
                self.labels.append(0)   # NonFight
                fp_count += 1

        cf_count = 0
        if confirmed_fights_dir.exists():
            for f in glob.glob(str(confirmed_fights_dir / "*.mp4")):
                self.video_paths.append(f)
                self.labels.append(1)   # Fight
                cf_count += 1

        if fp_count or cf_count:
            print(f"  [Production Feedback] +{fp_count} false positives (NonFight), +{cf_count} confirmed fights (Fight)")
        else:
            print(f"  [Production Feedback] No production clips yet in ml_training/ — using base dataset only.")

        print(f"Loaded {len(self.video_paths)} videos total for {split} split.")

    def __len__(self):
        return len(self.video_paths)

    def _extract_frames(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        if total_frames < self.num_frames:
            # Not enough frames, just read what we can and pad
            indices = list(range(total_frames))
        else:
            # Uniformly sample frames
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
        current_frame = 0
        idx_pos = 0
        
        while cap.isOpened() and idx_pos < len(indices):
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame == indices[idx_pos]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.resize)
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                # ImageNet normalization
                mean = np.array([0.43216, 0.394666, 0.37645])
                std = np.array([0.22803, 0.22145, 0.216989])
                frame = (frame - mean) / std
                frames.append(frame)
                idx_pos += 1
                
            current_frame += 1
            
        cap.release()
        
        # Pad if video was too short
        while len(frames) < self.num_frames:
            frames.append(np.zeros((*self.resize, 3), dtype=np.float32))
            
        # Stack to (T, H, W, C)
        video_tensor = np.stack(frames)
        # Convert to (C, T, H, W) for PyTorch 3D CNNs
        video_tensor = np.transpose(video_tensor, (3, 0, 1, 2))
        return torch.tensor(video_tensor, dtype=torch.float32)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            video_tensor = self._extract_frames(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            video_tensor = torch.zeros((3, self.num_frames, *self.resize), dtype=torch.float32)
            
        return video_tensor, torch.tensor([label], dtype=torch.float32)


def train_model():
    # 1. Setup Data
    dataset_path = "data/Violence Fight Detection dataset/RWF-2000"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    batch_size = 4  # Small batch size to fit in 6GB VRAM (RTX 4050)
    train_dataset = RWFDataset(dataset_path, split="train")
    val_dataset = RWFDataset(dataset_path, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    # Modify final fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1)
    )
    model = model.to(device)

    # 3. Setup Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scaler = torch.cuda.amp.GradScaler() # Mixed precision training to save VRAM

    # 4. Training Loop
    epochs = 10
    best_val_loss = float('inf')
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "rwf_r3d18_best.pth")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({"loss": loss.item()})
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_loss /= len(val_dataset)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print("Saved new best model!")

    # 5. Export to ONNX
    print("Exporting best model to ONNX...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    class InferenceModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        def forward(self, x):
            return torch.sigmoid(self.base_model(x))
            
    inference_model = InferenceModel(model)
    inference_model.eval()
    
    # Dummy input: (Batch=1, Channels=3, Time=16, Height=112, Width=112)
    dummy_input = torch.randn(1, 3, 16, 112, 112).to(device)
    onnx_path = os.path.join(model_dir, "violence_model.onnx")
    
    torch.onnx.export(
        inference_model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Successfully exported ONNX model to {onnx_path}")
    print("Training Complete! The system is now ready for production.")

if __name__ == "__main__":
    train_model()
