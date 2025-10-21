import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd

# -------------------------------
# 1. Dataset Class
# -------------------------------
class FrameVideoDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, num_frames=10):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_frames = num_frames
        
        # Load metadata
        csv_path = os.path.join(root_dir, 'metadata', f'{split}.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples from {split} set.")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_name = row['video_name']
        label = int(row['label'])
        action = row['action']

        frames_folder = os.path.join(self.root_dir, 'frames', self.split, action, video_name)
        frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])

        # Sample N frames evenly across the video
        frame_indices = torch.linspace(0, len(frame_files)-1, self.num_frames).long()
        selected_frames = [frame_files[i] for i in frame_indices]

        frames = []
        for f in selected_frames:
            img_path = os.path.join(frames_folder, f)
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        return torch.stack(frames), label  # [num_frames, C, H, W], label


# -------------------------------
# 2. Late Fusion Model
# -------------------------------
class LateFusionModel(nn.Module):
    def __init__(self, num_classes=10, fine_tune=True):
        super(LateFusionModel, self).__init__()

        # Load ResNet with the new weights API
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        base_model.fc = nn.Identity()  # remove classification layer
        self.feature_extractor = base_model
        self.classifier = nn.Linear(512, num_classes)

        # Optional: freeze earlier layers for transfer learning
        if fine_tune:
            for name, param in self.feature_extractor.named_parameters():
                # Freeze all except layer3, layer4, and avgpool
                if not any(x in name for x in ["layer3", "layer4"]):
                    param.requires_grad = False

    def forward(self, frames):
        # frames shape: [batch, num_frames, C, H, W]
        B, T, C, H, W = frames.size()
        frames = frames.view(B * T, C, H, W)

        # Extract features (trainable last layers)
        features = self.feature_extractor(frames)  # [B*T, 512]

        # Average features across frames (late fusion)
        features = features.view(B, T, -1).mean(dim=1)

        logits = self.classifier(features)
        return logits


# -------------------------------
# 3. Training and Evaluation
# -------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# -------------------------------
# 4. Main Script
# -------------------------------
def main():
    root_dir = '/dtu/datasets1/02516/ufc10'
    batch_size = 4
    num_epochs = 5
    num_classes = 10
    num_frames = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, num_frames=num_frames)
    val_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, num_frames=num_frames)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = LateFusionModel(num_classes=num_classes, fine_tune=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")

    # Save trained model
    torch.save(model.state_dict(), "late_fusion_model.pth")
    print("✅ Model saved as late_fusion_model.pth")


if __name__ == '__main__':
    main()
