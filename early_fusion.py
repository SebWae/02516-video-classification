import os
import pandas as pd
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

# --------------------------
# Dataset (FrameVideoDataset)
# --------------------------
class FrameVideoDataset(Dataset):
    def __init__(self, root_dir='/dtu/datasets1/02516/ufc10/', split='train', transform=None, stack_frames=True):
        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.transform = transform
        self.stack_frames = stack_frames
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.basename(video_path).split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        # Path to extracted frames
        frames_dir = video_path.replace('videos', 'frames').replace('.avi', '')
        frames = self.load_frames(frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        else:
            frames = [T.ToTensor()(frame) for frame in frames]

        if self.stack_frames:
            # Stack as tensor [C, T, H, W]
            frames = torch.stack(frames).permute(1, 0, 2, 3)

        return frames, label

    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_path = os.path.join(frames_dir, f'frame_{i}.jpg')
            frame = Image.open(frame_path).convert("RGB")
            frames.append(frame)
        return frames

# --------------------------
# Early Fusion 2D CNN Model
# --------------------------
class EarlyFusion2DCNN(nn.Module):
    def __init__(self, num_classes):
        super(EarlyFusion2DCNN, self).__init__()
        self.cnn2d = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # → [B*T, 32, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → [B*T, 32, 32, 32]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # → [B*T, 64, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → [B*T, 64, 16, 16]
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input shape: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)      # [B, T, C, H, W]
        x = x.reshape(B * T, C, H, W)     # [B*T, C, H, W]

        features = self.cnn2d(x)          # [B*T, 64, 16, 16]
        features = features.view(B, T, -1)  # [B, T, features]
        features = torch.mean(features, dim=1)  # [B, features]

        out = self.fc(features)           # [B, num_classes]
        return out

# --------------------------
# Train / Validation Loops
# --------------------------
def train(model, device, train_loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), correct / total

def validate(model, device, val_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(val_loader), correct / total

# --------------------------
# Main Function
# --------------------------
def main():
    root_dir = '/dtu/datasets1/02516/ufc10/'
    batch_size = 16
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])

    # Load datasets
    train_dataset = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames=True)
    val_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.df['label'].unique())
    print("Number of classes:", num_classes)

    model = EarlyFusion2DCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, device, val_loader, criterion)

        print(f"Epoch [{epoch}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

if __name__ == '__main__':
    main()
