from torch.utils.data import DataLoader
from datasets import FrameImageDataset, FrameVideoDataset
from torchvision import transforms as T

# root_dir = '/work3/ppar/data/ucf101'
root_dir = "ufc10"

transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)

for video_frames, labels in frameimage_loader:
    print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]