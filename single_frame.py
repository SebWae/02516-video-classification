from torch.utils.data import DataLoader
from datasets import FrameImageDataset, FrameVideoDataset
from torchvision import transforms as T

transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
frameimage_dataset = FrameImageDataset(split='val', transform=transform)
frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)

print("Managed to load data!")