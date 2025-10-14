import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from datasets import FrameImageDataset, FrameVideoDataset
from torchvision import transforms as T

# root_dir = '/work3/ppar/data/ucf101'
root_dir = "ufc10"

transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='train', transform=transform)
frameimage_loader = DataLoader(frameimage_dataset,  batch_size=1, shuffle=False)

# Get one batch
images, labels = next(iter(frameimage_loader))

# Show the first image in the batch
img = images[0].permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
label = labels[0]
if hasattr(label, "item"):
    label = label.item()
elif hasattr(label, "__getitem__"):
    label = label[0]
else:
    label = int(label)

plt.imshow(img)
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()