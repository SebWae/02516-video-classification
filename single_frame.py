import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import FrameImageDataset, FrameVideoDataset
from torchvision import transforms as T

img_size = 64
batch_size = 8
transform = T.Compose([T.Resize((img_size, img_size)),T.ToTensor()])
trainset = FrameImageDataset(split='train', transform=transform)
train_loader = DataLoader(trainset,  batch_size=batch_size, shuffle=False)
testset = FrameImageDataset(split='test', transform=transform)
test_loader = DataLoader(testset,  batch_size=batch_size, shuffle=False)

print("Managed to load data!")

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1), 
                     nn.ReLU(),
                     nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1), 
                     nn.ReLU(),
                     nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1), 
                     nn.ReLU(),
                     nn.BatchNorm2d(8))
        
        self.fc = nn.Sequential(nn.Linear(8*128*128, 10),
                   nn.ReLU(),
                   nn.Linear(10, 10),
                   nn.Softmax(dim=1))
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')