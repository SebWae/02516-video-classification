import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from datasets import FrameImageDataset
import utils

# size settings
img_size = 128
batch_size = 10

# transformations
transform = T.Compose([T.Resize((img_size, img_size)),T.ToTensor()])

# loading the train set
trainset = FrameImageDataset(split='train', transform=transform)
train_loader = DataLoader(trainset,  batch_size=batch_size, shuffle=False)

# loading the validation test
valset = FrameImageDataset(split='val', transform=transform)
val_loader = DataLoader(valset,  batch_size=batch_size, shuffle=False)

# loading the test set
testset = FrameImageDataset(split='test', transform=transform)
test_loader = DataLoader(testset,  batch_size=batch_size, shuffle=False)

# 2D CNN
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((4, 4)))
        
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            )

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



# initializing the model (2D CNN)
model = Network()
model.to(device)

# initializing Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# initializing learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# training the single frame CNN
out_dict = utils.train_single_frame(model=model, 
                                    train_loader=train_loader, 
                                    val_loader=val_loader, 
                                    device=device, 
                                    optimizer=optimizer, 
                                    scheduler=scheduler, 
                                    num_epochs=20)
print(out_dict)

# loading the saved model
model_trained = model.load_state_dict(torch.load('best_model.pt'))

# evaluating on the test set
utils.eval(device=device, model=model_trained, dataloader=test_loader)


