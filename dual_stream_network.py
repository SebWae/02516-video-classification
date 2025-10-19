import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from datasets import FrameImageDataset
import utils

# setting seed
utils.set_seed(261025)

# size settings
img_size = 128
batch_size = 10

# optimizer settings
lr = 5e-6
weight_decay = 1e-4
factor = 0.3
patience = 5
dropout_rate = 0.5
n_epochs = 500
opt_settings = {"lr": lr, 
                "weight_decay": weight_decay, 
                "factor": factor, 
                "patience": patience, 
                "dropout_rate": dropout_rate,
                "n_epochs": n_epochs}

# printing the training/optimizer settings
for param, val in opt_settings.items():
    print(f"{param}: {val}")

# transformations
# transform = T.Compose([T.Resize((img_size, img_size)),T.ToTensor()])
transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.RandomHorizontalFlip(p=0.5),
    # T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
# consider removing ColorJitter

# loading the train set
trainset = FrameImageDataset(root_dir='/dtu/datasets1/02516/ufc101_noleakage', split='train', transform=transform)
train_loader = DataLoader(trainset,  batch_size=batch_size, shuffle=False)

# loading the validation test
valset = FrameImageDataset(root_dir='/dtu/datasets1/02516/ufc101_noleakage', split='val', transform=transform)
val_loader = DataLoader(valset,  batch_size=batch_size, shuffle=False)

# loading the test set
testset = FrameImageDataset(root_dir='/dtu/datasets1/02516/ufc101_noleakage', split='test', transform=transform)
test_loader = DataLoader(testset,  batch_size=batch_size, shuffle=False)

# 2D CNN
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),       
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv2
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv3
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),

            # conv4
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),

            # conv5
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc = nn.Sequential(
            # full6
            nn.Linear(512*4*4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            # full7
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            # final output (10 classes)
            nn.Linear(2048, 10)
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
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# initializing learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=factor, patience=patience
)

# training the single frame CNN
out_dict = utils.train_single_frame(model=model, 
                                    train_loader=train_loader, 
                                    val_loader=val_loader, 
                                    device=device, 
                                    optimizer=optimizer, 
                                    scheduler=scheduler, 
                                    num_epochs=n_epochs)
print(f"Evaluation metrics from training phase: {out_dict}")

# loading the saved model
model.load_state_dict(torch.load('best_model.pt'))
model.to(device)

# evaluating on the test set
eval_results = utils.eval(device=device, model=model, dataloader=test_loader)
print(f"Evaluation metrics on the test set: {eval_results}")

