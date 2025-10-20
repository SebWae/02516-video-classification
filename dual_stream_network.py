import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from datasets_dual_stream import FrameImageDataset, OpticalFlowDataset, DualStreamDataset
import utils

# setting seed
utils.set_seed(261025)

# global settings
root_dir = '/dtu/datasets1/02516/ucf101_noleakage'
img_size = 412
batch_size = 10

# optimizer settings
window_size = 2
lr = 5e-6
weight_decay = 1e-4
factor = 0.3
patience = 5
dropout_rate = 0.5
n_epochs = 1
opt_settings = {"window_size": window_size,
                "lr": lr, 
                "weight_decay": weight_decay, 
                "factor": factor, 
                "patience": patience, 
                "dropout_rate": dropout_rate,
                "n_epochs": n_epochs}

# printing the training/optimizer settings
for param, val in opt_settings.items():
    print(f"{param}: {val}")

# transformations
transform = T.Compose([T.Resize((img_size, img_size)),T.ToTensor()])
# transform = T.Compose([
#     T.Resize((img_size, img_size)),
#     T.RandomHorizontalFlip(p=0.5),
#     # T.ColorJitter(brightness=0.2, contrast=0.2),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225])
# ])
# consider removing ColorJitter

# loading the train set
train_frames = FrameImageDataset(root_dir=root_dir, split='train', transform=transform)
train_flows = OpticalFlowDataset(root_dir=root_dir, split='train', transform=transform)
train_dual_stream = DualStreamDataset(frame_dataset=train_frames, flow_dataset=train_flows, window_size=window_size)
train_loader = DataLoader(train_dual_stream,  batch_size=batch_size, shuffle=False)

# loading the validation test
val_frames = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
val_flows = OpticalFlowDataset(root_dir=root_dir, split='val', transform=transform)
val_dual_stream = DualStreamDataset(frame_dataset=val_frames, flow_dataset=val_flows, window_size=window_size)
val_loader = DataLoader(val_dual_stream,  batch_size=batch_size, shuffle=False)

# loading the test set
test_frames = FrameImageDataset(root_dir=root_dir, split='test', transform=transform)
test_flows = OpticalFlowDataset(root_dir=root_dir, split='test', transform=transform)
test_dual_stream = DualStreamDataset(frame_dataset=test_frames, flow_dataset=test_flows, window_size=window_size)
test_loader = DataLoader(test_dual_stream,  batch_size=batch_size, shuffle=False)

# dual-stream network
class DualStreamNetwork(nn.Module):
    def __init__(self, dropout_rate=0.5, num_classes=10):
        super(DualStreamNetwork, self).__init__()
        
        self.conv_frame = nn.Sequential(
            # conv1 for frame data
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_flow = nn.Sequential(
            # conv1 for frame data
            nn.Conv2d(window_size*2*3, 96, kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.shared_conv = nn.Sequential(
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
            nn.Linear(512*12*12, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            # full7
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            # final output (10 classes)
            nn.Linear(2048, num_classes)
            )

    def forward_conv(self, x, conv_type='frame'):
        if conv_type == 'frame':
            x = self.conv_frame(x)
        else:
            x = self.conv_flow(x)
        x = self.shared_conv(x)
        return x

    def forward(self, x, conv_type='frame'):
        x = self.forward_conv(x, conv_type)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initializing the model (2D CNN)
model = DualStreamNetwork()
model.to(device)

# initializing Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# initializing learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=factor, patience=patience
)

# training the single frame CNN
out_dict = utils.train_dual_stream(model=model, 
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
eval_results = utils.eval_dual_stream(device=device, model=model, dataloader=test_loader)
print(f"Evaluation metrics on the test set: {eval_results}")

