import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision.models as models

from datasets import FrameImageDataset
import utils

# size settings
img_size = 128
batch_size = 10

# optimizer settings
lr = 1e-4
weight_decay = 5e-4
factor = 0.5
patience = 2
n_epochs = 20
opt_settings = {"lr": lr, 
                "weight_decay": weight_decay, 
                "factor": factor, 
                "patience": patience, 
                "n_epochs": n_epochs}

# printing the training/optimizer settings
for param, val in opt_settings.items():
    print(f"{param}: {val}")

# transformations
transform = T.Compose([T.Resize((img_size, img_size)),T.ToTensor()])
# transform = T.Compose([
#     T.Resize((img_size, img_size)),
#     T.RandomHorizontalFlip(),
#     T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
#     T.RandomRotation(15),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225])
# ])

# loading the train set
trainset = FrameImageDataset(split='train', transform=transform)
train_loader = DataLoader(trainset,  batch_size=batch_size, shuffle=False)

# loading the validation test
valset = FrameImageDataset(split='val', transform=transform)
val_loader = DataLoader(valset,  batch_size=batch_size, shuffle=False)

# loading the test set
testset = FrameImageDataset(split='test', transform=transform)
test_loader = DataLoader(testset,  batch_size=batch_size, shuffle=False)

# loading pre-trained ResNet18
model = models.resnet18(weights='IMAGENET1K_V1')

# adjusting the final layer to 10 classes
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 10)
)

# freezing base layers
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.fc.parameters():
#     param.requires_grad = True

# checking if GPU is available and defining device
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

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


