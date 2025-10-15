import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from datasets import FrameImageDataset, FrameVideoDataset

img_size = 64
batch_size = 10
transform = T.Compose([T.Resize((img_size, img_size)),T.ToTensor()])
trainset = FrameImageDataset(split='train', transform=transform)
train_loader = DataLoader(trainset,  batch_size=batch_size, shuffle=False)
testset = FrameImageDataset(split='test', transform=transform)
test_loader = DataLoader(testset,  batch_size=batch_size, shuffle=False)
print(f"Length of trainset: {len(trainset)}")
print(f"Length of testset: {len(testset)}")

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
        
        self.fc = nn.Sequential(nn.Linear(8*img_size*img_size, 10),
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

#We define the training as a function so we can easily re-use it.
def train(model, optimizer, num_epochs=10):
    def loss_fun(output, target):
        return F.nll_loss(torch.log(output), target)
    out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': []}
  
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        #For each epoch
        train_correct = 0
        train_loss = []
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Computing the average target across the frames (the target should be the same for all frames)
            average_target = torch.mean(target.float())
            #Forward pass your image through the network
            output = model(data)
            #Computing the average score for each class across the frames (which should be from the same video)
            average_output = torch.mean(output, dim=0)
            #Compute the loss
            loss = loss_fun(output, target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            #Compute how many were correctly classified
            predicted = average_output.argmax(0)
            train_correct += (average_target==predicted).sum().cpu().item()
        #Comput the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            average_target = torch.mean(target.float())
            with torch.no_grad():
                output = model(data)
                average_output = torch.mean(output, dim=0)
            test_loss.append(loss_fun(output, target).cpu().item())
            predicted = average_output.argmax(0)
            test_correct += (average_target==predicted).sum().cpu().item()
        out_dict['train_acc'].append(train_correct/len(trainset))
        out_dict['test_acc'].append(test_correct/len(testset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
    return out_dict

# initializing the model (2D CNN)
model = Network()
model.to(device)

# initializing the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# training and evaluating the single frame CNN
out_dict = train(model, optimizer, num_epochs=5)
print(out_dict)
