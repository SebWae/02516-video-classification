import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import os
import glob
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import json

class SimpleFrameVideoDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, n_sampled_frames=10, stack_frames=True):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.n_sampled_frames = n_sampled_frames
        self.stack_frames = stack_frames
        
        # Get video paths
        self.video_paths = sorted(glob.glob(f'{root_dir}/videos/{split}/*/*.avi'))
        print(f"Found {len(self.video_paths)} videos in {split} split")
        
        # Create class mapping
        self.class_to_idx = self._create_class_mapping()
        self.num_classes = len(self.class_to_idx)
        
        print(f"Found {self.num_classes} classes: {list(self.class_to_idx.keys())}")
        
    def _create_class_mapping(self):
        """Create mapping from class directory names to numerical labels"""
        # Extract class names from video paths
        class_names = set()
        for video_path in self.video_paths:
            class_name = video_path.split('/')[-2]  # Parent directory name
            class_names.add(class_name)
        
        class_names = sorted(list(class_names))
        mapping = {class_name: idx for idx, class_name in enumerate(class_names)}
        print(f"Class to label mapping: {mapping}")
        return mapping
    
    def get_class_names(self):
        """Return list of class names in label order"""
        return [class_name for class_name, _ in sorted(self.class_to_idx.items(), key=lambda x: x[1])]
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_frames(self, frames_dir):
        """Load frames from the frames directory"""
        frames = []
        
        for i in range(1, self.n_sampled_frames + 1):
            frame_path = os.path.join(frames_dir, f'frame_{i}.jpg')
            
            if not os.path.exists(frame_path):
                print(f"Warning: {frame_path} not found")
                continue
                
            try:
                frame = Image.open(frame_path).convert("RGB")
                frames.append(frame)
            except Exception as e:
                print(f"Error loading {frame_path}: {e}")
                continue
        
        if len(frames) == 0:
            print(f"No frames found in {frames_dir}")
            return None
            
        # If there are fewer frames than expected, repeat the last frame
        while len(frames) < self.n_sampled_frames:
            frames.append(frames[-1])
            
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        
        # Extract class name from path
        class_name = video_path.split('/')[-2]
        label = self.class_to_idx[class_name]
        
        # Convert video path to frames path
        frames_dir = video_path.replace('/videos/', '/frames/').replace('.avi', '')
        
        # Load frames
        frames = self.load_frames(frames_dir)
        
        if frames is None:
            # Return a black frame if loading fails
            dummy_frame = Image.new('RGB', (112, 112), (0, 0, 0))
            frames = [dummy_frame] * self.n_sampled_frames
        
        # Apply transforms
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        if self.stack_frames:
            # Stack frames into a single tensor: [C, T, H, W]
            frames_tensor = torch.stack(frames, dim=1)  # [C, T, H, W]
        else:
            frames_tensor = frames
            
        return frames_tensor, label

class CNN_3D(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(CNN_3D, self).__init__()
        
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization for all layers
        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(128)
        self.bn5 = nn.BatchNorm3d(256)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class Improved3DCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, dropout_rate=0.3):
        super(Improved3DCNN, self).__init__()
        
        #Different kernel sizes for temporal and spatial dimensions
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=(3,7,7), padding=(1,3,3))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3,5,5), padding=(1,2,2))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3,3,3), padding=1)
        
        # Pooling layers to reduce dimensions
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        
        # Batch normalization
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64) 
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        
        # Adaptive pooling and classifier with more capacity
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def train_model_with_early_stopping(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Early stopping parameters
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    min_epochs = 10
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"Early stopping enabled: patience={patience}, min_epochs={min_epochs}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss):
                print(f"NaN loss at batch {batch_idx}!")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        train_accuracy = 100 * correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        improved = False
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            improved = True
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  >>> New best model saved :) Val Acc: {val_accuracy:.2f}%")
            improved = True
        
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, '
              f'Patience: {patience_counter}/{patience}')
        
        # Early stopping check
        if epoch >= min_epochs and patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    # Load best model
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Acc')
    ax2.plot(val_accuracies, label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_on_test_set(model, test_loader, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(test_loader):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                current_acc = 100 * correct / total
                print(f'Batch {batch_idx}/{len(test_loader)}, Running Accuracy: {current_acc:.2f}%')
    
    test_accuracy = 100 * correct / total
    print(f'\nFinal Test Accuracy: {test_accuracy:.2f}%')
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of samples'})
    
    plt.title(f'Test Set Confusion Matrix\nOverall Accuracy: {test_accuracy:.2f}%', 
              fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('test_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print classification report
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, digits=4)
    print(report)
    
    return all_predictions, all_labels, cm, test_accuracy

def main():
    root_dir = '/dtu/datasets1/02516/101_noleakage/'

    print(f"\nContents of {root_dir}:")
    for item in os.listdir(root_dir):
        print(f"  {item}")
    
    HEIGHT, WIDTH = 118, 118
    
    transform = T.Compose([
        T.Resize((HEIGHT, WIDTH)),
        T.ToTensor(),
        T.RandomHorizontalFlip(p=0.5),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SimpleFrameVideoDataset(root_dir=root_dir, split='train', 
                                           transform=transform, stack_frames=True)
    val_dataset = SimpleFrameVideoDataset(root_dir=root_dir, split='val', 
                                         transform=transform, stack_frames=True)
    test_dataset = SimpleFrameVideoDataset(root_dir=root_dir, split='test', 
                                          transform=transform, stack_frames=True)
    
    # Get class names from dataset
    class_names = train_dataset.get_class_names()
    num_classes = len(class_names)
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=0)
    
    # Initialize model with dynamic num_classes
    model = Improved3DCNN(num_classes=num_classes)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        sample_input = torch.randn(2, 3, 10, 112, 112)
        output = model(sample_input)
        print(f"Forward pass successful! Output shape: {output.shape}")
    
    # Train model
    print("\n" + "="*60)
    print("STARTING TRAINING WITH EARLY STOPPING")
    print("="*60)
    
    train_losses, val_losses, train_accuracies, val_accuracies = train_model_with_early_stopping(
        model, train_loader, val_loader, num_epochs=100, learning_rate=0.0001
    )
    
    print("Training completed!")
    print(f"Best validation accuracy: {max(val_accuracies):.2f}%")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Evaluate on test set
    predictions, labels, cm, test_accuracy = evaluate_on_test_set(model, test_loader, class_names)
    
    # Save model
    torch.save(model.state_dict(), 'final_model.pth')
    print(f'\nModel saved as final_model.pth')
    
    # Save results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_accuracy,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'num_classes': num_classes,
        'total_parameters': total_params
    }
    
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('Results saved as experiment_results.json')

if __name__ == '__main__':
    main()