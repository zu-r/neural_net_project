import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy.io import loadmat
from PIL import Image
import io
import os
import matplotlib.pyplot as plt


class LoadDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(io.BytesIO(row['image.bytes']))
        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

class AffNISTTestDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Load affNIST dataset from .mat files in the specified folder
        
        Args:
        folder_path (str): Path to folder containing .mat files
        transform (callable, optional): Optional transform to be applied to images
        """
        self.images = []
        self.labels = []
        
        # Validate folder path exists
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")
        
        # Loop through mat files and load data
        for i in range(1, 33):  # Assuming files are named 1.mat to 32.mat
            file_path = os.path.join(folder_path, f'{i}.mat')
            if os.path.exists(file_path):
                try:
                    data = loadmat(file_path)
                    affNISTdata = data.get('affNISTdata', None)
                    
                    if affNISTdata is not None:
                        # Extract image and label data
                        image_data = affNISTdata['image'][0, 0]
                        label_data = affNISTdata['label_one_of_n'][0, 0]
                        
                        # Convert images from vector to 2D and prepare labels
                        for j in range(image_data.shape[1]//20):
                            # Reshape image to 40x40 and transpose
                            image = image_data[:, j].reshape(40, 40).T
                            
                            # Get label (convert one-hot to index)
                            label = np.argmax(label_data[:, j]) if label_data.ndim > 1 else label_data[j]
                            
                            self.images.append(image)
                            self.labels.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        # Validate that images were loaded
        if len(self.images) == 0:
            raise ValueError("No images were loaded from the dataset. Check your .mat files and file paths.")
        
        print(f"Total test images loaded: {len(self.images)}")
        
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert to PIL Image
        image = Image.fromarray(self.images[idx].astype(np.uint8))
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# The rest of the model architecture remains the same as in the original script
class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),  
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)  
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

class LeNetWithReLUAndSTN(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetWithReLUAndSTN, self).__init__()
        self.stn = SpatialTransformer()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.stn(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, train_loader, criterion, optimizer, test_loader, num_epochs=10):
    model.train()
    train_losses = []
    test_errors = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        
        # Calculate and store test error rate
        accuracy = test_model(model, test_loader)
        error_rate = 100 - accuracy
        test_errors.append(error_rate)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # Plot loss and error rate
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs. Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), test_errors)
    plt.xlabel("Epoch")
    plt.ylabel("Test Error Rate (%)")
    plt.title("Test Error Rate vs. Epoch")

    plt.tight_layout()
    plt.show()

    return train_losses, test_errors


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    # Hugging Face MNIST dataset for training
    splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
    df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
    
    # Transform for both training and testing
    transform = Compose([
        Resize((32, 32)),  # Resize to match LeNet input
        ToTensor(), 
        Normalize((0.5,), (0.5,))
    ])

    # Create training dataset from Hugging Face
    train_dataset = AffNISTTestDataset('training_batches/', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Create test dataset from local test_batches folder
    test_dataset = AffNISTTestDataset('test_batches/', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model
    model = LeNetWithReLUAndSTN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Train and test
    train_model(model, train_loader, criterion, optimizer, test_loader=test_loader,num_epochs=10)
    test_model(model, test_loader)

if __name__ == "__main__":
    main()