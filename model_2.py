import io
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomAffine, ColorJitter, RandomApply, RandomHorizontalFlip, RandomRotation
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

torch.manual_seed(23)

class MnistTrainDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(io.BytesIO(row['image']['bytes']))
        label = row['label']
        if self.transform:
            image = self.transform(image)
        image = (image + 1) * (255.0 / 2)
        return image, torch.tensor(label, dtype=torch.long)

class AffNISTTestDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.images = []
        self.labels = []
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")
        for i in range(1, 33):
            file_path = os.path.join(folder_path, f'{i}.mat')
            if os.path.exists(file_path):
                try:
                    data = loadmat(file_path)
                    affNISTdata = data.get('affNISTdata', None)
                    if affNISTdata is not None:
                        image_data = affNISTdata['image'][0, 0]
                        label_data = affNISTdata['label_one_of_n'][0, 0]
                        for j in range(image_data.shape[1]//20):
                            image = image_data[:, j].reshape(40, 40)
                            label = np.argmax(label_data[:, j]) if label_data.ndim > 1 else label_data[j]
                            self.images.append(image)
                            self.labels.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        if len(self.images) == 0:
            raise ValueError("No images were loaded from the dataset. Check your .mat files and file paths.")
        print(f"Total test images loaded: {len(self.images)}")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx].astype(np.uint8))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        image = (image + 1) * (255.0 / 2)
        return image, torch.tensor(label, dtype=torch.long)

class LeNetWithDropout(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(84, num_classes)
        

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, criterion, optimizer, test_loader, num_epochs=10):
    model.train()
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)
        accuracy = test_model(model, test_loader)
        test_accuracies.append(accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, Train Acc: {100 * correct / total:.2f}%, Test Acc: {accuracy:.2f}%")

    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), 100 - np.array(train_accuracies))
    plt.xlabel("Epoch")
    plt.ylabel("Training Error Rate")
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), 100 - np.array(test_accuracies))
    plt.xlabel("Epoch")
    plt.ylabel("Test Error Rate")
    plt.tight_layout()
    plt.show()
    return train_losses, train_accuracies, test_accuracies

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
    # print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return accuracy


def custom_affine(img):
    # Random parameters for transformation
    rotation = random.uniform(-20, 20)  # Rotation in degrees
    shear = random.uniform(-0.2, 0.2)  # Uniform shear factor
    scale_x = random.uniform(0.8, 1.2)  # Horizontal expansion/shrinking
    scale_y = random.uniform(0.8, 1.2)  # Vertical expansion/shrinking
    translate_x = random.uniform(-10, 10)  # Horizontal translation
    translate_y = random.uniform(-10, 10)  # Vertical translation

    # Convert rotation angle to radians
    theta = math.radians(rotation)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # Compute the affine transformation matrix
    a = scale_x * cos_theta  # Combine horizontal scaling and rotation
    b = scale_x * -sin_theta
    c = translate_x
    d = scale_y * sin_theta  # Combine vertical scaling and rotation
    e = scale_y * cos_theta
    f = translate_y

    # Apply the affine transformation
    width, height = img.size
    transformed_img = img.transform(
        (width, height),
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BICUBIC,
    )

    # Ensure no ink falls outside bounds (optional step for AffNIST compatibility)
    transformed_img = transformed_img.crop((0, 0, 40, 40)).resize((40, 40))

    return transformed_img


def main():
    train_transform = Compose([
        RandomApply([custom_affine], p=0.75),
        RandomRotation(degrees=(-20, 20)),
        Resize((32, 32)),
        ToTensor(),
        Normalize((1.0,), (1.0,))
    ])

    test_transform = Compose([
        Resize((32, 32)),
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    torch.set_printoptions(threshold=float('inf'), linewidth=300, precision=0)

    df_train = pd.read_parquet("hf://datasets/ylecun/mnist/mnist/train-00000-of-00001.parquet")
    train_dataset = MnistTrainDataset(df_train, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for images, l in train_loader:
        for i in range(10):
            print(images[i])
            print(l[i])
        break

    print("------------------------------")

    test_dataset = AffNISTTestDataset('test_batches/', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = LeNetWithDropout(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.004, momentum=0.9)

    train_model(model, train_loader, criterion, optimizer, test_loader, num_epochs=10)
    torch.save(model, 'LeNet5_2.pth')

if __name__ == "__main__":
    main()