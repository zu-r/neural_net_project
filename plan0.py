import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomAffine, ColorJitter, RandomApply
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import rbf

torch.manual_seed(23)

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
                        for j in range(image_data.shape[1] // 20):
                            image = image_data[:, j].reshape(40, 40).T
                            label = np.argmax(label_data[:, j]) if label_data.ndim > 1 else label_data[j]
                            self.images.append(image)
                            self.labels.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        if len(self.images) == 0:
            raise ValueError("No images were loaded from the dataset. Check your .mat files and file paths.")
        print(f"Total images loaded: {len(self.images)}")
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

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.C1 = LeNetConv2d(1, 6, kernel_size=5, stride=1)
        self.S2 = LeNetConv2d(6, 6, kernel_size=2, stride=2)
        self.C3 = LeNetConv2d(6, 16, kernel_size=5, stride=1)
        self.S4 = LeNetConv2d(16, 16, kernel_size=2, stride=2)
        self.C5 = LeNetLinear(16*5*5, 120)
        self.F6 = LeNetLinear(120, 84)
        self.Rbf = rbf.Gaussian()

    def forward(self, x):
        x = 1.7159 * torch.tanh(self.C1(x) * 2/3)
        x = 1.7159 * torch.tanh(self.S2(x) * 2/3)
        x = 1.7159 * torch.tanh(self.C3(x) * 2/3)
        x = 1.7159 * torch.tanh(self.S4(x) * 2/3)
        x = x.view(-1, 16*5*5)
        x = 1.7159 * torch.tanh(self.C5(x) * 2/3)
        x = 1.7159 * torch.tanh(self.F6(x) * 2/3)
        x = self.Rbf(x)
        return x

class LeNetConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(LeNetConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        f_in = kernel_size * kernel_size * in_channels
        nn.init.uniform_(self.conv.weight, a=-2.4/f_in, b=2.4/f_in)

    def forward(self, x):
        return self.conv(x)

class LeNetLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(LeNetLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        f_in = in_features
        nn.init.uniform_(self.linear.weight, a=-2.4/f_in, b=2.4/f_in)

    def forward(self, x):
        return self.linear(x)

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.min(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return (100 * correct / total)


def train_model(model, train_loader, criterion, optimizer, test_loader, num_epochs=10):
    model.train()
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
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

class LeNetLoss(nn.Module):
    def __init__(self):
        super(LeNetLoss, self).__init__()

    def forward(self, output, labels):
        j = torch.tensor([1.0])
        yDp = output[range(output.size(0)), labels]
        reg = torch.log(torch.exp(-j) + torch.exp(-output).sum(dim=1))
        return (yDp + reg).mean()

def main():
    transform = Compose([Resize((32,32)), ToTensor(), Normalize((0.5,), (0.5,))])

    train_dataset = AffNISTTestDataset('training_batches/', transform=transform)
    test_dataset = AffNISTTestDataset('test_batches/', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = LeNet(num_classes=10)
    criterion = LeNetLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02)

    train_model(model, train_loader, criterion, optimizer, test_loader, num_epochs=10)
    test_model(model, test_loader)
    torch.save(model.state_dict(), 'LeNet0.pth')


if __name__ == "__main__":
    main()
