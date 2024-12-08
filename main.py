import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from io import BytesIO
from PIL import Image
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(BytesIO(row["image.bytes"]))
        image = image.convert("L")
        image = np.array(image).astype(np.uint8)
        label = row["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

def preprocess_image(image):
    image = Image.fromarray(image)
    image = image.resize((32, 32))
    image = np.asarray(image).astype(np.float32) / 255.0
    return torch.tensor(image).unsqueeze(0)

def load_data():
    splits = {
        'train': 'hf://datasets/ylecun/mnist/mnist/train-00000-of-00001.parquet',
        'test': 'hf://datasets/ylecun/mnist/mnist/test-00000-of-00001.parquet',
    }
    df_train = pd.read_parquet(splits['train'])
    df_test = pd.read_parquet(splits['test'])

    train_dataset = MNISTDataset(df_train, transform=preprocess_image)
    test_dataset = MNISTDataset(df_test, transform=preprocess_image)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.rbf = nn.Linear(84, 10)

        for layer in [self.conv1, self.conv2]:
            nn.init.uniform_(layer.weight, a=-2.4 / layer.in_channels, b=2.4 / layer.in_channels)
        for layer in [self.fc1, self.fc2, self.rbf]:
            nn.init.uniform_(layer.weight, a=-2.4 / layer.in_features, b=2.4 / layer.in_features)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = 1.7159 * torch.tanh(2 / 3 * x)
        x = self.pool2(self.conv2(x))
        x = 1.7159 * torch.tanh(2 / 3 * x)
        x = x.view(-1, 16 * 5 * 5)
        x = 1.7159 * torch.tanh(2 / 3 * self.fc1(x))
        x = 1.7159 * torch.tanh(2 / 3 * self.fc2(x))
        x = self.rbf(x)
        return x

def train_model(model, train_loader, test_loader, epochs=20, lr=0.001):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            with torch.no_grad():
                for param in model.parameters():
                    param -= lr * param.grad

            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

        # evaluate_model(model, test_loader)


def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}%")

if __name__ == "__main__":
    train_loader, test_loader = load_data()
    model = LeNet5()

    train_model(model, train_loader, test_loader)
    evaluate_model(model, test_loader)

