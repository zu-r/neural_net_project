import io
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

class LoadDataset(Dataset):
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
        return image, torch.tensor(label, dtype=torch.long)
    
class CustomSGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                # Update with momentum
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * param.grad
                param.data += self.velocity[i]

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
    
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(1920 * 1 * 1, 96) 
        self.fc2 = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.relu(self.layer3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["test"])

transform = Compose([Resize((32,32)), ToTensor(), Normalize((0.5,), (0.5,))])

train_dataset = LoadDataset(df_train, transform=transform)
test_dataset = LoadDataset(df_test, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = LeNet(num_classes=10)
criterion = nn.CrossEntropyLoss()

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
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

optimizer = CustomSGD(model.parameters(), lr=0.01, momentum=0.9)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Train and test using the custom optimizer
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
test_model(model, test_loader)
