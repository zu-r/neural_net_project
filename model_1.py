import io
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import p1rbf as p1rbf
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib .pyplot as plt
from matplotlib.ticker import MaxNLocator
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
        image = (image + 1) * (255.0 / 2)
        return image, torch.tensor(label, dtype=torch.long)
    
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.C1 = LeNetConv2d(1, 6, kernel_size=5, stride=1)
        self.S2 = LeNetConv2d(6, 6, kernel_size=2, stride=2)
        self.C3 = LeNetConv2d(6, 16, kernel_size=5, stride=1)
        self.S4 = LeNetConv2d(16, 16, kernel_size=2, stride=2)
        self.C5 = LeNetLinear(16*5*5, 120)
        self.F6 = LeNetLinear(120, 84)
        self.Rbf = p1rbf.Gaussian()

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

class LeNetLoss(nn.Module):
    def __init__(self):
        super(LeNetLoss, self).__init__()

    def forward(self, output, labels):
        j = torch.tensor([0.1])
        yDp = output[range(output.size(0)), labels]

        e_incorrect = torch.exp(-output)
        e_mask = torch.arange(10).unsqueeze(0).expand(output.size(0), -1) != labels.unsqueeze(1)
        e_incorrect = e_incorrect * e_mask

        reg = torch.log(torch.exp(-j) + e_incorrect.sum(dim=1))
        return (yDp + reg).mean()

def test_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.min(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Accuracy {100.0 * correct/total}%")
    return 100 - (100.0 * correct / total)

def main():
    splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
    df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
    transform = Compose([Resize((32,32)), ToTensor(), Normalize((0.5,), (0.5,))])

    train_dataset = LoadDataset(df_train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = LeNet()
    criterion = LeNetLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02)

    model.train()
    for epoch in range(20):
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{20}], Loss: {total_loss / len(train_loader):.4f}")
        test_model(model, train_loader)

    torch.save(model, "LeNet5_1.pth")

if __name__ == "__main__":
    main()