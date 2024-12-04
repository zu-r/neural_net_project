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
        image = Image.open(io.BytesIO(row['image.bytes']))
        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
    
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.C1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.S2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.C3 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.S4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.C5 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.F6 = nn.Linear(120, 84)
        self.RBF = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.C1(x))
        x = self.S2(x)
        x = F.relu(self.C3(x))
        x = self.S4(x)
        x = F.relu(self.C5(x))
        x = x.view(-1, 120)
        x = F.relu(self.F6(x))
        x = self.RBF(x)
        return x

class CustomLoss(nn.Module):
    def __init__(self, j=0.1):
        super(CustomLoss, self).__init__()
        self.j = j

    def forward(self, outputs, labels):
        correct = outputs[range(outputs.size(0)), labels]
        incorrect = outputs + self.j
        incorrect[range(outputs.size(0)), labels] = 0
        loss = torch.sum(self.j + incorrect - correct.unsqueeze(1), dim=1)
        return torch.mean(loss)

class SDLM(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, damping=1e-4):
        defaults = {'lr': lr, 'damping': damping}
        super(SDLM, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            damping = group['damping']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad.data
                hessian_diag = torch.autograd.grad(
                    grad, param, grad_outputs=torch.ones_like(grad), retain_graph=True, create_graph=True
                )[0].data
                
                param.data -= lr * grad / (hessian_diag + damping)
        
        return loss

splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["test"])

transform = Compose([
    Resize((32, 32)),
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

train_dataset = LoadDataset(df_train, transform=transform)
test_dataset = LoadDataset(df_test, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = LeNet5(num_classes=10)
criterion = CustomLoss(j=0.1)
optimizer = SDLM(model.parameters(), lr=0.01, damping=1e-4)

def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
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

train_model(model, train_loader, criterion, optimizer, num_epochs=20)
test_model(model, test_loader)
