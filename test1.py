from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from p1main import *
import mnist
import torch
import numpy as np
import torchvision

def test(dataloader,model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.min(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"test accuracy: {100.0 * correct/total}%")
 
def main():
    pad = torchvision.transforms.Pad(2, fill=0, padding_mode='constant')
    mnist_test = mnist.MNIST(split="test", transform=pad)
    test_dataloader = DataLoader(mnist_test, batch_size=1, shuffle=False)
    model = torch.load("LeNet1.pth")
    test(test_dataloader, model)

if __name__=="__main__":
    main()