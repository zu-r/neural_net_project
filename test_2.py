from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model_2 import *
import mnist
import torch
import numpy as np
import torchvision

torch.manual_seed(23)

def test(dataloader, model):
    correct_predictions = 0 
    total_samples = 0 

    for batch, (image, label) in enumerate(dataloader):
        predictions = torch.argmax(model(image), dim=1)  

        correct_predictions += (predictions == label).sum().item()

        total_samples += label.size(0)

    test_accuracy = correct_predictions / total_samples * 100
    print("Test accuracy:", test_accuracy)

def main():
    pad=torchvision.transforms.Pad(2,fill=0,padding_mode='constant')
    mnist_test=mnist.MNIST(split="test",transform=pad)
    test_dataloader= DataLoader(mnist_test,batch_size=1,shuffle=False)
    model = torch.load(r"LeNet5-2.pth")
    test(test_dataloader,model)
    
if __name__=="__main__":
    main()