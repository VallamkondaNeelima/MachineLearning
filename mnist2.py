import numpy as np
import torch
import helper
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
trainset = datasets.MNIST('MNIST_data/',download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
dataiter = iter(trainloader)
images,labels = dataiter.next()

x = images.view(64,-1)

class network(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(784,256)
        self.output = nn.Linear(256,10)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self,x):

        return self.softmax(self.output(self.sigmoid(self.hidden(x))))

print(network().forward(x))
