import numpy as np
import torch
import helper
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
trainset = datasets.MNIST('MNIST_data/',download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
dataiter = iter(trainloader)
images,labels = dataiter.next()
print(images.shape)
print(labels.shape)
print(images[0].shape)
plt.imshow(images[0].numpy().squeeze(axis=0),cmap='Greys_r')
plt.show()

image = images[0].view(1,-1)
print(image.shape)

def softmax(x,y):
    return torch.exp(x-y)/(torch.sum(torch.exp(x-y),dim=1).view(-1,1))
w = torch.randn(28*28,256)
w1 = torch.randn(256,10)
b = torch.randn(1,256)
b1 = torch.randn(1,10)
h = softmax(torch.mm(image,w)+b,(torch.mm(image,w)+b).max())
print(h.shape)

o = torch.mm(h,w1)+b1
#print(o)
#print(o.shape)
print(softmax(o,o.max()))














