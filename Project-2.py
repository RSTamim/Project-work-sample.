#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import string

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize


# # A Simple Convolutional Network

# In[ ]:


ts = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 5.5))
])


# In[ ]:


cifar_train = CIFAR10(root='./data/', download=True, train=True, transform=ts)
cifar_test = CIFAR10(root='./data/', download=True, train=False, transform=ts)


# In[ ]:


print(len(cifar_train))
print(len(cifar_test))


# In[ ]:


train_sampler = SubsetRandomSampler(np.arange(len(cifar_train)))
test_sampler = SubsetRandomSampler(np.arange(len(cifar_test)))


# In[ ]:


train_loader = DataLoader(cifar_train, 32, sampler=train_sampler)
test_loader = DataLoader(cifar_test, 32, sampler=test_sampler)


# In[ ]:


for x, y in train_loader:
    print(x.shape)
    print(y.shape)


# In[ ]:


classes = [
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


# In[ ]:


class LeNet5(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(2, 2)
        
        self.dense_stack = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))        
        x = torch.flatten(x, 1)
        x = self.dense_stack(x)
        return x


# In[ ]:


model = LeNet5(num_classes=10)


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)


# In[ ]:


print(len(train_loader.dataset))
print(len(train_loader))


# In[ ]:


epochs = 5

for e in range(epochs):
    
    running_loss = 0.0
    
    for i, (imgs, labels) in enumerate(train_loader):
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 500 == 499:
            print('[Epoch %d, Step %5d] loss: %.3f' %
                  (e + 1, i + 1, running_loss / 500))
            running_loss = 0.0


# In[ ]:


PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)


# In[ ]:


model = LeNet5()
model.load_state_dict(torch.load(PATH))


# In[ ]:


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# In[ ]:


test(test_loader, model, nn.CrossEntropyLoss())


# # A Custom Dataset and DataLoader¶
# 

# In[ ]:


class ASLDataset(Dataset):
    char_to_int = {c: ord(c) - ord('A') for c in string.ascii_uppercase}
    char_to_int['del'] = 26
    char_to_int['nothing'] = 27
    char_to_int['space'] = 28
    int_to_char = {value: key for key, value in char_to_int.items()}
        
    def __init__(self, directory: str, train: bool = True, transform=None, label_transform=None):
        super().__init__()
        
        self.directory = os.path.join(directory, 'train' if train else 'test')
        self.transform = transform
        self.label_transform = label_transform
        
        self.x = None
        self.y = None
        
        self._load_images()
    
    def __getitem__(self, idx):
        x, y = torchvision.io.read_image(self.x[idx]).type(torch.float32), self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        if self.label_transform:
            y = self.label_transform(y)
        
        return x, y
    
    def __len__(self):
        return len(self.y)
    
    def _load_images(self):
        self.x = []
        self.y = []
        
        for c in os.listdir(self.directory):
            class_name = c
            class_dir = os.path.join(self.directory, class_name)
            for img in os.listdir(class_dir):
                self.x.append(os.path.join(class_dir, img))
                self.y.append(self.char_to_int[class_name])
                
        self.y = torch.tensor(self.y, dtype=torch.int64)
    
    @staticmethod
    def get_classname(idx: int) -> str:
        return ASLDataset.int_to_char[idx]


# In[ ]:


ts = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# In[ ]:


asl_train = ASLDataset('./data/kaggle-data/asl', transform=ts)
asl_test = ASLDataset('./data/kaggle-data/asl', transform=ts, train=False)


# In[ ]:


print(len(asl_train))
print(len(asl_test))


# In[ ]:


train_sampler = SubsetRandomSampler(np.arange(len(asl_train)))
test_sampler = SubsetRandomSampler(np.arange(len(asl_test)))


# In[ ]:


train_loader = DataLoader(asl_train, 32, sampler=train_sampler)
test_loader = DataLoader(asl_test, 32, sampler=test_sampler)


# In[ ]:


for x, y in train_loader:
    print(x.shape)
    print(y.shape)


# # AlexNet

# In[ ]:


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# In[ ]:


model = AlexNet(29)


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)


# In[ ]:


print(len(train_loader.dataset))
print(len(train_loader))


# In[ ]:


epochs = 10

for e in range(epochs):
    running_loss = 0.0
    
    for i, (imgs, labels) in enumerate(train_loader):
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 7 == 6:
            print('[Epoch %d, Step %5d] loss: %.3f' %
                  (e + 1, i + 1, running_loss / 7))
            running_loss = 0.0


# In[ ]:


test(test_loader, model, nn.CrossEntropyLoss())


# In[ ]:


class_correct = list(0. for i in range(29))
class_total = list(0. for i in range(29))

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(29):
    print('Accuracy of %5s : %2d %%' % (
        ASLDataset.int_to_char[i], 100 * class_correct[i] / class_total[i]))


# # Pre-trained AlexNet and Transfer Learning¶
# 

# In[ ]:


from torchvision import models


# In[ ]:


model = models.alexnet(pretrained=True)


# In[ ]:


for param in model.parameters():
    param.requires_grad = False


# In[ ]:


print(model)


# In[ ]:


new_clf = nn.Sequential(
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=9216, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=4096, out_features=1000, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=1000, out_features=29, bias=True),
)


# In[ ]:


model.classifier = new_clf


# In[ ]:


print(model)


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)


# In[ ]:


epochs = 10

for e in range(epochs):
    running_loss = 0.0
    for i, (imgs, labels) in enumerate(train_loader):
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 7 == 6:
            print('[Epoch %d, Step %5d] loss: %.3f' %
                  (e + 1, i + 1, running_loss / 7))
            running_loss = 0.0


# In[ ]:


test(test_loader, model, nn.CrossEntropyLoss())


# # ResNet152¶
# 

# In[ ]:


model = models.resnet152(pretrained=True)


# In[ ]:


print(model)


# In[ ]:


new_fc = torch.nn.Sequential(
    nn.Linear(in_features=2048, out_features=1000, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=1000, out_features=29, bias=True),
)


# In[ ]:


model.fc = new_fc


# In[ ]:


print(model)


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)


# In[ ]:


epochs = 5

for e in range(epochs):
    running_loss = 0.0
    for i, (imgs, labels) in enumerate(train_loader):
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 7 == 6:
            print('[Epoch %d, Step %5d] loss: %.3f' %
                  (e + 1, i + 1, running_loss / 7))
            running_loss = 0.0

