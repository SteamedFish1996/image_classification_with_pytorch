# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:57:51 2019

@author: zzy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision.datasets as dset
from torchvision import transforms

import matplotlib.pyplot as plt

torch.manual_seed(1) #reproducible
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate

train_transformations = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

data_dir = '.\image\site1\sample'
dataset = dset.ImageFolder(root=train_data_dir,transform=train_transformations)

print(dataset.class_to_idx)
print(dataset.__len__)
print(dataset[1000][0].size())
print(dataset[900][1])  # 得到的是类别4，即'FALSE'
print(dataset.classes[dataset[900][1]])
"""# plot one example
plt.subplot(1, 2, 1)
img = transforms.ToPILImage()(dataset[0][0])
plt.imshow(img)
plt.title('Class:'+dataset.classes[0])
plt.subplot(1, 2, 2)
img2 = transforms.ToPILImage()(dataset[201][0])
plt.imshow(img2)
plt.title('Class:'+dataset.classes[1])
plt.show()"""

train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2),)
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),)
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


cnn = CNN()
if torch.cuda.is_available():   # Moves all model parameters and buffers to the GPU.
    cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted



if __name__=="__main__":
    import time
    start_time = time.time() 
    #main()
    print('total time cost: %.2f' % (time.time() - start_time), 'seconds')     
