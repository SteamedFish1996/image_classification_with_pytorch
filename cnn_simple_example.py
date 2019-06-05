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

from data import prepare_dataset
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible
EPOCH = 2              # train the training data n times
BATCH_SIZE = 50
LR = 0.001              # learning rate

dpi = 96
size = 700

train_transformations = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(32,padding=4),
        transforms.RandomResizedCrop(2000), #从原图像随机切割一张（size， size）的图像
        transforms.ToTensor(),
        #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

data_dir = './image/site1/sample'
train_dir = './data/train/'
test_dir = './data/test/'
rate=0.1   #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
prepare_dataset(data_dir , train_dir, test_dir, rate)
train_dataset = dset.ImageFolder(root=train_dir,transform=train_transformations)
test_dataset = dset.ImageFolder(root=test_dir,transform=train_transformations)
print(train_dataset.class_to_idx)
print(train_dataset.__len__)
print(train_dataset[1000][0].size())
print(train_dataset[900][1])  # 得到的是类别4，即'FALSE'
print(train_dataset.classes[train_dataset[900][1]])
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

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,num_workers=0)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

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
#train
for epoch in range(EPOCH):
    running_loss = 0.0
    for step, image_data in enumerate(train_loader):
        images, labels = image_data
        """print an example
        print(labels[0])
        img = transforms.ToPILImage()(images[0])
        plt.imshow(img)
        plt.title('Class:' + str(labels.numpy()[0]))
        plt.show()
        break
        """

        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        # forward + backward + optimize
        output = cnn(images)
        loss = loss_func(output, labels)
        optimizer.zero_grad()   # zeros the paramster gradients
        loss.backward()     # loss 求导
        optimizer.step()    # 更新参数
        running_loss += loss.item()  # tensor.item()  获取tensor的数值
        if step % 50 == 49:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step+ 1, running_loss / 50))  # 每2000次迭代，输出loss的平均值
            running_loss = 0.0

print('Finished Training')


if __name__ == "__main__":
    import time
    start_time = time.time() 
    #main()
    print('total time cost: %.2f' % (time.time() - start_time), 'seconds')     
