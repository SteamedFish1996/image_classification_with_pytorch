import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


import os
from skimage import io


 
# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate


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

def main():
    train_data_dir = '.\image\site1\sample'
    site1_set = Site1Dataset(train_data_dir)
    print(site1_set.lenth)
    #数据预处理 定义训练集的转换，随机翻转图像，剪裁图像，应用平均和标准正常化方法
    train_transformations = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    site1_set = Site1Dataset(train_data_dir)

    #train_data=Site1Dataset(dir = train_data_dir , transform=transforms.ToTensor())

    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    #train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    cnn = CNN()
    if torch.cuda.is_available():
        cnn.cuda()      # Moves all model parameters and buffers to the GPU.
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    import time
    start_time = time.time()    #代码计时开始
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):

            # !!!!!!!! Change in here !!!!!!!!! #
            b_x = x.cuda()    # Tensor on GPU
            b_y = y.cuda()    # Tensor on GPU

            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = cnn(test_x)

                # !!!!!!!! Change in here !!!!!!!!! #
                pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU

                accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
    # 储存网络 
    torch.save(net1, '/pkls/cnn1.pkl')  # save entire net

"""
train_data_dir = '.\image\site1\sample'
site1_set = Site1Dataset(train_data_dir)
print(site1_set.lenth)
sample = site1_set.__getitem__(900)
print(sample['label'])
#io.imshow(sample['image'])
plt.figure()
# 使用灰度方式显示图片
plt.imshow(sample['image'])
plt.show()
"""


if __name__=="__main__":
    import time
    start_time = time.time() 
    #main()
    print('total time cost: %.2f' % (time.time() - start_time), 'seconds')     
