from torch.utils.data import Dataset
import os
from skimage import io
import matplotlib.pyplot as plt
# 数据的加载
class Site1Dataset(Dataset):
    """站点一数据集"""
    classes = ['01', '02', '05', '07', 'FALSE', 'Other']
    sizes = []  # sizes = [200, 200, 200, 200, 200, 200]
    lenth = 0

    def __init__(self, root_dir, transform=None):
        """
        Initialize file path or list of file names.

        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        for label in self.classes:
            path = os.path.join(self.root_dir, label)
            self.sizes.append(len(os.listdir(path)))
        for i in self.sizes:
            self.lenth += i

    def __len__(self):
        """返回整个数据集的大小"""
        return self.lenth

    def __getitem__(self, index):
        """
        覆写这个方法使得dataset[i]可以返回数据集中第i个样本
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        """
        if index < self.sizes[0]:
            label_num = 0
            idx = index
        elif index < self.sizes[0] + self.sizes[1]:
            label_num = 1
            idx = index - self.sizes[0]
        elif index < self.sizes[0] + self.sizes[1] + self.sizes[2]:
            label_num = 2
            idx = index - (self.sizes[0] + self.sizes[1])
        elif index < self.sizes[0] + self.sizes[1] + self.sizes[2] + self.sizes[3]:
            label_num = 3
            idx = index - (self.sizes[0] + self.sizes[1] + self.sizes[2])
        elif index < self.sizes[0] + self.sizes[1] + self.sizes[2] + self.sizes[3] + self.sizes[4]:
            label_num = 4
            idx = index - (self.sizes[0] + self.sizes[1] + self.sizes[2] + self.sizes[3])
        path = os.path.join(self.root_dir, self.classes[label_num])
        filenames = os.listdir(path)
        filename = filenames[idx]
        img = io.imread(os.path.join(path, filename))
        sample = {'image': img, 'label': self.classes[label_num]}
        if self.transform:
            sample = self.transform(sample)
        return sample

def main():
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


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print('total time cost: %.2f' % (time.time() - start_time), 'seconds')