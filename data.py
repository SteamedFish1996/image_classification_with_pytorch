# -*- coding: utf-8 -*-
import os, random, shutil

def my_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def create_dir(tar_dir,labels):
    for i in labels:
        path = os.path.join(tar_dir,i)
        my_mkdir(path)
        path = os.path.join(tar_dir,i)
        my_mkdir(path)

def prepare_dataset(root_dir, train_dir, test_dir, rate):
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir) 
        os.makedirs(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir) 
        os.makedirs(test_dir)
    labels = []
    for i in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir,i)):
            labels.append(i)
    #创建数据存放目录
    create_dir(train_dir,labels)
    create_dir(test_dir,labels)
    for label in labels:
        data_dir = os.path.join(root_dir,label)
        images = os.listdir(data_dir)
        test_image_number=int(len(images)*rate)
        test_images = random.sample(images, test_image_number)
        for image in images:
            if image in test_images:
                shutil.copy(os.path.join(data_dir,image),os.path.join(test_dir,label))
            else:
                shutil.copy(os.path.join(data_dir,image),os.path.join(train_dir,label))


if __name__ == '__main__':
    root_dir = r'./image/site1/sample'
    train_dir = './data/train/'
    test_dir = './data/test/'
    rate=0.1   #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    prepare_dataset(root_dir, train_dir, test_dir, rate)