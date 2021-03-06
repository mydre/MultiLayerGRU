"""datasets.py"""
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.dataset import *
import numpy as np
import gzip
import pdb

class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"

class MyDataSet(Dataset):
    '''
    读取数据、初始化数据
    '''
    def __init__(self,folder,data_name,label_name,p_width,transform=None):
        (train_set,train_labels) = load_data(folder,data_name,label_name,p_width)
        self.train_set = train_set
        self.train_labels = train_labels
        self.p_width = p_width
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,)),])

    def __getitem__(self,index):#返回的是单个item（如一个图片）
        img,target = self.train_set[index],int(self.train_labels[index])
        #img = torch.from_numpy(img)
        #img = img.float().div(255).mul(2).add(-1)
        img = self.transform(img)
        img = img.view(1,self.p_width,self.p_width)
        return img,target

    def __len__(self):
        return len(self.train_set)


def load_data(data_folder,data_name,label_name,p_width):
    '''
    data_folder:文件目录
    data_name:数据文件名
    label_name: 标签数据文件名
    '''
    with gzip.open(os.path.join(data_folder,label_name),'rb') as lbpath:  # 从二进制文件中读取数据
        y_train = np.frombuffer(lbpath.read(),np.uint8,offset=8) # 偏移8个字节

    with gzip.open(os.path.join(data_folder,data_name),'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(),np.uint8,offset=16).reshape((-1,1,p_width,p_width)) # 偏移16个字节

    return (x_train,y_train)


def return_data2(args):
    trainDataset = MyDataSet('datasets/MNIST/','train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz',args.pixel_width)
    testDataset = MyDataSet('datasets/MNIST/','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz',args.pixel_width)
    batch_size = args.batch_size
    # train_loader
    train_loader = DataLoader(
        dataset=trainDataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=1,
        # pin_memory=True,
        # drop_last=True,
    )

    test_loader = DataLoader(
        dataset=testDataset,
        batch_size=batch_size,
        shuffle=False,
        #num_workers=1,
        #pin_memory=True,
        #drop_last=False
    )

    unlabel_loader = DataLoader(
        dataset=testDataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=1,
        # pin_memory=True,
        drop_last=True
    )
    data_loader = dict()
    data_loader['train'] = train_loader
    data_loader['test'] = test_loader
    data_loader['un_label'] = unlabel_loader
    return data_loader


if __name__ == '__main__':
    import argparse
    os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--dset_dir', type=str, default='datasets')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    data_loader = return_data(args)
