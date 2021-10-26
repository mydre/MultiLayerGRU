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

class MyDataSetArray(Dataset):
    '''
    读取数据、初始化数据
    '''
    def __init__(self,folder,data_name,label_name,p_width):
        (data,label) = load_data(folder,data_name,label_name,p_width)
        self.data= data
        self.label = label
        self.p_width = p_width
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self,index):#返回的是单个item（如一个图片）
        img,target = self.data[index],int(self.label[index])
        img = torch.from_numpy(img)
        img = img.view(-1,self.p_width,self.p_width)
        img = img.float()
        return img,target

    def __len__(self):
        return len(self.data)


def load_data(folder,data_name,label_name,p_width):
    '''
    data_folder:文件目录
    data_name:数据文件名
    label_name: 标签数据文件名
    '''
    data = np.load(os.path.join(folder,data_name))
    label = np.load(os.path.join(folder,label_name))

    return (data,label)


def return_data(args):
    trainDataset = MyDataSetArray('datasets/ARRAY/','data.npy','label.npy',args.pixel_width)
    testDataset = MyDataSetArray('datasets/ARRAY/','data_test.npy','label_test.npy',args.pixel_width)
    batch_size = args.batch_size
    # train_loader
    train_loader = DataLoader(
        dataset=trainDataset,
        batch_size=batch_size,
        shuffle=True,
        #num_workers=1,
        #pin_memory=True,
        # drop_last=True
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
