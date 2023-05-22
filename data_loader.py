import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math
from utils import load_data
import pandas as pd

def Normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    another_trans_data = data - mean
    another_trans_data = another_trans_data / std
    return another_trans_data

class KDD99Loader(object):
    def __init__(self, data_path, mode="train"):
        self.mode=mode
        data = np.load(data_path)

        labels = data["kdd"][:,-1]
        features = data["kdd"][:,:-1]
        
        N, D = features.shape
        
        normal_data = features[labels==1]
        normal_labels = labels[labels==1]

        N_normal = normal_data.shape[0]

        attack_data = features[labels==0]
        attack_labels = labels[labels==0]

        N_attack = attack_data.shape[0]

        randIdx = np.arange(N_attack)
        np.random.shuffle(randIdx)
        N_train = N_attack // 2

        self.train = attack_data[randIdx[:N_train]]
        self.train_labels = attack_labels[randIdx[:N_train]]

        self.test = attack_data[randIdx[N_train:]]
        self.test_labels = attack_labels[randIdx[N_train:]]

        self.test = np.concatenate((self.test, normal_data),axis=0)
        self.test_labels = np.concatenate((self.test_labels, normal_labels),axis=0)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]


    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
           return np.float32(self.test[index]), np.float32(self.test_labels[index])

class Hospital(object):
    def __init__(self, data_path, mode="train"):
        data,labels = load_data(data_path)
        print("shape:",data.shape)
        self.mode=mode
        features = data
        N, D = features.shape
        features = Normalize(features)
        normal_data = features[labels==0]
        normal_labels = labels[labels==0]

        N_normal = normal_data.shape[0]

        attack_data = features[labels==1]
        attack_labels = labels[labels==1]

        N_attack = attack_data.shape[0]
        print("len:",N_normal,N_attack)
        randIdx = np.arange(N_normal)
        np.random.shuffle(randIdx)
        N_train = N_normal-N_attack

        self.train = normal_data[randIdx[:N_train]]
        self.train_labels = normal_labels[randIdx[:N_train]]

        self.test = normal_data[randIdx[N_train:]]
        self.test_labels = normal_labels[randIdx[N_train:]]

        self.test = np.concatenate((self.test, attack_data),axis=0)
        self.test_labels = np.concatenate((self.test_labels, attack_labels),axis=0)


    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]


    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
           return np.float32(self.test[index]), np.float32(self.test_labels[index])
        

def get_loader1(data_path, batch_size, mode='train',typename="Hospital"):
    """Build and return data loader."""
    if typename == "Hospital":
        dataset = Hospital(data_path, mode)
        test_dataset = Hospital(data_path, "test")
        data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True)
        test_data_loader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=True)
        return data_loader,test_data_loader
    else:
        dataset = KDD99Loader(data_path, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader

def get_loader(data_path, batch_size, mode='train',typename="Hospital"):
    """Build and return data loader."""
    if typename == "Hospital":
        dataset = Hospital(data_path, mode)
    else:
        dataset = KDD99Loader(data_path, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader

def main():
    # data_path = 'dataset/features.xlsx'
    # df = pd.read_excel(data_path)
    # clomns_name = df.columns.tolist()
    # nouse_names = ['Unnamed: 0', '个人编号','是否欺诈']
    # use_names = [name for name in clomns_name if name not in nouse_names]
    # labels = np.array(pd.read_excel(data_path,usecols=["是否欺诈"]))
    # data = np.array(pd.read_excel(data_path,usecols=use_names))
    # print(labels.shape,data.shape)
    data_path = "dataset/features.txt"
    data = np.loadtxt(data_path)
    print(data.shape)
    return

if __name__ == "__main__":
    main()