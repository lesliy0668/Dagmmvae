import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.cluster import MeanShift
import pandas as pd

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def mean_shift(x, bandwidth=100,post=False,points=None):
    ms = MeanShift(bandwidth=bandwidth,bin_seeding=False, n_jobs=8,max_iter=5)
    ms.fit(x)
    #print ('time for clustering', time.time() - tic)
    res = ms.labels_
    
    cluster_centers = ms.cluster_centers_
    l = ms.labels_
    l = np.unique(l)
    return len(l)

def load_data(data_path):
    filetype = data_path.split(".")[-1]
    print("filetype:",filetype)
    if filetype == "txt":
        
        data = np.loadtxt(data_path)
        labels = data[:,-1]
        data = data[:,:-1]
    else:
        df = pd.read_excel(data_path)
        clomns_name = df.columns.tolist()
        nouse_names = ['Unnamed: 0', '个人编号','是否欺诈']
        use_names = [name for name in clomns_name if name not in nouse_names]
        labels = np.array(pd.read_excel(data_path,usecols=["是否欺诈"]))
        labels = labels[:,-1]
        data = np.array(pd.read_excel(data_path,usecols=use_names))
    return data,labels