#!/usr/bin/env python3

from sklearn import datasets as sklearn_dataset
import numpy as np
import torchvision
import torchvision.transforms as tt
import tarfile
import os, gzip
import pickle
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR100

__all__ = [
    'load_cifar10',
    'load_svhn',
    'load_mnist',
    'load_fashionmnist',
    'load_cifar100',
    'load_olivetti',
    'load_casia_webface'
]
    
    
def load_cifar10(root = './'):
    transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Look into the data directory
    data_dir = os.path.join(root, 'cifar10')
    
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, data_dir+'/')

    
    # Extract from archive
    with tarfile.open(data_dir+'/cifar10.tgz', 'r:gz') as tar:
        tar.extractall(path=root)
    
    
    # Look into the data directory
    # data_dir = os.path.join(root, 'cifar10')
    #print(os.listdir(data_dir))
    #classes = os.listdir(data_dir + "/train")
    
    #train_ds = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
    #valid_ds = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)
    train_ds = ImageFolder(data_dir+'/train', transform)
    valid_ds = ImageFolder(data_dir+'/test', transform)
    return train_ds, valid_ds

def load_svhn(root = './'):
    transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ])
    
    train_ds = torchvision.datasets.SVHN(root=root, split='train', download=True, transform=transform)
    valid_ds = torchvision.datasets.SVHN(root=root, split='test', download=True, transform=transform)
    
    return train_ds, valid_ds

def load_mnist(root = './'):
    transform = tt.Compose([
        tt.ToTensor()])#,
        #tt.Normalize((0.5, ), (0.5,))
   # ])
    
    train_ds = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    valid_ds = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)

    return train_ds, valid_ds

def load_fashionmnist(root = './'):
    
    # Define the transformation to be applied to the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training dataset
    train_ds = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)

    # Download and load the test dataset
    valid_ds = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)
    
    return train_ds, valid_ds


class CustomCIFAR100(CIFAR100):
    def __init__(self, root, train, download, transform):
        super().__init__(root = root, train = train, download = download, transform = transform)
        self.coarse_map = {
            0:[4, 30, 55, 72, 95],
            1:[1, 32, 67, 73, 91],
            2:[54, 62, 70, 82, 92],
            3:[9, 10, 16, 28, 61],
            4:[0, 51, 53, 57, 83],
            5:[22, 39, 40, 86, 87],
            6:[5, 20, 25, 84, 94],
            7:[6, 7, 14, 18, 24],
            8:[3, 42, 43, 88, 97],
            9:[12, 17, 37, 68, 76],
            10:[23, 33, 49, 60, 71],
            11:[15, 19, 21, 31, 38],
            12:[34, 63, 64, 66, 75],
            13:[26, 45, 77, 79, 99],
            14:[2, 11, 35, 46, 98],
            15:[27, 29, 44, 78, 93],
            16:[36, 50, 65, 74, 80],
            17:[47, 52, 56, 59, 96],
            18:[8, 13, 48, 58, 90],
            19:[41, 69, 81, 85, 89]
        }
        
    #def __len__(self):
    #    len(self.main_dataset)
        
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        coarse_y = None
        for i in range(20):
            for j in self.coarse_map[i]:
                if y == j:
                    coarse_y = i
                    break
            if coarse_y != None:
                break
        if coarse_y == None:
            print(y)
            assert coarse_y != None
        return x, coarse_y# y, coarse_y

def load_cifar100(root = './'):
    
    # Define the transformation to be applied to the data
    
    transform = transforms.Compose([
        # transforms.Resize(224),
        tt.ToTensor(),
        tt.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
    
    # Download and load the training dataset
    train_ds = CustomCIFAR100(root=root, train=True, transform=transform, download=True)

    # Download and load the test dataset
    test_ds = CustomCIFAR100(root=root, train=False, transform=transform, download=True)
    
    return train_ds, test_ds

def load_olivetti(root='./', test_data_ration = 0.2):
    
    filename = 'olivetti_face_data.pkl'
    filepath = f'{root}/{filename}'
    if not os.path.isfile(filepath):
        print(f'Dataset not found. Creating data and saving to: {filepath}')
        face_data = sklearn_dataset.fetch_olivetti_faces(data_home=root)
        X = face_data.data
        X = X.reshape(X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1])))
        Y = face_data.target
        
        test_data_ration = test_data_ration
        class_ids = list(set(Y))
        test_ids = []
        for cl in class_ids:
            temp_test_index = np.where(Y==cl)
            test_ids.extend(list(np.random.permutation(temp_test_index[0])[:round(temp_test_index[0].shape[0]*test_data_ration)]))
            
        X_test = X[test_ids, :, :]  
        Y_test = Y[test_ids]

        train_ids = [i for i in range(Y.shape[0]) if i not in test_ids]
        X_train = X[train_ids, :, :]  
        Y_train = Y[train_ids]
        
        X_train = torch.from_numpy(np.expand_dims(X_train, axis=1))
        X_test = torch.from_numpy(np.expand_dims(X_test, axis=1))
        
        train_ds = list(zip(X_train, Y_train))
        valid_ds = list(zip(X_test, Y_test))
        
        all_data = (train_ds, valid_ds)
        with open(f'{root}/{filename}', 'wb') as fp:
            pickle.dump(all_data, fp)
    else:
        print(f'Dataset exists. Loading data from: {filepath}')
        with open(filepath, 'rb') as fp:
            train_ds, valid_ds = pickle.load(fp)
        
    return train_ds, valid_ds

# def load_cifar100(root = './'):
    
#     # Define the transformation to be applied to the data
#     transform = transforms.Compose([
#         tt.ToTensor(),
#         tt.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
#                      (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
#     #(0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))])
#     # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
#     # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])#
#     # (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])#

#     # Download and load the training dataset
#     train_ds = datasets.CIFAR100(root=root, train=True, transform=transform, download=True)

#     # Download and load the test dataset
#     test_ds = datasets.CIFAR100(root=root, train=False, transform=transform, download=True)
    
#     return train_ds, test_ds

def load_casia_webface(root='./'):
    
    with gzip.open(root, 'rb') as f:
        train_ds, test_ds = pickle.load(f)
    
    return train_ds, test_ds