#!/usr/bin/env python3

import torchvision
import torchvision.transforms as tt
import tarfile
import os, gzip
import pickle
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision import datasets, transforms

__all__ = [
    'load_cifar10',
    'load_svhn',
    'load_mnist',
    'load_fashionmnist',
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


def load_casia_webface(root='./'):
    
    with gzip.open(root, 'rb') as f:
        train_ds, test_ds = pickle.load(f)
    
    return train_ds, test_ds