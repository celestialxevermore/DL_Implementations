import torch 
from torch.utils.data import DataLoader
import numpy as np
#from dataloaders.mnist import mnist_dataloader 
from torchvision import datasets
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),]
)

def dataloader_mnist_train(args):
    #train dataset
    mnist_trainset = datasets.FashionMNIST(
        root = args['data_path'], 
        train=True,
        download=True,
        transform = transform
    )
    #train dataloader 
    mnist_trainloader = DataLoader(
        mnist_trainset,
        batch_size = args['batch_size'],
        shuffle=True

    )
    return mnist_trainloader,len(mnist_trainset)

def dataloader_mnist_valid(args):
        #train dataset
    mnist_validset = datasets.FashionMNIST(
        root = args['data_path'], 
        train=False,
        download=True,
        transform = transform
    )
    #train dataloader 
    mnist_validloader = DataLoader(
        mnist_validset,
        batch_size = args['batch_size'],
        shuffle=False

    )
    return mnist_validloader,len(mnist_validset)


DATALOADER_DICT={}
DATALOADER_DICT["mnist"]={"train":dataloader_mnist_train,"val":dataloader_mnist_valid}
