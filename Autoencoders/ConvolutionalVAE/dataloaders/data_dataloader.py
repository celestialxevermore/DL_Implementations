import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
import torchvision 
from torchvision import datasets
from torch.utils.data import DataLoader 
from torchvision.utils import make_grid 


transform = transforms.Compose([ 
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])

def get_mnist_train_dataloader(args):
    mnist_trainset = datasets.FashionMNIST(
        root = args['input_dir'], 
        train = True,
        download=True,
        transform = transform
    )
    mnist_train_dataloader = DataLoader( 
        mnist_trainset,
        batch_size = args['batch_size'],
        shuffle=True,
    )
    return mnist_train_dataloader

def mnist_valid_dataloader(args):
    mnist_validset = datasets.FashionMNIST(
        root = args['input_dir'],
        train=False,
        download=True,
        transform = transform
    )
    mnist_valid_dataloader = DataLoader( 
        mnist_validset,
        batch_size = args['batch_size'], 
        shuffle = False,
    )
    return mnist_valid_dataloader 




DATALOADERDICT = {}

DATALOADERDICT["mnist"] = {"train":get_mnist_train_dataloader,"valid":mnist_valid_dataloader}
