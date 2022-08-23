import torch 
import torchvision 
from tqdm import tqdm 
from torchvision import datasets 
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms 

#transforms 
transform = transforms.Compose([
    transforms.ToTensor(),
])

def dataloader_mnist_train(args):
    mnist_trainset = datasets.FashionMNIST(
        root=args['data_path'],
        train=True,
        download=True,
        transform=transform 
    )
    mnist_train_dataloader = DataLoader(
        mnist_trainset, 
        batch_size = args['batch_size'],
        shuffle=True
    )

    return mnist_train_dataloader 


def dataloader_mnist_valid(args):
    mnist_validset = datasets.FashionMNIST(
        root=args['data_path'],
        train=False,
        download=True, 
        transform =transform
    )
    mnist_valid_dataloader = DataLoader(
        mnist_validset,
        batch_size=args['batch_size'],
        shuffle=False 
    )

    return mnist_valid_dataloader

DATALOADER_DICT={}
DATALOADER_DICT["m"]={"train":dataloader_mnist_train,"val":dataloader_mnist_valid}

