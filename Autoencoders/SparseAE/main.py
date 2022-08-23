from dataloaders.mnist import DATALOADER_DICT
import torch 
import torchvision
import torch.nn as nn 
import matplotlib 
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms 
import torch.nn.functional as F 
import torch.optim as optim 
import os 
import time 
import numpy as np 
import argparse 

from tqdm import tqdm 
from torchvision import datasets 
#from torch.utils.data import Dataloader 
from torchvision.utils import save_image 
from modules.loss import *
from modules.modeling import SparseAutoencoder


matplotlib.style.use('ggplot')

def get_args(description='SparseAE on Reconstructing the images'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-e','--epochs',type=int,default=10,help='number of epochs to train our network')
    parser.add_argument('-l','--reg_param',type=float,default=0.001,help = 'regularization parameter lambda')
    parser.add_argument('-sc','--add_sparse',type=str,default='yes',help='whether to add sparsity constraint or not')
    parser.add_argument('--rho',type=float)
    parser.add_argument('--batch_size',type=int,help="batch")
    parser.add_argument('--learning_rate',type=float,help="learning rate")
    parser.add_argument('--data_path',type=str,help="the place to which the datasets are stored")
    parser.add_argument('--output_path',type=str,help="output directory")
    

    args = vars(parser.parse_args())
    #print(args.keys())
    #a=input()
    return args 

# gpu불러오기
def init_device(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device 

#make the 'images' directory
def make_dir(args):
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

def save_outputs_image(img,name):
    img = img.view(img.size(0),1,28,28)
    save_image(img,name)

#모델(깡통) 불러오기
def load_model(args,device):
    model = SparseAutoencoder()
    model.to(device)

    return model

def prep_optimizer(model,args):
    param_optimizer = list(model.named_parameters())
    optimizer = optim.Adam(model.parameters(),lr=args['learning_rate'])
    return optimizer

def plot(train_loss,valid_loss,args):
    output_path = args['output_path']
    plt.figure(figsize=(10,7))
    plt.plot(train_loss,color='orange',label='train loss')
    plt.plot(valid_loss,color='red',label = 'validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_path+'/loss.png')


#훈련
def train_epoch(epoch,args,model,train_dataloader,device,optimizer):
    print("Training..")
    model.train()
    running_loss=0.0
    counter=0
    rho = args['rho']
    beta = args['reg_param']
        
    for ind, data in tqdm(enumerate(train_dataloader),total=int(len(train_dataloader.dataset)/train_dataloader.batch_size)):
        counter +=1 
        img, _ = data 
        img = img.to(device)
        img = img.view(img.size(0),-1)
        optimizer.zero_grad()
        outputs = model(img)
        
        mse_loss = get_mse_loss(outputs,img)
        if args['add_sparse']=='yes':
            sparsity = sparse_loss(model,device,rho,img)
            loss = mse_loss + beta * sparsity 
        else:
            loss = mse_loss 
        
        #print("loss :",loss)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 
    

    epoch_loss = running_loss / counter
    print(f"Train Loss: {epoch_loss:.3f}")
    save_outputs_image(outputs.cpu().data,f"{args['output_path']}/images/train{epoch}.png")

    return epoch_loss 

def eval_epoch(epoch,args,model,test_dataloader,device):
    print("Validating...")
    model.eval()
    running_loss = 0.0
    counter=0

    with torch.no_grad():
        for ind,data in tqdm(enumerate(test_dataloader),total = int(len(test_dataloader.dataset)/test_dataloader.batch_size)):
            counter+=1
            img, _  = data 
            img = img.to(device)
            img = img.view(img.size(0),-1)
            outputs = model(img)

            loss = get_mse_loss(outputs,img)
            running_loss += loss.item()
    epoch_loss = running_loss / counter 
    print(f"Valid Loss : {epoch_loss:.3f}")
    outputs = outputs.view(outputs.size(0),1,28,28).cpu().data
    save_image(outputs,f"{args['output_path']}/images/reconstruction{epoch}.png")

    return epoch_loss 





def main():
    args = get_args()
    
    #Initializing the base hyper parameters
    EPOCHS = args['epochs']
    BETA = args['reg_param']
    ADD_SPARSITY = args['add_sparse']
    RHO = args['rho']
    LEARNING_RATE = args['learning_rate']
    BATCH_SIZE = args['batch_size']

    device = init_device(args)
    model = load_model(args,device) #to(device)

  
    print(f"Add sparsity regularization: {ADD_SPARSITY}")
    ### dataloader loading 
    train_dataloader,train_length = DATALOADER_DICT["mnist"]["train"](args)
    valid_dataloader,valid_length = DATALOADER_DICT["mnist"]["val"](args)

    ### prepare the optimizer 
    optimizer = prep_optimizer(model,args)

    train_loss = []
    valid_loss = []
    start= time.time()
    for epoch in range(args['epochs']):
        print(f"Epoch {epoch+1} of {args['epochs']}")
        train_epoch_loss = train_epoch(epoch,args,model,train_dataloader,device,optimizer)
        val_epoch_loss = eval_epoch(epoch,args,model,valid_dataloader,device)
        train_loss.append(train_epoch_loss)
        valid_loss.append(val_epoch_loss)
    end = time.time()
    print(f"{(end-start)/60:.3} minutes")
    plot(train_loss,valid_loss,args)
    torch.save(model.state_dict(),f"{args['output_path']}/sparse_ae{args['epochs']}.pth")


if __name__ =="__main__":
    main()