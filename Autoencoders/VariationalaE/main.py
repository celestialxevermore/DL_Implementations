import torch 
import os 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import argparse 
import matplotlib 
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms 
from src import modeling 
from src import loss_
from dataloaders.data_dataloaders import DATALOADER_DICT

from torchvision import datasets
from torch.utils.data import DataLoader 
from torchvision.utils import save_image
#construct the argument parser and parser the arguments
from tqdm import tqdm 
matplotlib.style.use('ggplot')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

#torch.distributed.init_process_group(backend="nccl")

# args takin
def get_args(description='SparseAE on Reconstructing the images'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-e','--epochs',type=int,default=10,help='number of epochs to train our network')
    parser.add_argument('-l','--reg_param',type=float,default=0.001,help = 'regularization parameter lambda')
    parser.add_argument('--batch_size',type=int,help="batch")
    parser.add_argument('--learning_rate',type=float,help="learning rate")
    parser.add_argument('--data_path',type=str,help="the place to which the datasets are stored")
    parser.add_argument('--output_path',type=str,help="output directory")
    

    args = vars(parser.parse_args())
    #print(args.keys())
    #a=input()
    return args 

#we need bin, optimizer

def init_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#We need the model, and GPU device
def load_model(args,device):
    model = modeling.LinearVAE().to(device)
    return model

def prep_optimizer(args,model,device):
    optimizer = optim.Adam(model.parameters(),lr=args['learning_rate']) 
    return optimizer 

def train_epoch(epoch,args,model,device,optimizer,train_dataloader):
    print("Training ... ")
    print(train_dataloader)
    print(train_dataloader.dataset)
    model.train()
    running_loss = 0.0 
    for ind, data in tqdm(enumerate(train_dataloader),total=int(len(train_dataloader.dataset)/args['batch_size'])):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0),-1) # batch x 784
        optimizer.zero_grad()
        reconstruction,mu,log_var = model(data)

        #Loss
        bce_loss = loss_.get_bce_loss(reconstruction,data)
        kld_loss = loss_.get_KLD_loss(mu,log_var)
        loss = bce_loss + kld_loss
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    train_loss = running_loss / len(train_dataloader.dataset)
    print(f"Train Loss: {train_loss:.3f}")
    save_image(reconstruction.cpu().data,f"{args['output_path']}/train_images/train{epoch}.png")
    return train_loss 
        

def valid_epoch(epoch,args,model,device,optimizer,valid_dataloader):
    print("Evaluating ...")
    model.eval()
    running_loss = 0.0 
    print(valid_dataloader)
    with torch.no_grad():
        for ind,data in tqdm(enumerate(valid_dataloader),total = int(len(valid_dataloader.dataset)/args['batch_size'])):
            data , _ = data 
            data = data.to(device)
            print("data shape 1 :",data.shape)
            data = data.view(data.size(0),-1)
            print("data shape :",data.shape)
            
            reconstruction, mu,log_var = model(data)

            #loss 
            bce_loss = loss_.get_bce_loss(reconstruction,data)
            kld_loss = loss_.get_KLD_loss(mu,log_var)
            loss = bce_loss + kld_loss 
            running_loss += loss.item()
            
            #각 배치의 마지막을 저장 
            if ind == int(len(valid_dataloader.dataset)/args['batch_size'])-1:
                num_rows = 8 
                both = torch.cat((data.view(args['batch_size'], 1, 28, 28)[:8], 
                                  reconstruction.view(args['batch_size'], 1, 28, 28)[:8]))
                #save_image(outputs,f"{args['output_path']}/images/reconstruction{epoch}.png")
                save_image(both.cpu(), f"{args['output_path']}/output{epoch}.png",nrow =num_rows)

    val_loss = running_loss / len(valid_dataloader.dataset) 
    return val_loss       

def main():
    args = get_args()
    print("epochs : {}".format(args['epochs']))
    print("batch size : {}".format(args['batch_size']))
    print("learning_rate : {}".format(args['learning_rate']))
    device = init_device()
    # device, model, optimizer 
    model = load_model(args,device)
    #optimizer 
    optimizer = prep_optimizer(args,model,device)
    
    train_dataloader = DATALOADER_DICT["m"]["train"](args)
    valid_dataloader = DATALOADER_DICT["m"]["val"](args)
    
    
    train_loss=[]
    valid_loss=[]

    for epoch in range(args['epochs']):
        print(f"Epoch {epoch+1} of {args['epochs']}")
        train_epoch_loss = train_epoch(epoch,args,model,device,optimizer,train_dataloader)
        valid_epoch_loss = valid_epoch(epoch,args,model,device,optimizer,valid_dataloader)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {valid_epoch_loss:.4f}")





if __name__ == '__main__':
    main() 

