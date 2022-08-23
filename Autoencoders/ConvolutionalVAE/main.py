import torch 
import torch.nn as nn
import torch.nn.functional as F 
from tqdm import tqdm 
from src import modeling 
import torch.optim as optim 
import matplotlib 
import matplotlib.pyplot as plt 
import argparse
from torchvision.utils import make_grid
from src import utils 
from src import losses
from dataloaders.data_dataloader import DATALOADERDICT
matplotlib.style.use('ggplot')

def get_args(description = "getting the parameters needed for ConvVAE"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-e','--epochs',type=int,default=10,help='number of epochs to train our network')
    parser.add_argument('--learning_rate',type=float, default=1e-4,help="learning_rate")
    parser.add_argument('--batch_size',type=int,default=64,help='batch_size')
    parser.add_argument('--grid_images',type=list,default = [],help = 'a list to save all the reconstructed images')
    parser.add_argument("--input_dir",type=str,default="")
    parser.add_argument("--output_dir",type=str,default="")
    args = vars(parser.parse_args())
    return args

def init_device(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device 

def load_model(args,device):
    model = modeling.ConvVAE().to(device)
    return model 

def prep_optimizer(args,model,device):
    optimizer = optim.Adam(model.parameters(),lr=args['learning_rate']) 
    return optimizer 

def train_epoch(epoch,args,model, device, optimizer,train_dataloader):
    model.train()
    running_loss = 0.0 
    counter = 0 

    for ind,data in tqdm(enumerate(train_dataloader), total = int(len(train_dataloader.dataset)/args['batch_size'] )):
        counter +=1 
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()

        #forward
        reconstruction, mu, log_var = model(data)
        bce_loss = losses.get_bce_loss(reconstruction,data)
        kld_loss = losses.get_kld_loss(mu,log_var)
        loss = bce_loss + kld_loss 

        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter 
    return train_loss 


def valid_epoch(epoch,args,model, device, optimizer,valid_dataloader):
    model.eval() 
    running_loss = 0.0 
    counter = 0 
    with torch.no_grad():
        for ind, data in tqdm(enumerate(valid_dataloader),total = int(len(valid_dataloader.dataset)/args['batch_size'])):
            counter +=1 
            data = data[0]
            data = data.to(device)
            reconstruction,mu,log_var = model(data) #mu and log_var is from the encoded latent distribution
            bce_loss = losses.get_bce_loss(reconstruction,data)
            kld_loss = losses.get_kld_loss(mu,log_var)
            loss = bce_loss + kld_loss 
            running_loss += loss.item()

            if ind == int(len(valid_dataloader.dataset)/args['batch_size'])-1:
                recon_images = reconstruction 
    val_loss = running_loss / counter 

    return val_loss,recon_images 



def main():
    # what we need : args, model, optimizer, dataloader 
    args = get_args()
    device = init_device(args)
    model = load_model(args,device)
    
    optimizer = prep_optimizer(args,model,device)

    train_dataloader = DATALOADERDICT["mnist"]["train"](args)
    valid_dataloader = DATALOADERDICT["mnist"]["valid"](args)

    train_loss=[]
    valid_loss = []

    for epoch in range(args['epochs']):
        print(f"Epoch {epoch+1} of {args['epochs']}]")

        train_epoch_loss = train_epoch(epoch,args,model,device,optimizer,train_dataloader)
        valid_epoch_loss,recon_images = valid_epoch(epoch,args,model,device,optimizer,valid_dataloader)

        #save the reconstructed images from the val loop 
        utils.save_reconstructed_images(epoch+1,args,recon_images)
        #convert the reconstructed images to Pytorch image grid 
        image_grid = make_grid(recon_images.detach().cpu())
        args['grid_images'].append(image_grid)
        print(f"Train Loss : {train_epoch_loss:.4f}")
        print(f"Valid Loss: {valid_epoch_loss:.4f}")

        
    utils.image_to_vid(args['grid_images'],args)

    #save the loss plots to disk 
    utils.save_loss_plot(args,train_loss,valid_loss)
    print("Training Complete")

if __name__ == '__main__':
    main()
