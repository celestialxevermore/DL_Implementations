import torch 
import torch.nn as nn 
import torch.nn.functional as F

def get_mse_loss(decoded,input_image):
    mse_loss = nn.MSELoss()(decoded,input_image)
    return mse_loss 


def kl_divergence(model,device,rho,rho_hat):
    #print("IN KL DIVERGENCE")
    rho_hat = torch.mean(torch.sigmoid(rho_hat),1)
    rho = torch.tensor([rho]*len(rho_hat)).to(device)
    #print(f"rho : {rho.shape} rho_hat : {rho_hat.shape} \n")

    return torch.sum(rho * torch.log(rho/rho_hat) + (1-rho)*torch.log((1-rho)/(1-rho_hat)))


def sparse_loss(model,device,rho,images):
    model_children = list(model.children())
    values=images 
    #print("values shape :",values.shape)
    loss=0
    for ind in range(len(model_children)):
        #print(model_children[ind])
        values = model_children[ind](values)
        loss += kl_divergence(model,device,rho,values)
    return loss 
