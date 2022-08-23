import torch 
import torch.nn as nn 
import torch.nn.functional as F 

def get_bce_loss(reconstruction,data):
    bce_loss = nn.BCELoss(reduction='sum')
    loss = bce_loss(reconstruction,data)
    return loss 

def get_kld_loss(mu,log_var):
    KLD = -0.5 * torch.sum(1+log_var - mu.pow(2) - log_var.exp())
    return KLD 