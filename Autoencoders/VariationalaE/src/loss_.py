import torch 
import torch.nn as nn 

def get_bce_loss(reconstruction,data):
    bce_loss = nn.BCELoss(reduction='sum')
    loss = bce_loss(reconstruction,data)
    return loss 


def get_KLD_loss(mu,log_var):
    '''
    This function will add the reconstruction loss(BCE Loss) and
    KL_divergence.
    KL_Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss : reconstruction loss 
    :param mu : the mean from the latent vector 
    :param logvar : log variance from the latent vector
    '''

    KLD = -0.5 * torch.sum(1+log_var - mu.pow(2) - log_var.exp())

    return KLD 
