import torch 
import torch.nn as nn 
import torch.nn.functional as F 

# The LinearVAE() Module

features=16
#define a simple linear VAE

class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE,self).__init__() 
        #Encoder
        self.enc1 = nn.Linear(in_features=784,out_features=512)
        self.enc2 = nn.Linear(in_features=512,out_features=features*2)

        #Decoder
        self.dec1 = nn.Linear(in_features = features, out_features=512)
        self.dec2 = nn.Linear(in_features = 512,out_features=784)

    def reparameterize(self,mu,log_var):
        '''
        :param mu : mean from the encoders' latent space 64 x 16 / [:,0,:]
        :param log_var : log variance from the encoders' latent space 64 x 16 [:,0,:]
        '''
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) 
        sample = mu + (eps * std) # 평균 + 표준편차
        print("sample shape : ",sample.shape) # 64 x 16
        #
        return sample 
    
    def forward(self,x):
        # encoding 
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1,2,features) # 64 bx 2 x 16 

        # get 'mu' and 'log var' 
        mu = x[:,0,:] #the first feature values as mean
        log_var = x[:,1,:] #the other feature values as variance
        '''
        Then we get mu and log_var at lines 37 and 38. 
        These two have the same value as the encoder’s last layer output.
        '''
        print("x shape : {}, input x mu : {} log_Var : {}".format(x.shape,mu.shape,log_var.shape))
        #get the latent vector through reparamenterization
        z = self.reparameterize(mu,log_var) 
        '''
        At line 35, we get the latent vector z 
        through reparameterization trick using mu and log_var.
        '''
        #decoding 
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction,mu,log_var
