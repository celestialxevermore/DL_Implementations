import imageio 
import numpy as np 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 

from torchvision.utils import save_image 

to_pil_image = transforms.ToPILImage()




def image_to_vid(images,args):
    imgs = [np.array(to_pil_image(img)) for img in images]
    output_dir = args['output_dir']
    imageio.mimsave(output_dir+"/generated_images.gif",imgs)

def save_reconstructed_images(epoch,args,recon_images):
    save_image(recon_images.cpu(),f"{args['output_dir']}/output{epoch}.jpg")

def save_loss_plot(args,train_loss,valid_loss):
    #loss plots 
    output_dir = args['output_dir']
    plt.figure(figsize=(10,7))
    plt.plot(train_loss,color='orange',label='train_loss')
    plt.plot(valid_loss,color='red',label='validation_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{output_dir}/loss.jpg")
    plt.show()
