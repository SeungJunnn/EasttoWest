from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from models.latent_optimizer import PostSynthesisProcessing
from models.image_to_latent import ImageToLatent
from models.latent_optimizer import VGGProcessing
from utilities.images import load_images

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fineblending(dlatents_1, dlatents_2):
    latents = torch.empty(size=dlatents_1.shape).to(device)
    for i in range(18):
        latents[:,i] = (dlatents_1[:,i] * i + dlatents_2[:,i] * (17-i))/17
    return latents


def latentmix(path1, path2, device):
    dlatents=[]
    dlatents_1 = torch.from_numpy(np.load(path1)).to(device)
    dlatents_2 = torch.from_numpy(np.load(path2)).to(device)
    
    dlatents.append(dlatents_2)
    for i in range(18): #num of stylegan AdaIN layers
        dlatent = torch.empty(size=dlatents_1.shape).to(device)
        dlatent[:,:i]=dlatents_1[:,:i]
        dlatent[:,i:]=dlatents_2[:,i:]
        dlatents.append(dlatent)
    dlatents.append(dlatents_1)

    mean = (dlatents_1+dlatents_2) / 2
    dlatents.append(mean)
    dlatents.append(fineblending(dlatents_1, dlatents_2))
    dlatents.append(fineblending(dlatents_2, dlatents_1))

    return dlatents

def postprocess(pred_images):
    post_processing = PostSynthesisProcessing()
    post_process = lambda image: post_processing(image).detach().cpu().numpy().astype(np.uint8)[0]
    return post_process(pred_images)

def save_images(latent_list, generator):
    for i,dlatent in enumerate(latent_list):
        pred_images = generator(dlatent)
        pred_images = postprocess(pred_images)
        pred_images = np.transpose(pred_images, (1,2,0))
        if i < len(latent_list)-3:
            plt.imsave('mix_'+str(i)+'.png', pred_images)
        elif i == len(latent_list)-3:
            plt.imsave('mix.png', pred_images)
        else:
            plt.imsave('mix_fine_{}.png'.format(str(i)), pred_images)
        print('Embedding for Image number {} is finished.'.format(str(i)))

def main():
    parser = argparse.ArgumentParser(description='East to West')
    parser.add_argument('--latent1_path', type=str, default='sj.npy', help='')
    parser.add_argument('--latent2_path', type=str, default='', help='')
    args = parser.parse_args()

    synthesizer = StyleGANGenerator("stylegan_ffhq").model.synthesis

    dlatents=latentmix(args.latent1_path, args.latent2_path,device)
    save_images(dlatents, synthesizer)

if __name__ == "__main__":
    main()






