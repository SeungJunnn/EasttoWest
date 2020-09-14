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

synthesizer = StyleGANGenerator("stylegan_ffhq").model.synthesis

augments = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def normalized_to_normal_image(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1).float()
    std=torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1).float()
    
    image = image.detach().cpu()
    
    image *= std
    image += mean
    image *= 255
    
    image = image.numpy()[0]
    image = np.transpose(image, (1,2,0))
    return image.astype(np.uint8)

def Embedding(image_path, image_name):
    vgg_processing = VGGProcessing()
    post_processing = PostSynthesisProcessing()
    image_to_latent = ImageToLatent().to(device)
    image_to_latent.load_state_dict(torch.load('./image_to_latent.pt'))
    image_to_latent.eval()
    post_process = lambda image: post_processing(image).detach().cpu().numpy().astype(np.uint8)[0]

    reference_image = torch.from_numpy(load_images([os.path.join(image_path,image_name)])).to(device)
    reference_image = vgg_processing(reference_image).detach()

    pred_dlatents = image_to_latent(reference_image)

    pred_images = synthesizer(pred_dlatents)
    pred_images = post_process(pred_images)
    pred_images = np.transpose(pred_images, (1,2,0))

    plt.imsave('./outputs/'+image_name, pred_images)
    print('Embedding for Image {} is finished.'.format(image_name))

def main():
    parser = argparse.ArgumentParser(description='East to West')
    parser.add_argument('--aligned_path', type=str, default='./inputs', help='path for aligned images')
    parser.add_argument('--output_path', type=str, default='./outputs', help='path for output images')
    args = parser.parse_args()

    if not (os.path.exists(args.aligned_path)):
        print('Check the path for the aligned images')
        raise FileNotFoundError
    if not (os.path.exists(args.output_path)):
        os.makedirs(args.output_path)

    for image in os.listdir(args.aligned_path):
        Embedding(args.aligned_path, image)

if __name__ == "__main__":
    main()
