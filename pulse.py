from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from models.latent_optimizer import PostSynthesisProcessing
from models.image_to_latent import ImageToLatent
from models.latent_optimizer import VGGProcessing
from utilities.images import load_images
from loss import LossBuilder

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from functools import partial
import argparse

class PULSE(nn.Module):
    def __init__(self):
        self.mapper = StyleGANGenerator("stylegan_ffhq").model.mapping
        self.truncation = StyleGANGenerator("stylegan_ffhq").model.mapping
        self.synthesizer = StyleGANGenerator("stylegan_ffhq").model.synthesis

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)

    def gaussian_fit(self):
        if os.path.isfile('gaussian_fit.pt'):
            gaussian_fit = torch.load('gaussian_fit.pt')
        else:
            latent = torch.randn((1000000,512),dtype=torch.float32, device="cuda")
            latent_out = torch.nn.LeakyReLU(5)(self.mapper(latent))
            gaussian_fit = {"mean": latent_out.mean(0), "std": latent_out.std(0)}
            torch.save(gaussian_fit,"gaussian_fit.pt")
        return gaussian_fit

    def foward(self,
               ref_im,
               loss_str,
               eps,
               tile_latent,
               opt_name,
               steps,
               learning_rate,
               lr_schedule,
               save_intermediate):

        gaussian_fit = self.gaussian_fit()

        if(tile_latent):
            latent = torch.randn(
                (batch_size, 1, 512), dtype=torch.float, requires_grad=True, device='cuda')
        else:
            latent = torch.randn(
                (batch_size, 18, 512), dtype=torch.float, requires_grad=True, device='cuda')

        var_list = [latent]

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        opt_func = opt_dict[opt_name]
        opt = SphericalOptimizer(opt_func, var_list, lr=learning_rate)

        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9*(1-np.abs(x/steps-1/2)*2)+1)/10,
            'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10),
        }
        schedule_func = schedule_dict[lr_schedule]
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt.opt, schedule_func)

        loss_builder = LossBuilder(ref_im, loss_str, eps).cuda()

        min_loss = np.inf
        min_l2 = np.inf
        best_summary = ""
        start_t = time.time()
        gen_im = None

        print("Optimizing")

        for j in range(steps):
            opt.opt.zero_grad()

            # Duplicate latent in case tile_latent = True
            if (tile_latent):
                latent_in = latent.expand(-1, 18, -1)
            else:
                latent_in = latent

            # Apply learned linear mapping to match latent distribution to that of the mapping network
            latent_in = self.lrelu(latent_in*self.gaussian_fit["std"] + self.gaussian_fit["mean"])

            # Normalize image to [0,1] instead of [-1,1]
            gen_im = (self.synthesizer(latent_in)+1)/2

            # Calculate Losses
            loss, loss_dict = loss_builder(latent_in, gen_im)
            loss_dict['TOTAL'] = loss

            # Save best summary for log
            if(loss < min_loss):
                min_loss = loss
                best_summary = f'BEST ({j+1}) | '+' | '.join(
                [f'{x}: {y:.4f}' for x, y in loss_dict.items()])
                best_im = gen_im.clone()

            loss_l2 = loss_dict['L2']

            if(loss_l2 < min_l2):
                min_l2 = loss_l2

            # Save intermediate HR and LR images
            if(save_intermediate):
                yield (best_im.cpu().detach().clamp(0, 1),loss_builder.D(best_im).cpu().detach().clamp(0, 1))

            loss.backward()
            opt.step()
            scheduler.step()

        total_t = time.time()-start_t
        current_info = f' | time: {total_t:.1f} | it/s: {(j+1)/total_t:.2f} | batchsize: {batch_size}'
        print(best_summary+current_info)
        if(min_l2 <= eps):
            yield (gen_im.clone().cpu().detach().clamp(0, 1),loss_builder.D(best_im).cpu().detach().clamp(0, 1))
        else:
            print("Could not find a face that downscales correctly within epsilon")

        

