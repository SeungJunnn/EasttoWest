from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from models.latent_optimizer import PostSynthesisProcessing
from models.image_to_latent import ImageToLatent
from models.latent_optimizer import VGGProcessing
from utilities.images import load_images
from loss import LossBuilder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt
from functools import partial
import argparse
from pathlib import Path
from PIL import Image
from math import log10, ceil

class Images(Dataset):
    def __init__(self, root_dir, duplicates):
        self.root_path = Path(root_dir)
        self.image_list = list(self.root_path.glob("*.png"))
        self.duplicates = duplicates # Number of times to duplicate the image in the dataset to produce multiple HR images

    def __len__(self):
        return self.duplicates*len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx//self.duplicates]
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        if(self.duplicates == 1):
            return image,img_path.stem
        else:
            return image,img_path.stem+f"_{(idx % self.duplicates)+1}"


class PULSE(nn.Module):
    def __init__(self):
        super(PULSE, self).__init__()

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

    def forward(self,
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

def main():
    parser = argparse.ArgumentParser(description='PULSE')
    parser.add_argument('--input_dir', type=str, default='./anime', help='')
    parser.add_argument('--output_dir', type=str, default='./realization', help='')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to use during optimization')

    parser.add_argument('--loss_str', type=str, default="100*L2+0.05*GEOCROSS", help='Loss function to use')
    parser.add_argument('--eps', type=float, default=2e-3, help='Target for downscaling loss (L2)')
    parser.add_argument('--tile_latent', action='store_true', help='Whether to forcibly tile the same latent 18 times')
    parser.add_argument('--opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
    parser.add_argument('--steps', type=int, default=100, help='Number of optimization steps')
    parser.add_argument('--learning_rate', type=float, default=0.4, help='Learning rate to use during optimization')
    parser.add_argument('--lr_schedule', type=str, default='linear1cycledrop', help='fixed, linear1cycledrop, linear1cycle')
    parser.add_argument('--save_intermediate', action='store_true', help='Whether to store and save intermediate HR and LR images during optimization')
    args = parser.parse_args()

    dataset = Images(args.input_dir, duplicates=3)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    model = PULSE()
    # model = DataParallel(model)

    toPIL = torchvision.transforms.ToPILImage()

    for ref_im, ref_im_name in dataloader:
        if args.save_intermediate:
            padding = ceil(log10(100))
            for i in range(args.batch_size):
                int_path_HR = Path(out_path / ref_im_name[i] / "HR")
                int_path_LR = Path(out_path / ref_im_name[i] / "LR")
                int_path_HR.mkdir(parents=True, exist_ok=True)
                int_path_LR.mkdir(parents=True, exist_ok=True)
            for j,(HR,LR) in enumerate(model(ref_im, args.loss_str, args.eps, args.tile_latent, args.opt_name, args.steps, args.learning_rate, args.lr_schedule, args.save_intermediate)):
                for i in range(args.batch_size):
                    toPIL(HR[i].cpu().detach().clamp(0, 1)).save(
                        int_path_HR / f"{ref_im_name[i]}_{j:0{padding}}.png")
                    toPIL(LR[i].cpu().detach().clamp(0, 1)).save(
                        int_path_LR / f"{ref_im_name[i]}_{j:0{padding}}.png")
        else:
            #out_im = model(ref_im,**kwargs)
            for j,(HR,LR) in enumerate(model(ref_im, args.loss_str, args.eps, args.tile_latent, args.opt_name, args.steps, args.learning_rate, args.lr_schedule, args.save_intermediate)):
                for i in range(args.batch_size):
                    toPIL(HR[i].cpu().detach().clamp(0, 1)).save(
                        out_path / f"{ref_im_name[i]}.png")

if __name__ == "__main__":
    main()
        

