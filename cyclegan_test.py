import os
import random
import argparse
import itertools
import numpy as np
from os import listdir
from os.path import join
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# CT dataset
class CT_Dataset(Dataset):
    def __init__(self, path, transform):
      # Path of 'full_dose' and 'quarter_dose' folders
      self.path_full = join(path, 'full_dose')
      self.path_quarter = join(path, 'quarter_dose')
      self.transform = transform

      # File list of full dose data
      self.file_full = list()
      for file_name in sorted(listdir(self.path_full)):
        self.file_full.append(file_name)
    
      # File list of quarter dose data
      self.file_quarter = list()
      for file_name in sorted(listdir(self.path_quarter)):
        self.file_quarter.append(file_name)
  
    def __len__(self):
      return min(len(self.file_full), len(self.file_quarter))
  
    def __getitem__(self, idx):
      # Load full dose/quarter dose data
      x_F = np.load(join(self.path_full, self.file_full[idx]))
      x_Q = np.load(join(self.path_quarter, self.file_quarter[idx]))
  
      # Convert to HU scale
      x_F = (x_F - 0.0192) / 0.0192 * 1000
      x_Q = (x_Q - 0.0192) / 0.0192 * 1000

      # Normalize images
      x_F[x_F < -1000] = -1000
      x_Q[x_Q < -1000] = -1000

      x_F = x_F / 4000
      x_Q = x_Q / 4000

      # Apply transform
      x_F = self.transform(x_F)
      x_Q = self.transform(x_Q)

      file_name = self.file_quarter[idx]

      return x_F, x_Q, file_name

# Transform for the random crop
class RandomCrop(object):
    def __init__(self, patch_size):
      self.patch_size = patch_size
  
    def __call__(self, img):
      # Randomly crop the image into a patch with the size [self.patch_size, self.patch_size]
      w, h = img.size(-1), img.size(-2)
      i = random.randint(0, h - self.patch_size)
      j = random.randint(0, w - self.patch_size)

      return img[:, i:i + self.patch_size, j:j + self.patch_size]


# Make dataloader for training/test
def make_dataloader(path, batch_size):
    # Path of 'train' and 'test' folders
    path_train = join(path, 'train')
    path_test = join(path, 'test')

    # Transform for training data: convert to tensor, random horizontal/verical flip, random crop
    # You can change transform if you want.
    train_transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.RandomHorizontalFlip(p=0.5),
      torchvision.transforms.RandomVerticalFlip(p=0.5),
      RandomCrop(128)
    ])

    # Transform for test data: convert to tensor
    test_transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor()
    ])

    # Generate CT dataset for training/test
    train_dataset = CT_Dataset(path_train, train_transform)
    test_dataset = CT_Dataset(path_test, test_transform)
  
    # Generate dataloader for training/test
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    return train_dataloader, test_dataloader

    
class ResnetBlock(torch.nn.Module):
    def __init__(self, n_channels):
        super(ResnetBlock, self).__init__()
        
        self.res_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_channels, n_channels, kernel_size=3),
            nn.InstanceNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_channels, n_channels, kernel_size=3),
            nn.InstanceNorm2d(n_channels),
        )

    def forward(self, x):
        out = x + self.res_block(x)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            #nn.BatchNorm2d(mid_channels, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            #nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU()
        )
  
    def forward(self, x):
        out = self.conv(x)
        return out
  

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            #nn.BatchNorm2d(mid_channels, affine=True, track_running_stats=True),
            nn.ReLU(),
            torch.nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            #nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.deconv(x)
        return out


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, ngf, n_res_blocks):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3),
                                         nn.Conv2d(in_channels, ngf, 7),
                                         nn.InstanceNorm2d(ngf),
                                         nn.ReLU(inplace=True))
        self.conv_block = ConvBlock(self.ngf, self.ngf*2, self.ngf*4)
        self.res_layers = nn.Sequential(*[ResnetBlock(self.ngf*4) for i in range(n_res_blocks)])
        self.deconv_block = DeconvBlock(self.ngf*4, self.ngf*2, self.ngf)
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(self.ngf, out_channels, 7),
                                        nn.Tanh())
                                        
        
    def forward(self, x):
        
        h = self.first_layer(x)
        h = self.conv_block(h)
        h = self.res_layers(h)
        h = self.deconv_block(h)
        res = self.last_layer(h)

        out = res + x
        return out
    


# Functions for caculating PSNR, SSIM
def psnr(A, ref):
    ref[ref < -1000] = -1000
    A[A < -1000] = -1000
    val_min = -1000
    val_max = np.amax(ref)
    ref = (ref - val_min) / (val_max - val_min)
    A = (A - val_min) / (val_max - val_min)
    out = peak_signal_noise_ratio(ref, A)
    return out

def ssim(A, ref):
    ref[ref < -1000] = -1000
    A[A < -1000] = -1000
    val_min = -1000
    val_max = np.amax(ref)
    ref = (ref - val_min) / (val_max - val_min)
    A = (A - val_min) / (val_max - val_min)
    out = structural_similarity(ref, A, data_range=2)
    return out



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-data", type=str, default="/data2/youngju/CycleGAN/AAPM_data")
    parser.add_argument("--path-checkpoint", type=str, default="/data2/youngju/CycleGAN/CT_denoising")
    parser.add_argument("--model-name", type=str, default="cyclegan_final")
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4) # 2e-4,5
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--G_ngf", type=int, default=32)
    parser.add_argument("--G_n_res_blocks", type=int, default=6)
    
    
    args = parser.parse_args()
    path_result = join(args.path_checkpoint, args.model_name)
    if not os.path.isdir(path_result):
        os.makedirs(path_result)
      
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    # Plot loss graph (adversarial loss)
    loss_path = f'{path_result}/loss_fig'
    if not os.path.isdir(loss_path):
        os.makedirs(loss_path)
        
    for name in ['G_adv_loss_F', 'G_adv_loss_Q', 'D_adv_loss_F', 'D_adv_loss_Q']:
        loss_arr = torch.load(join(path_result, name + '.npy'))
        num_epoch = len(loss_arr)
        break
    
    x_axis = np.arange(1, num_epoch + 1)
    plt.figure(1)
    for name in ['G_adv_loss_F', 'G_adv_loss_Q', 'D_adv_loss_F', 'D_adv_loss_Q']:
        loss_arr = torch.load(join(path_result, name + '.npy'))
        plt.plot(x_axis, loss_arr, label=name)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig(join(loss_path, 'cyclegan_adv.png'))
    plt.show()

    # Plot loss graph (cycle consistency loss, identity loss)
    plt.figure(2)
    for name in ['G_cycle_loss_F', 'G_cycle_loss_Q', 'G_iden_loss_F', 'G_iden_loss_Q']:
        loss_arr = torch.load(join(path_result, name + '.npy'))
        plt.plot(x_axis, loss_arr, label=name)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig(join(loss_path, 'cyclegan_cyc_idn.png'))
    plt.show()
    
    train_dataloader, test_dataloader = make_dataloader(args.path_data, args.batch_size)
    G_ngf = args.G_ngf
    n_res_blocks = args.G_n_res_blocks
    in_channels = test_dataloader.dataset[0][0].shape[0]
    out_channels = test_dataloader.dataset[0][0].shape[0]
    G_F2Q = Generator(in_channels, out_channels, G_ngf, n_res_blocks).to(args.device)
    G_Q2F = Generator(in_channels, out_channels, G_ngf, n_res_blocks).to(args.device)
    
    epoch = 70
    # Load the last checkpoint
    checkpoint = torch.load(join(path_result, args.model_name + f'_e-{epoch}.pth'))
    G_Q2F.load_state_dict(checkpoint['G_Q2F_state_dict'])
    G_Q2F.eval()

    save_path = f'{path_result}/test_outputs'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Test and save
    with torch.no_grad():
        for _, x_Q, file_name in tqdm(test_dataloader):
            x_Q = x_Q.to(args.device)
            x_QF = G_Q2F(x_Q)[0].detach().cpu().numpy()
            x_QF = x_QF * 4000

            np.save(join(save_path, file_name[0]), x_QF)

    QF_psnr_list = []
    QF_ssim_list = []
    OF_psnr_list = []
    OF_ssim_list = []

    fig_path = f'{path_result}/loss_fig'
    if not os.path.isdir(loss_path):
        os.makedirs(loss_path)
            
    for num in tqdm(range(len(test_dataloader.dataset))):
        num = num+1
        path_quarter = join(args.path_data, f'test/quarter_dose/{num}.npy')
        path_full = join(args.path_data, f'test/full_dose/{num}.npy')
        path_output = join(save_path, f'{num}.npy')

        quarter = np.load(path_quarter)
        full = np.load(path_full)
        output = np.load(path_output)
        output = output.reshape(quarter.shape[0], quarter.shape[1])

        quarter = (quarter - 0.0192) / 0.0192 * 1000
        full = (full - 0.0192) / 0.0192 * 1000

        QF_psnr_list.append(psnr(quarter, full))
        QF_ssim_list.append(ssim(quarter, full))
        OF_psnr_list.append(psnr(output, full))
        OF_ssim_list.append(ssim(output, full))
        
        quarter = np.clip(quarter, -1000, 1000)
        full = np.clip(full, -1000, 1000)
        output = np.clip(output, -1000, 1000)
    
    print(f'🔥 Mean PSNR between input and ground truth: \n {np.mean(QF_psnr_list)}')
    print(f'🔥 Mean SSIM between input and ground truth: \n {np.mean(QF_ssim_list)}')
    print(f'🔥 Mean PSNR between output and ground truth: \n {np.mean(OF_psnr_list)}')
    print(f'🔥 Mean SSIM between output and ground truth: \n {np.mean(OF_ssim_list)}')