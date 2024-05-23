import os
import random
import argparse
import itertools
import numpy as np
from os import listdir
from os.path import join
from tqdm.auto import tqdm
from omegaconf import OmegaConf 
from torch.utils.data import Dataset

import torch
import torchvision
import torch.nn as nn
from torch.nn import init
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

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
        # Resnet-based Block
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
        # includes two convolution layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )
  
    def forward(self, x):
        out = self.conv(x)
        return out
  

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DeconvBlock, self).__init__()
        # includes two transposed-convolution layers
        self.deconv = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(),
            torch.nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.deconv(x)
        return out


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, ngf, n_res_blocks):
        super(Generator, self).__init__()
        # in_channels: the number of channels of the input
        # ngf: the number of convolution filters of the first layer
        # out_channels: the number of channels of the output
        # n_res_blocks: the number of the resnet-based blocks
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
                                        nn.Tanh()
                                        )
                                        
        
    def forward(self, x):
        
        h = self.first_layer(x)
        h = self.conv_block(h)
        h = self.res_layers(h)
        h = self.deconv_block(h)
        res = self.last_layer(h)

        # residual path
        out = res + x
        return out

def init_weights(net):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        
    print('Initialize network.')
    net.apply(init_func)

class Mean():
    def __init__(self):
        self.numel = 0
        self.mean = 0
    
    def __call__(self, val):
        self.mean = self.mean * (self.numel / (self.numel + 1)) + val / (self.numel + 1)
        self.numel += 1
    
    def result(self):
        return self.mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-data", type=str, default="/data2/youngju/CycleGAN/AAPM_data")
    parser.add_argument("--path-checkpoint", type=str, default="/data2/youngju/CycleGAN/Submission/CT_denoising")
    parser.add_argument("--model-name", type=str, default="supervised_v1")
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--G_ngf", type=int, default=32)
    parser.add_argument("--G_n_res_blocks", type=int, default=6)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
  
    args = parser.parse_args()
    #path_result = join(args.path_checkpoint, args.model_name)
    path_result = args.path_checkpoint
    if not os.path.isdir(path_result):
      os.makedirs(path_result)
    
    flags  = OmegaConf.create({})
    for k, v in vars(args).items():
        print(">>>", k, ":" , v)
        setattr(flags, k, v)
        
    OmegaConf.save(flags, os.path.join(path_result, "supervised_config.yaml"))
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    train_dataloader, test_dataloader = make_dataloader(args.path_data, args.batch_size)
  
    in_channels = train_dataloader.dataset[0][0].shape[0]
    out_channels = train_dataloader.dataset[0][0].shape[0]
    G_ngf = args.G_ngf # the number of convolution filters of the first layer for Generator
    n_res_blocks = args.G_n_res_blocks # the number of the ResNet-based blocks
    
    # Define the Generator
    G_Q2F = Generator(in_channels, out_channels, G_ngf, n_res_blocks).to(args.device)
    
    # Optimizer and Scheduler
    G_optim = torch.optim.Adam(G_Q2F.parameters(), args.lr, betas=(args.beta1, args.beta2))
    step_scheduler = torch.optim.lr_scheduler.StepLR(G_optim, step_size=2, gamma=0.95)
    
    sup_loss = nn.L1Loss()

    # Loss function
    loss_name = ['sup_loss']

    init_weights(G_Q2F)
    trained_epoch = 0
    losses_list = {name: list() for name in loss_name}
    
    for epoch in tqdm(range(trained_epoch, args.num_epoch), desc='Epoch', total=args.num_epoch, initial=trained_epoch):
        losses = {name: Mean() for name in loss_name}
        for x_F, x_Q, _ in tqdm(train_dataloader, desc='Step'):
            x_F = x_F.to(args.device)
            x_Q = x_Q.to(args.device)

            x_QF = G_Q2F(x_Q)
            G_sup_loss = sup_loss(x_QF, x_F)
            
            G_optim.zero_grad()
            G_sup_loss.backward()
            G_optim.step()
            
            losses['sup_loss'](G_sup_loss.item())
            
        step_scheduler.step()    
        for name in loss_name:
            losses_list[name].append(losses[name].result())
            
        torch.save({'epoch': epoch + 1, 'G_Q2F_state_dict': G_Q2F.state_dict(), 'G_optim_state_dict': G_optim.state_dict()}, join(path_result, args.model_name + f'.pth'))
        
        for name in loss_name:
            torch.save(losses_list[name], join(path_result, name + '.npy'))
    
if __name__ == "__main__":
    main()