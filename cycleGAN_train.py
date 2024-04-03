import random
import numpy as np
from os import listdir
from os.path import join
from torch.utils.data import Dataset

import torch
import torchvision
import torch.nn as nn
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
    random.seed(0)
    random.shuffle(self.file_full)
    
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



# class ResnetBlock(torch.nn.Module):
#     def __init__(self, in_channels, mid_channels, out_channels):
#         super(ResnetBlock, self).__init__()
        
#         self.res_block = nn.Sequential(
#             #nn.ReflectionPad2d(1),
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3),
#             nn.InstanceNorm2d(mid_channels),
#             nn.ReLU(),
#             #nn.ReflectionPad2d(1),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3),
#             nn.InstanceNorm2d(out_channels),
#         )

#     def forward(self, x):
#         out = x + self.res_block(x)
#         return out
    
class ResnetBlock(torch.nn.Module):
    def __init__(self, n_channels):
        super(ResnetBlock, self).__init__()
        
        self.res_block = nn.Sequential(
            #nn.ReflectionPad2d(1),
            nn.Conv2d(n_channels, n_channels, kernel_size=3),
            nn.InstanceNorm2d(n_channels),
            nn.ReLU(inplace=True),
            #nn.ReflectionPad2d(1),
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
            nn.BatchNorm2d(mid_channels, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
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
            nn.BatchNorm2d(mid_channels, affine=True, track_running_stats=True),
            nn.ReLU(),
            torch.nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.deconv(x)
        return out


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, ngf, n_res_blocks):
        super(Generator, self).__init__()
        self.mid_channels = ngf
        self.conv1 = ConvBlock(in_channels, self.mid_channels, self.mid_channels*2)
        self.conv2 = ConvBlock(self.mid_channels*2, self.mid_channels*4, self.mid_channels*8)
        self.res_layers = nn.Sequential([ResnetBlock(self.mid_channels*8) for i in range(n_res_blocks)])
        self.deconv1 = DeconvBlock(self.mid_channels*8, self.mid_channels*4, self.mid_channels*2)
        self.deconv2 = DeconvBlock(self.mid_channels*2, self.mid_channels, out_channels)
        
    def forward(self, x):
        
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.res_layers(h)
        h = self.deconv1(h)
        res = self.deconv2(h)

        out = res + x
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels, ndf):
        super(Discriminator, self).__init__()
        # in_channels: the number of channels of the input
        # ndf: the number of convolution filters of the first layer
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, bias=False),
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, bias=False),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=1, bias=False),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, ndf*1, kernel_size=4, stride=1, bias=False)          
        )

  
    def forward(self, x):
        out = self.discriminator(x)
        return out




def main():
    




























if __name__ == "__main__":
    main()