import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, in_channels=2, out_channels=3):
        super(Generator, self).__init__()
        
        def down_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def up_block(in_feat, out_feat, dropout=0.0):
            layers = [
                nn.ConvTranspose2d(in_feat, out_feat, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_feat, 0.8),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers.append(nn.Dropout(dropout, inplace=True))
            return layers

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            *down_block(64, 128),
            *down_block(128, 256),
            *down_block(256, 512),
            *down_block(512, 512),
            *down_block(512, 512),
            *down_block(512, 512),
            
            *up_block(512, 512, dropout=0.5),
            *up_block(1024, 512, dropout=0.5),
            *up_block(1024, 512, dropout=0.5),
            *up_block(1024, 512),
            *up_block(1024, 256),
            *up_block(512, 128),
            *up_block(256, 64),
        )
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        self.height_upsample = nn.Upsample(size=(64, 48), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.model(x)
        x = self.final(x)
        x = self.height_upsample(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

def get_gan_models(in_channels=2, out_channels=3):
    generator = Generator(in_channels=in_channels, out_channels=out_channels)
    discriminator = Discriminator(in_channels=out_channels)
    return generator, discriminator