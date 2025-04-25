import torch
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    def __init__(self, n_classes=4, latent_dim=100, img_size=64, channels=3):
        super(Generator, self).__init__()

        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels

        self.label_emb = nn.Embedding(self.n_classes, self.latent_dim)
        self.init_size = self.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))

        self.res_block = ResidualBlock(128, 128)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        out = self.res_block(out)
        img = self.conv_blocks(out)
        return img



class Discriminator(nn.Module):
    def __init__(self, n_classes=4, img_size=64, channels=3):
        super(Discriminator, self).__init__()

        self.n_classes = n_classes
        self.img_size = img_size
        self.channels = channels

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        self.res_block = ResidualBlock(128, 128)

        # Calculate the correct flattened dimension
        ds_size = self.img_size // 2 ** 4
        self.flatten_dim = 128 * ds_size * ds_size

        # Use the computed flatten_dim for linear layers
        self.adv_layer = nn.Sequential(nn.Linear(self.flatten_dim, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(self.flatten_dim, n_classes), nn.Softmax(dim=1))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = self.res_block(out)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label


class Discriminator_Dropout(nn.Module):
    def __init__(self, n_classes=4, img_size=64, channels=3):
        super(Discriminator_Dropout, self).__init__()

        self.n_classes = n_classes
        self.img_size = img_size
        self.channels = channels

        def discriminator_block(in_filters, out_filters, bn=True, dropout=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            if dropout:  # Thêm dropout vào mỗi block
                block.append(nn.Dropout2d(0.25))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.channels, 16, bn=False, dropout=True),
            *discriminator_block(16, 32, dropout=True),
            *discriminator_block(32, 64, dropout=True),
            *discriminator_block(64, 128, dropout=True),
        )

        self.res_block = ResidualBlock(128, 128)

        # Tính toán kích thước đầu vào cho fully connected layer
        ds_size = self.img_size // 2 ** 4
        self.flatten_dim = 128 * ds_size * ds_size

        # Thêm dropout trước fully connected layers
        self.dropout = nn.Dropout(0.5)  # Dropout với tỷ lệ 50%

        self.adv_layer = nn.Sequential(
            nn.Linear(self.flatten_dim, 1),
            nn.Sigmoid()
        )
        self.aux_layer = nn.Sequential(
            nn.Linear(self.flatten_dim, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        out = self.conv_blocks(img)
        out = self.res_block(out)
        out = out.view(out.shape[0], -1)
        out = self.dropout(out)  # Áp dụng dropout
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label

class Discriminator_PatchGan(nn.Module):
    def __init__(self, n_classes=4, img_size=64, channels=3):
        super(Discriminator_PatchGan, self).__init__()

        self.n_classes = n_classes
        self.img_size = img_size
        self.channels = channels

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1),  # Kernel=4, Stride=2, Padding=1
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.channels, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )

        # Tùy chọn: Residual block giữ lại nếu bạn muốn
        self.res_block = ResidualBlock(512, 512)

        # PatchGAN: Output kích thước nhỏ như (B, 1, 4, 4)
        self.adv_layer = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

        # Classification (auxiliary classifier head)
        # Global pooling + FC cho nhãn
        self.label_pool = nn.AdaptiveAvgPool2d(1)
        self.aux_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        out = self.conv_blocks(img)
        out = self.res_block(out)

        validity = torch.sigmoid(self.adv_layer(out))  # PatchGAN output, e.g., (B, 1, 4, 4)
        label_feat = self.label_pool(out)              # (B, 512, 1, 1)
        label = self.aux_layer(label_feat)             # (B, n_classes)

        return validity, label