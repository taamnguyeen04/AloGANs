import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from dataset import Affectnet
from model import Generator, Discriminator

def save_checkpoint(filepath, epoch, G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_optimizer):
    checkpoint = {
        'epoch': epoch,
        'G_XtoY_state_dict': G_XtoY.state_dict(),
        'G_YtoX_state_dict': G_YtoX.state_dict(),
        'D_X_state_dict': D_X.state_dict(),
        'D_Y_state_dict': D_Y.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_optimizer, device):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        G_XtoY.load_state_dict(checkpoint['G_XtoY_state_dict'])
        G_YtoX.load_state_dict(checkpoint['G_YtoX_state_dict'])
        D_X.load_state_dict(checkpoint['D_X_state_dict'])
        D_Y.load_state_dict(checkpoint['D_Y_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        return checkpoint['epoch']
    return 0

def train():
    batch_size = 4
    lr = 1e-4
    num_epochs = 100
    image_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    out_path = "out"
    os.makedirs(out_path, exist_ok=True)

    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = Affectnet(root="C:/Users/tam/Documents/Data/Affectnet", is_train=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    val_dataset = Affectnet(root="C:/Users/tam/Documents/Data/Affectnet", is_train=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=True
    )

    G_XtoY = Generator().to(device)
    G_YtoX = Generator().to(device)
    D_X = Discriminator().to(device)
    D_Y = Discriminator().to(device)

    g_optimizer = torch.optim.Adam(list(G_XtoY.parameters()) + list(G_YtoX.parameters()), lr=lr, betas=(0.5, 0.99))
    d_optimizer = torch.optim.Adam(list(D_X.parameters()) + list(D_Y.parameters()), lr=lr, betas=(0.5, 0.99))

    adversarial_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()

    start_epoch = load_checkpoint(os.path.join(out_path, 'cyclegan_checkpoint.pt'), G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_optimizer, device)

    neutral_img, happy_img = next(iter(val_dataloader))
    neutral_img = neutral_img.to(device)
    happy_img = happy_img.to(device)


    for epoch in range(start_epoch, num_epochs):
        for i, (x_real, y_real) in enumerate(train_dataloader):
            x_real, y_real = x_real.to(device), y_real.to(device)

            # Train D
            y_fake = G_XtoY(x_real)
            x_fake = G_YtoX(y_real)

            d_loss_x = adversarial_loss(D_X(x_real), torch.ones_like(D_X(x_real))) + \
                       adversarial_loss(D_X(x_fake.detach()), torch.zeros_like(D_X(x_fake)))
            d_loss_y = adversarial_loss(D_Y(y_real), torch.ones_like(D_Y(y_real))) + \
                       adversarial_loss(D_Y(y_fake.detach()), torch.zeros_like(D_Y(y_fake)))

            d_loss = (d_loss_x + d_loss_y) / 2
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train G
            y_fake = G_XtoY(x_real)
            x_fake = G_YtoX(y_real)

            x_recon = G_YtoX(y_fake)
            y_recon = G_XtoY(x_fake)
            g_loss_x = adversarial_loss(D_X(x_fake), torch.ones_like(D_X(x_fake)))
            g_loss_y = adversarial_loss(D_Y(y_fake), torch.ones_like(D_Y(y_fake)))

            cycle_loss_x = cycle_loss(x_recon, x_real)
            cycle_loss_y = cycle_loss(y_recon, y_real)

            g_loss = g_loss_x + g_loss_y + 10 * (cycle_loss_x + cycle_loss_y)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if i % 50 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

            # Evaluation step
            if (i % 2 == 0) and (i != 0):
                with torch.no_grad():
                    paired_images = []

                    for j in range(neutral_img.shape[0]):
                        x_real_img = neutral_img[j].unsqueeze(0)
                        x_fake_img = G_XtoY(x_real_img)
                        paired_images.append(torch.cat([x_real_img, x_fake_img], dim=3))
                    for j in range(happy_img.shape[0]):
                        y_real_img = happy_img[j].unsqueeze(0)
                        y_fake_img = G_YtoX(y_real_img)
                        paired_images.append(torch.cat([y_real_img, y_fake_img], dim=3))

                    x_concat = torch.cat(paired_images, dim=2)

                    x_concat = ((x_concat + 1) / 2).clamp_(0, 1)

                    sample_path = os.path.join(out_path, f'epoch{epoch}_step{i}-images.jpg')
                    save_image(x_concat, sample_path, nrow=1, padding=0)

                    print(f"Saved evaluation image at {sample_path}")

        save_checkpoint(os.path.join(out_path, 'cyclegan_checkpoint.pt'), epoch, G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_optimizer)

if __name__ == '__main__':
    train()
