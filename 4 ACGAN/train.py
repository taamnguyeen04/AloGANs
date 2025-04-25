import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score
from model import Generator, Discriminator, Discriminator_PatchGan
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
import torchvision


def save_checkpoint(filepath, epoch, G, D, g_optimizer, d_optimizer):
    checkpoint = {
        'epoch': epoch,
        'Generator': G.state_dict(),
        'Discriminator': D.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, G, D, g_optimizer, d_optimizer, device):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        G.load_state_dict(checkpoint['Generator'])
        D.load_state_dict(checkpoint['Discriminator'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        return checkpoint['epoch']
    return 0

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = torch.ones(d_interpolates.size()).to(device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train():
    batch_size = 8
    lr = 1e-5
    num_epochs = 50
    image_size = 64
    lambda_gp = 10
    lambda_cls = 2
    num_classes = 4
    # class_names = ["c1", "c2", "c3"]
    best_f1 = 0
    best_accuracy = 0
    start_epoch = 0
    out_path = "out"
    writer = SummaryWriter(log_dir=os.path.join(out_path, 'logs'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(out_path, exist_ok=True)

    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.4059, 0.3560, 0.3149], std=[0.2782, 0.2522, 0.2329]),
    ])

    full_dataset = ImageFolder(root=r"C:\Users\tam\Downloads\ck-aio-hutech\train2", transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    # train_dataset = ImageFolder(root=r"C:\Users\tam\Documents\Data\aio-hutech\train", transform=transform)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=False)
    #
    # val_dataset = ImageFolder(root=r"C:\Users\tam\Documents\Data\aio-hutech\test", transform=transform)
    # test_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

    G = Generator(n_classes=num_classes, img_size=64, channels=3).to(device)
    D = Discriminator(n_classes=num_classes, img_size=64, channels=3).to(device)

    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.99))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.99))

    # classification_loss = nn.CrossEntropyLoss() # trường hợp không có smoothing
    classification_loss = nn.CrossEntropyLoss()

    # start_epoch = load_checkpoint(os.path.join(out_path, 'last.pt'), G, D, g_optimizer, d_optimizer, device)

    # best_D_path = os.path.join(out_path, 'best_D.pt')
    # if os.path.exists(best_D_path):
    #     print("Loading best Discriminator from:", best_D_path)
    #     D.load_state_dict(torch.load(best_D_path, map_location=device))

    for epoch in range(start_epoch, num_epochs):
        G.train()
        D.train()

        for i, (real_images, real_labels) in enumerate(train_dataloader):
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)

            # Train D
            z = torch.randn(batch_size, 100).to(device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            gen_images = G(z, gen_labels)

            real_validity, real_cls_logits = D(real_images)
            fake_validity, fake_cls_logits = D(gen_images.detach())

            d_adv_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            d_cls_loss_real = classification_loss(real_cls_logits, real_labels)
            d_cls_loss_fake = classification_loss(fake_cls_logits, gen_labels)
            gp = compute_gradient_penalty(D, real_images.data, gen_images.data, device)

            d_loss = d_adv_loss + lambda_gp * gp + lambda_cls * (d_cls_loss_real + d_cls_loss_fake)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train G
            if i % 5 == 0:
                z = torch.randn(batch_size, 100).to(device)
                gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
                gen_images = G(z, gen_labels)
                fake_validity, fake_cls_logits = D(gen_images)

                g_adv_loss = -torch.mean(fake_validity)
                g_cls_loss = classification_loss(fake_cls_logits, gen_labels)
                g_loss = g_adv_loss + lambda_cls * g_cls_loss

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            if i % 10 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
            # if i % 20 == 0:  # Ghi log mỗi 20 batch thay vì mỗi batch
            #     global_step = epoch * len(train_dataloader) + i
            #     writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
            #     writer.add_scalar("Loss/Generator", g_loss.item(), global_step)
            #     writer.add_scalar("Loss/D_adv", d_adv_loss.item(), global_step)
            #     writer.add_scalar("Loss/D_cls_real", d_cls_loss_real.item(), global_step)
            #     writer.add_scalar("Loss/D_cls_fake", d_cls_loss_fake.item(), global_step)
            #     writer.add_scalar("Loss/GP", gp.item(), global_step)

            # if i % 50 == 0:
            #     G.eval()
            #     with torch.no_grad():
            #         z = torch.randn(8, 100).to(device)
            #         gen_labels = torch.randint(0, num_classes, (8,), device=device)
            #         gen_images = G(z, gen_labels)  # (B, C, H, W)
            #
            #         # scale từ [-1, 1] về [0, 1] nếu bạn dùng Tanh ở output G
            #         gen_images = (gen_images + 1) / 2
            #
            #         grid = torchvision.utils.make_grid(gen_images, nrow=4)
            #         save_path = os.path.join(out_path, f"gen_epoch{epoch}_iter{i}.png")
            #         torchvision.utils.save_image(grid, save_path)
            #     G.train()

        D.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for img, label in test_dataloader:
                img = img.to(device)
                label = label.to(device)
                _, cls_out = D(img)
                preds = cls_out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        # f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch [{epoch}/{num_epochs}] - Accu Score on D: {accuracy:.4f} - best: {best_accuracy: .4f}")
        # writer.add_scalar("Metrics/F1_score_D", f1, epoch)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print("Best D")
            torch.save(D.state_dict(), os.path.join(out_path, "best_D.pt"))

        save_checkpoint(os.path.join(out_path, 'last.pt'), epoch, G, D, g_optimizer, d_optimizer)
    writer.close()
if __name__ == '__main__':
    train()
