from pytorch_lightning.accelerators import accelerator
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

import argparse
import os, sys
import multiprocessing

# load image files from specific folder
# it return unmask image & mask image pair
class MaskDataset(Dataset):
    # unmask_img_folder & mask_img_folder has same image named files
    def __init__(
        self, unmask_img_folder, mask_img_folder, mask_postfix='_cloth', img_size=128, transform=None
    ):
        self.unmask_img_folder = unmask_img_folder
        self.mask_img_folder = mask_img_folder
        self.mask_postfix = '_cloth'
        self.img_size = img_size
        self.file_names = os.listdir(self.mask_img_folder)
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                ]
            )

        self.loss_func = nn.BCELoss()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        unmask_img = Image.open(self.unmask_img_folder + os.sep + self.file_names[idx].replace(self.mask_postfix, ''))
        mask_img = Image.open(self.mask_img_folder + os.sep + self.file_names[idx])

        unmask_img_tensor = self.transform(unmask_img)
        mask_img_tensor = self.transform(mask_img)

        return unmask_img_tensor, mask_img_tensor


class UnmaskingModel(pl.LightningModule):
    def __init__(self, img_size=128, lr=1e-4):
        super(UnmaskingModel, self).__init__()
        self.img_size = img_size
        self.lr = lr
        self.loss_func = nn.BCELoss()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        self.save_hyperparameters()

    def add_gaussian_noise(self, img_tensor, mean=0.0, std=1.0):
        return img_tensor + torch.randn(img_tensor.size()) * std + mean

    def build_generator(self):
        # Generator
        # input: [b, 3, 128, 128]
        # output: [b, 3, 128, 128]
        model = nn.Sequential(
            nn.ConvTranspose2d(3, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        return model

    def build_discriminator(self):
        # Discriminator
        # input: [b, 3, 128, 128]
        # output: [b, 1]
        model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        return model

    def forward(self, unmask_img, mask_img):
        # discriminator model forwarding
        discriminator_p_real = self.discriminator(unmask_img)
        discriminator_p_fake = self.discriminator(self.generator(unmask_img))

        # generator model forwarding
        # fake_img2 = self.generator(self.add_gaussian_noise(mask_img))
        generator_p_fake = self.discriminator(self.generator(mask_img))

        return discriminator_p_real, discriminator_p_fake, generator_p_fake

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def training_step(self, batch, batch_idx):
        self.generator.train()
        self.discriminator.train()

        unmask_img, mask_img = batch
        discriminator_p_real, discriminator_p_fake, generator_p_fake = self.forward(unmask_img, mask_img)

        loss_d = self.loss_func(
            discriminator_p_real, torch.ones_like(discriminator_p_real)
        ) + self.loss_func(discriminator_p_fake, torch.zeros_like(discriminator_p_fake))
        loss_g = self.loss_func(generator_p_fake, torch.ones_like(generator_p_fake))

        loss = loss_d + loss_g

        self.log("loss", loss)

    def validation_step(self, batch, batch_idx):
        self.generator.eval()
        self.discriminator.eval()

        with torch.no_grad():
            unmask_img, mask_img = batch
            discriminator_p_real, discriminator_p_fake, generator_p_fake = self.forward(unmask_img, mask_img)

            loss_d = self.loss_func(
                discriminator_p_real, torch.ones_like(discriminator_p_real)
            ) + self.loss_func(
                discriminator_p_fake, torch.zeros_like(discriminator_p_fake)
            )
            loss_g = self.loss_func(generator_p_fake, torch.ones_like(generator_p_fake))

            loss = loss_d + loss_g

            self.log("val_loss", loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()

    # model preparation
    model = UnmaskingModel(img_size=args.img_size, lr=args.lr)

    # data preparation
    dataset = MaskDataset(
        unmask_img_folder="../data/img_align_celeba_png",
        mask_img_folder="../data/img_align_celeba_png_masked",
    )
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [
            int(args.train_ratio * len(dataset)),
            len(dataset) - int(args.train_ratio * len(dataset)),
        ],
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=multiprocessing.cpu_count()
    )

    # training
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        progress_bar_refresh_rate=0,
        max_epochs=args.epochs,
        accelerator="ddp",
    )
    trainer.fit(model, train_loader, val_loader)
