from pytorch_lightning import callbacks
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import numpy as np

import argparse
import os, sys
import multiprocessing

# model architecture definition
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            #layers += [nn.ReLU()]
            layers += [nn.LeakyReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=3, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=3)

        #self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        x = self.dec1_1(dec1_2)
        #x = self.fc(x)

        return x

# dataset definition
class MaskDataset(Dataset):
    def __init__(
        self, unmask_img_folder, mask_img_folder, img_size=256, mask_postfix='_cloth', transform=None
    ):
        self.unmask_img_folder = unmask_img_folder
        self.mask_img_folder = mask_img_folder
        self.mask_postfix = mask_postfix
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

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        unmask_img = Image.open(self.unmask_img_folder + os.sep + self.file_names[idx].replace(self.mask_postfix, ''))
        mask_img = Image.open(self.mask_img_folder + os.sep + self.file_names[idx])

        unmask_img_tensor = self.transform(unmask_img)
        mask_img_tensor = self.transform(mask_img)
        semantic_target = ((mask_img_tensor==unmask_img_tensor).float().sum(dim=0) == 3.).float().unsqueeze(0)

        return mask_img_tensor, unmask_img_tensor, semantic_target

class MaskingDataModule(pl.LightningDataModule):
    def __init__(self, unmask_img_folder, mask_img_folder, batch_size, train_ratio=0.8, img_size=256, mask_postfix='_cloth', transform=None):
        super(MaskingDataModule, self).__init__()
        self.unmask_img_folder = unmask_img_folder
        self.mask_img_folder = mask_img_folder
        self.mask_postfix = mask_postfix
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.transform = transform

    def setup(self, stage=None) -> None:
        full_dataset = MaskDataset(self.unmask_img_folder, self.mask_img_folder, self.img_size, self.mask_postfix, self.transform)
        self.train_set, self.val_set = torch.utils.data.random_split(
            full_dataset,
            [
                int(self.train_ratio * len(full_dataset)),
                len(full_dataset) - int(self.train_ratio * len(full_dataset)),
            ],
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
        
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=multiprocessing.cpu_count())
        
# model definition
class UnmaskingModel(pl.LightningModule):
    def __init__(self, lr=1e-4, img_size=256):
        super(UnmaskingModel, self).__init__()
        self.lr = lr
        self.generator = UNet()
        
        self.gen_loss_func = nn.L1Loss(reduction='sum')
        #self.gen_loss_func = nn.MSELoss(reduction='sum')
        
        self.tensorboard_input_imgs = []
        self.tensorboard_pred_imgs = []
        
        self.save_hyperparameters()

    def forward(self, mask_img):
        unmask_predicted = self.generator(mask_img)
        return unmask_predicted

    def predict(self, mask_img):
        with torch.no_grad():
            return self.forward(mask_img)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def training_step(self, batch, batch_idx):
        self.generator.train()

        mask_img, unmask_img, semantic_target = batch
        unmask_predicted = self.forward(mask_img)
        loss = self.gen_loss_func(unmask_predicted, unmask_img)
        
        self.log("loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        self.generator.eval()
        
        with torch.no_grad():
            mask_img, unmask_img, semantic_target = batch
            unmask_predicted = self.forward(mask_img)
            loss = self.gen_loss_func(unmask_predicted, unmask_img)
            
            self.log("val_loss", loss)
            
            if batch_idx % 3000 == 0:
                self.tensorboard_input_imgs.append(mask_img)
                self.tensorboard_pred_imgs.append(unmask_predicted)

            return loss

class PrintImageCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            input_grid = torchvision.utils.make_grid(torch.cat(pl_module.tensorboard_input_imgs), nrow=6, padding=2)
            trainer.logger.experiment.add_image(f"UnmaskingModel_epoch:{trainer.current_epoch}_inputs", input_grid, trainer.current_epoch)
        
        pred_grid = torchvision.utils.make_grid(torch.cat(pl_module.tensorboard_pred_imgs), nrow=6, padding=2)
        trainer.logger.experiment.add_image(f"UnmaskingModel_epoch:{trainer.current_epoch}_predictions", pred_grid, trainer.current_epoch)

        pl_module.tensorboard_input_imgs = []
        pl_module.tensorboard_pred_imgs = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unmask_img_folder", type=str, default="../../data/celeba-mask-pair/unmask_images/raw")
    parser.add_argument("--mask_img_folder", type=str, default="../../data/celeba-mask-pair/mask_images/raw")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()

    # model preparation
    model = UnmaskingModel(lr=args.lr)

    # data module preparation
    dataset = MaskingDataModule(
        mask_img_folder=args.mask_img_folder,
        unmask_img_folder=args.unmask_img_folder,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        img_size=args.img_size,
    )

    # register callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./',
        filename='UNet_AutoEncoder_UnmaskingModel',
        save_top_k=1,
        mode='min',
    )
    
    # training
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        progress_bar_refresh_rate=1,
        max_epochs=args.epochs,
        accelerator="ddp",
        callbacks=[checkpoint_callback, PrintImageCallback()]
    )
    trainer.fit(model, dataset)
