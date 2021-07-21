from pytorch_lightning import callbacks
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from torch_model_collection import UNet

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

import argparse
import os, sys
import multiprocessing

# dataset definition
class MaskDataset(Dataset):
    def __init__(
        self, unmask_img_folder, mask_img_folder, img_size=128, mask_postfix='_cloth', transform=None
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
        semantic_target = ((mask_img_tensor==unmask_img_tensor).float().sum(dim=1) == 3.).float().unsqueeze(1)

        return mask_img_tensor, unmask_img_tensor, semantic_target

class MaskingDataModule(pl.LightningDataModule):
    def __init__(self, unmask_img_folder, mask_img_folder, batch_size, train_ratio=0.8, img_size=128, mask_postfix='_cloth', transform=None):
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
    def __init__(self, lr=1e-4):
        super(UnmaskingModel, self).__init__()
        self.lr = lr
        self.generator = UNet(use_semantic_label=True)
        #self.loss_func = nn.MSELoss()
        self.gen_loss_func = nn.MSELoss(reduction='sum')
        self.semantic_loss_func = nn.BCELoss(reduction='sum')

        self.tensorboard_input_imgs = []
        self.tensorboard_pred_imgs = []
        
        self.save_hyperparameters()

    def forward(self, mask_img):
        unmask_predicted = self.generator(mask_img)
        return unmask_predicted

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def training_step(self, batch, batch_idx):
        self.generator.train()

        mask_img, unmask_img, semantic_target = batch
        unmask_predicted, semantic_predicted = self.forward(mask_img)
        gen_loss = self.gen_loss_func(unmask_predicted, unmask_img)
        semantic_loss = self.semantic_loss_func(semantic_predicted, semantic_target)
        loss = gen_loss + semantic_loss

        self.log("loss", loss)
        self.log("train/gen_loss", gen_loss)
        self.log("train/semantic_loss", semantic_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        self.generator.eval()
        
        with torch.no_grad():
            mask_img, unmask_img, semantic_target = batch
            unmask_predicted, semantic_predicted = self.forward(mask_img)
            gen_loss = self.gen_loss_func(unmask_predicted, unmask_img)
            semantic_loss = self.semantic_loss_func(semantic_predicted, semantic_target)
            loss = gen_loss + semantic_loss

            self.log("val_loss", loss)
            self.log("val/gen_loss", gen_loss)
            self.log("val/semantic_loss", semantic_loss)

            if batch_idx % 3000 == 0:
                self.tensorboard_input_imgs.append(mask_img)
                self.tensorboard_pred_imgs.append(unmask_predicted)

            return loss

class PrintImageCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        input_grid = torchvision.utils.make_grid(torch.cat(pl_module.tensorboard_input_imgs), nrow=6, padding=2)
        trainer.logger.experiment.add_image(f"UnmaskingModel_epoch:{trainer.current_epoch}_inputs", input_grid, trainer.current_epoch)
        
        pred_grid = torchvision.utils.make_grid(torch.cat(pl_module.tensorboard_pred_imgs), nrow=6, padding=2)
        trainer.logger.experiment.add_image(f"UnmaskingModel_epoch:{trainer.current_epoch}_predictions", pred_grid, trainer.current_epoch)

        pl_module.tensorboard_input_imgs = []
        pl_module.tensorboard_pred_imgs = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unmask_img_folder", type=str, default="../data/celeba-mask-pair/unmask_images/raw")
    parser.add_argument("--mask_img_folder", type=str, default="../data/celeba-mask-pair/mask_images/raw")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
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