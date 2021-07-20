from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from torch_model_collection import UNet

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

import argparse
import os, sys
import multiprocessing

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

        self.loss_func = nn.BCELoss()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        unmask_img = Image.open(self.unmask_img_folder + os.sep + self.file_names[idx].replace(self.mask_postfix, ''))
        mask_img = Image.open(self.mask_img_folder + os.sep + self.file_names[idx])

        unmask_img_tensor = self.transform(unmask_img)
        mask_img_tensor = self.transform(mask_img)

        return mask_img_tensor, unmask_img_tensor


class UnmaskingModel(pl.LightningModule):
    def __init__(self, img_size=128, lr=1e-4):
        super(UnmaskingModel, self).__init__()
        self.img_size = img_size
        self.lr = lr
        self.loss_func = nn.MSELoss()
        self.model = UNet()
        
        self.save_hyperparameters()

    def forward(self, mask_img):
        unmask_predicted = self.model(mask_img)

        return unmask_predicted
        

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def training_step(self, batch, batch_idx):
        self.model.train()

        mask_img, unmask_img = batch
        unmask_predicted = self.forward(mask_img)
        loss = self.loss_func(unmask_predicted, unmask_img)
        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        
        with torch.no_grad():
            mask_img, unmask_img = batch
            unmask_predicted = self.forward(mask_img)
            loss = self.loss_func(unmask_predicted, unmask_img)
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
        mask_img_folder="../data/celeba-mask-pair/mask_images/raw",
        unmask_img_folder="../data/celeba-mask-pair/unmask_images/raw",
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
