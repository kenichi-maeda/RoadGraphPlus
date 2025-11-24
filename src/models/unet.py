import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn



class UNet(nn.Module):
    def __init__(
            self,
            n_class
        ):
        """
        UNet model
        Args:
            n_class:    set 1 for binary segmentation
        Reference: 
        https://medium.com/data-science/cook-your-first-u-net-in-pytorch-b3297a844cf3
        """

        super().__init__()
        
        # input: 512x512
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 256x256
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 128x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 64x64
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input: 32x32
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)


        # decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)
    
    def forward(self, x: torch.Tensor):
        xe11 = F.relu(self.e11(x))
        xe12 = F.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.e21(xp1))
        xe22 = F.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.e31(xp2))
        xe32 = F.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.e41(xp3))
        xe42 = F.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))

        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.d41(xu44))
        xd42 = F.relu(self.d42(xd41))

        out = self.outconv(xd42)

        return out


class RoadMapUNet(pl.LightningModule):
    def __init__(self, **args):
        """
        Initializes PyTorch Lightning module
        """
        super().__init__()

        self.save_hyperparameters()

        self.unet = UNet(n_class=1)
        pos_weight = torch.tensor([5.0]).to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    def forward(self, images):
        """
        Forward pass
        """
        return self.unet(images)

    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
    
    def training_step(self, batch, batch_idx):
        image = batch["image"]                      # (B,3,512,512)
        mask = batch["mask"].unsqueeze(1)           # list of length B
        bs = image.size(0)

        pred = self(image)
        loss = self.loss_fn(pred, mask)

        prob = torch.sigmoid(pred)
        pred_binary = (prob > 0.5).float()

        intersection = (pred_binary * mask).sum()
        dice = 2 * intersection / (pred_binary.sum() + mask.sum() + 1e-8)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("train/acc", dice, prog_bar=True, sync_dist=True, batch_size=bs)

        return loss
    

    def validation_step(self, batch, batch_idx):
        image = batch["image"]                      # (B,3,512,512)
        mask = batch["mask"].unsqueeze(1)           # list of length B
        bs = image.size(0)

        with torch.no_grad():
            pred = self(image)

        loss = self.loss_fn(pred, mask)

        prob = torch.sigmoid(pred)
        pred_binary = (prob > 0.5).float()

        intersection = (pred_binary * mask).sum()
        dice = 2 * intersection / (pred_binary.sum() + mask.sum() + 1e-8)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("val/acc", dice, prog_bar=True, sync_dist=True, batch_size=bs)

    
    def test_step(self, batch, batch_idx):
        image = batch["image"]                          # (B,3,512,512)
        mask = batch["mask"].unsqueeze(1)               # list of length B
        bs = image.size(0)

        with torch.no_grad():
            pred = self(image)

        loss = self.loss_fn(pred, mask)

        prob = torch.sigmoid(pred)
        pred_binary = (prob > 0.5).float()

        intersection = (pred_binary * mask).sum()
        dice = 2 * intersection / (pred_binary.sum() + mask.sum() + 1e-8)

        self.log("test/loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("test/acc", dice, prog_bar=True, sync_dist=True, batch_size=bs)
    