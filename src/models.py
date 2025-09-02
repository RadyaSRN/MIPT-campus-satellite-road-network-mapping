import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import pytorch_lightning as pl
from torchmetrics.classification import BinaryJaccardIndex


class UNet(nn.Module):
  """UNet model"""
  def __init__(self, in_channels=3, out_classes=1, base_channels=64):
    """Initializes the model"""
    super(UNet, self).__init__()

    self.conv1 = self._conv_block(in_channels, base_channels)
    self.pool1 = nn.MaxPool2d(kernel_size=2)

    self.conv2 = self._conv_block(base_channels, base_channels * 2)
    self.pool2 = nn.MaxPool2d(kernel_size=2)

    self.conv3 = self._conv_block(base_channels * 2, base_channels * 4)
    self.pool3 = nn.MaxPool2d(kernel_size=2)

    self.conv4 = self._conv_block(base_channels * 4, base_channels * 8)
    self.pool4 = nn.MaxPool2d(kernel_size=2)

    self.conv5 = self._conv_block(base_channels * 8, base_channels * 16)

    self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
    self.conv_up4 = self._conv_block(base_channels * 16, base_channels * 8)

    self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
    self.conv_up3 = self._conv_block(base_channels * 8, base_channels * 4)

    self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
    self.conv_up2 = self._conv_block(base_channels * 4, base_channels * 2)

    self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
    self.conv_up1 = self._conv_block(base_channels * 2, base_channels)

    self.final_conv = nn.Conv2d(base_channels, out_classes, kernel_size=1)

  def _conv_block(self, in_channels, out_channels):
    """Convolution block"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

  def forward(self, x):
    """Forward-pass"""
    x1 = self.conv1(x)
    x2 = self.conv2(self.pool1(x1))
    x3 = self.conv3(self.pool2(x2))
    x4 = self.conv4(self.pool3(x3))
    x5 = self.conv5(self.pool4(x4))

    d4 = self.up4(x5)
    d4 = torch.cat([x4, d4], dim=1)
    d4 = self.conv_up4(d4)

    d3 = self.up3(d4)
    d3 = torch.cat([x3, d3], dim=1)
    d3 = self.conv_up3(d3)

    d2 = self.up2(d3)
    d2 = torch.cat([x2, d2], dim=1)
    d2 = self.conv_up2(d2)

    d1 = self.up1(d2)
    d1 = torch.cat([x1, d1], dim=1)
    d1 = self.conv_up1(d1)

    out = self.final_conv(d1)
    return out


class UNetLightningModel(pl.LightningModule):
  """UNet PyTorch Lightning wrapper"""
  def __init__(self, lr=1e-4, weight_decay=1e-2):
    super().__init__()
    self.model = UNet(in_channels=3, out_classes=1, base_channels=64)
    self.criterion = nn.BCEWithLogitsLoss()
    self.iou_metric = BinaryJaccardIndex()
    self.lr = lr
    self.weight_decay = weight_decay

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    imgs, masks = batch
    outputs = self(imgs)
    masks = masks.unsqueeze(1)
    loss = self.criterion(outputs, masks)
    outputs_binary = (torch.sigmoid(outputs) > 0.5).int()
    iou = self.iou_metric(outputs_binary, masks.int())

    self.log('train_loss', loss)
    self.log('train_iou', iou)
    return loss

  def validation_step(self, batch, batch_idx):
    imgs, masks = batch
    outputs = self(imgs)
    masks = masks.unsqueeze(1)
    loss = self.criterion(outputs, masks)
    outputs_binary = (torch.sigmoid(outputs) > 0.5).int()
    iou = self.iou_metric(outputs_binary, masks.int())

    self.log('val_loss', loss)
    self.log('val_iou', iou)
    return loss

  def configure_optimizers(self):
    return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class DeepLabV3LightningModel(pl.LightningModule):
  """DeepLabV3 PyTorch Lightning wrapper"""
  def __init__(self, lr=1e-4, weight_decay=1e-2):
    super().__init__()
    self.model = models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        aux_loss=False,
        num_classes=2)
    self.criterion = nn.CrossEntropyLoss()
    self.iou_metric = BinaryJaccardIndex()
    self.lr = lr
    self.weight_decay = weight_decay

  def forward(self, x):
    return self.model(x)['out']

  def training_step(self, batch, batch_idx):
    imgs, masks = batch
    outputs = self(imgs)
    masks = masks.long()
    loss = self.criterion(outputs, masks)
    iou = self.iou_metric(outputs.argmax(dim=1), masks)

    self.log('train_loss', loss)
    self.log('train_iou', iou)
    return loss

  def validation_step(self, batch, batch_idx):
    imgs, masks = batch
    outputs = self(imgs)
    masks = masks.long()
    loss = self.criterion(outputs, masks)
    iou = self.iou_metric(outputs.argmax(dim=1), masks)

    self.log('val_loss', loss)
    self.log('val_iou', iou)
    return loss

  def configure_optimizers(self):
    return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)