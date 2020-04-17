import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src import loss


class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()

		if not mid_channels:
			mid_channels = out_channels

		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)


class DownScale(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			ConvBlock(in_channels, out_channels)
		)

	def forward(self, x):
		return self.maxpool_conv(x)


class UpScale(nn.Module):
	def __init__(self, in_channels, out_channels, bilinear=True):
		super().__init__()

		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
			self.conv = ConvBlock(in_channels, out_channels // 2, in_channels // 2)
		else:
			self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
			self.conv = ConvBlock(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
		diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])

		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)


class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		return self.conv(x)


class UNet(nn.Module):
	def __init__(self, n_channels, n_classes, bilinear=True):
		super().__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		self.inc = ConvBlock(n_channels, 64)
		self.down1 = DownScale(64, 128)
		self.down2 = DownScale(128, 256)
		self.down3 = DownScale(256, 512)

		factor = 2 if bilinear else 1

		self.down4 = DownScale(512, 1024 // factor)
		self.up1 = UpScale(1024, 512, bilinear)
		self.up2 = UpScale(512, 256, bilinear)
		self.up3 = UpScale(256, 128, bilinear)
		self.up4 = UpScale(128, 64 * factor, bilinear)
		self.outc = OutConv(64, n_classes)

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)

		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)

		logits = self.outc(x)

		return logits

	def eval_dice(self, loader, device):
		"""Evaluation without the densecrf with the dice coefficient"""
		self.eval()
		mask_type = torch.float32 if self.n_classes == 1 else torch.long
		n_val = len(loader)  # the number of batch
		tot = 0

		with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
			for batch in loader:
				imgs, true_masks = batch['image'], batch['mask']
				imgs = imgs.to(device=device, dtype=torch.float32)
				true_masks = true_masks.to(device=device, dtype=mask_type)

				with torch.no_grad():
					mask_pred = self(imgs)

				if self.n_classes > 1:
					tot += F.cross_entropy(mask_pred, true_masks).item()
				else:
					pred = torch.sigmoid(mask_pred)
					pred = (pred > 0.5).float()
					tot += loss.dice_coeff(pred, true_masks).item()
				pbar.update()

		return tot / n_val
