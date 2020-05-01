import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import get_device_states, check_backward_validity, CheckpointFunction, \
	checkpoint_sequential, checkpoint
from tqdm import tqdm


class ConvBlock(nn.Sequential):
	def __init__(self, in_channels, out_channels, mid_channels=None):
		if not mid_channels:
			mid_channels = out_channels

		super().__init__(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			#  nn.Dropout2d(0.2),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			#  nn.Dropout2d(0.2)
		)


class DownScale(nn.Sequential):
	def __init__(self, in_channels, out_channels):
		super().__init__(
			nn.MaxPool2d(2),
			ConvBlock(in_channels, out_channels)
		)


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
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

		x = torch.cat([x2, x1], dim=1)

		return self.conv(x)


class OutConv(nn.Conv2d):
	def __init__(self, in_channels, out_channels):
		super().__init__(in_channels, out_channels, kernel_size=1)


class UNet(nn.Module):
	def __init__(self, n_channels, n_classes, bilinear=True):
		super().__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		nf = 54

		self.inc = nn.Sequential(
			# out_size = ((I+P) / F ) / S + 1
			nn.Conv2d(n_channels, nf, kernel_size=3, padding=1, padding_mode="circular"),
			nn.BatchNorm2d(nf),
			nn.ReLU(inplace=True),
			nn.Conv2d(nf, nf, kernel_size=3, padding=1, padding_mode="circular"),
			nn.BatchNorm2d(nf),
			nn.ReLU(inplace=True),
		)

		self.down1 = DownScale(nf * 2 ** 0, nf * 2 ** 1)
		self.down2 = DownScale(nf * 2 ** 1, nf * 2 ** 2)
		self.down3 = DownScale(nf * 2 ** 2, nf * 2 ** 3)

		factor = 2 if bilinear else 1

		self.down4 = DownScale(nf * 2 ** 3, nf * 2 ** 4 // factor)

		self.up1 = UpScale(nf * 2 ** 4, nf * 2 ** 3, bilinear)
		self.up2 = UpScale(nf * 2 ** 3, nf * 2 ** 2, bilinear)
		self.up3 = UpScale(nf * 2 ** 2, nf * 2 ** 1, bilinear)
		self.up4 = UpScale(nf * 2 ** 1, nf * 2 ** 0 * factor, bilinear)

		self.outc = OutConv(nf, n_classes)

	# self.inc = ConvBlock(n_channels, 64)
	#
	# self.down1 = DownScale(64, 128)
	# self.down2 = DownScale(128, 256)
	# self.down3 = DownScale(256, 512)
	#
	# factor = 2 if bilinear else 1
	#
	# self.down4 = DownScale(512, 1024 // factor)
	#
	# self.up1 = UpScale(1024, 512, bilinear)
	# self.up2 = UpScale(512, 256, bilinear)
	# self.up3 = UpScale(256, 128, bilinear)
	# self.up4 = UpScale(128, 64 * factor, bilinear)
	#
	# self.outc = OutConv(64, n_classes)

	def forward(self, x):
		xi = self.inc(x)

		xd1 = self.down1(xi)
		xd2 = self.down2(xd1)
		xd3 = self.down3(xd2)
		xd4 = self.down4(xd3)

		xu1 = self.up1(xd4, xd3)
		xu2 = self.up2(xu1, xd2)
		xu3 = self.up3(xu2, xd1)
		xu4 = self.up4(xu3, xi)

		logits = self.outc(xu4)

		return logits
