#!/usr/bin/env python3

import argparse
import logging
import os
import sys

import torch
from torch import nn, optim
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src import networks
from src import data


def train_net(model: nn.Module,
			  device,
			  input_dir: str,
			  truth_dir: str,
			  epochs=5,
			  batch_size=1,
			  lr=0.001,
			  save_cp=True,
			  img_scale=0.5):
	# Load the validating and training dataset
	dataset_train = data.Cityscapes(os.sep.join((input_dir, "train")), os.sep.join((truth_dir, "train")), img_scale)
	dataset_val = data.Cityscapes(os.sep.join((input_dir, "val")), os.sep.join((truth_dir, "val")), img_scale)

	# Define loaders for each split
	train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
	val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False, drop_last=True)

	# Count steps
	global_step = 0

	# Logging
	logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(dataset_train)}
        Validation size: {len(dataset_val)}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

	#optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8, )
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if model.n_classes > 1 else 'max', patience=2)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(epochs):
		model.train()

		epoch_loss = 0
		with tqdm(total=len(dataset_train), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
			for batch in train_loader:
				imgs = batch['image']
				true_masks = batch['mask']
				assert imgs.shape[1] == model.n_channels, \
					f'Network has been defined with {model.n_channels} input channels, ' \
					f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
					'the images are loaded correctly.'

				imgs = imgs.to(device=device, dtype=torch.float32)
				true_masks = true_masks.to(device=device, dtype=torch.long)

				masks_pred = model(imgs)
				loss = criterion(masks_pred, true_masks)
				epoch_loss += loss.item()

				pbar.set_postfix(**{'loss (batch)': loss.item()})

				optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_value_(model.parameters(), 0.1)
				optimizer.step()

				pbar.update(imgs.shape[0])
				global_step += 1
				if global_step % (len(dataset_train) // (10 * batch_size)) == 0:
					val_score = model.eval_dice(val_loader, device)
					scheduler.step(val_score)

					if model.n_classes > 1:
						logging.info('Validation cross entropy: {}'.format(val_score))
					else:
						logging.info('Validation Dice Coeff: {}'.format(val_score))

# if save_cp:
# 	try:
# 		os.mkdir(dir_checkpoint)
# 		logging.info('Created checkpoint directory')
# 	except OSError:
# 		pass
# 	torch.save(net.state_dict(),
# 			   dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
# 	logging.info(f'Checkpoint {epoch + 1} saved !')


def get_args():
	parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-i', "--input-dir", metavar="INPUT", type=str, required=True)
	parser.add_argument('-t', "--truth-dir", metavar="TRUTH", type=str, required=True)
	parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
						help='Number of epochs', dest='epochs')
	parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
						help='Batch size', dest='batchsize')
	parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
						help='Learning rate', dest='lr')
	parser.add_argument('-m', '--model', dest='model', type=str, default=False,
						help='Model .pth file')
	parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
						help='Downscaling factor of the images')

	return parser.parse_args()


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

	# Parse the CLI arguments
	args = get_args()

	# Create the network
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logging.info(f'Using device {device}')

	net = networks.UNet(n_channels=3, n_classes=len(data.Cityscapes.labels), bilinear=True)

	if args.model:
		net.load_state_dict(
			torch.load(args.model, map_location=device)
		)
		logging.info(f'Model loaded from {args.model}')

	net.to(device=device)

	# cudnn.benchmark = True

	try:
		train_net(net,
				  device,
				  args.input_dir,
				  args.truth_dir,
				  epochs=args.epochs,
				  batch_size=args.batchsize,
				  lr=args.lr,
				  img_scale=args.scale)
	except KeyboardInterrupt:
		torch.save(net.state_dict(), 'INTERRUPTED.pth')
		logging.info('Saved interrupt')
		try:
			sys.exit(0)
		except SystemExit:
			os._exit(0)
