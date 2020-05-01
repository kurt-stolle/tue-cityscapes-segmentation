#!/usr/bin/env python3

import argparse
import csv
import logging
import os
import sys
from typing import List

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from tqdm import tqdm

from src import networks
from src import data
from src import iou


def train(model, dataset_train, dataset_val, epoch, epochs, train_loader, val_loader, device, criterion, metrics,
		  optimizer, scheduler, global_step) -> int:
	with tqdm(total=len(dataset_train), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
		for imgs, true_masks in train_loader:
			assert imgs.shape[1] == model.n_channels, \
				f'Network has been defined with {model.n_channels} input channels, ' \
				f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
				'the images are loaded correctly.'

			model.train()

			optimizer.zero_grad()

			imgs = imgs.to(device=device, dtype=torch.float32)
			imgs.required_grad = True
			true_masks = true_masks.to(device=device, dtype=torch.long)

			masks_pred = model(imgs)

			m_loss = criterion(masks_pred, true_masks)
			m_loss_item = m_loss.item()

			m_iou = iou.IoU(masks_pred, true_masks, device=device)
			m_iou_item = m_iou.item()

			(m_loss + (1 - m_iou) * 0.1).backward()

			nn.utils.clip_grad_value_(model.parameters(), 0.1)
			optimizer.step()

			pbar.set_postfix(**{'loss': m_loss_item, 'iou': m_iou_item})
			pbar.update(imgs.shape[0])

			metrics["train_loss"].append((global_step, m_loss_item))
			metrics["train_iou"].append((global_step, m_iou_item))

			if (global_step % (len(dataset_train) // train_loader.batch_size // 4)) == 0:
				val(model, dataset_val, val_loader, device, criterion, metrics, scheduler, global_step)

			global_step += 1

	return global_step


def val(model, dataset_val, val_loader, device, criterion, metrics, scheduler, global_step):
	model.eval()

	logging.info("\nValidating current state...")

	n_val = len(dataset_val) // val_loader.batch_size  # the number of batch
	tot_loss = 0
	tot_iou = 0
	for imgs, true_masks in val_loader:
		imgs = imgs.to(device=device, dtype=torch.float32)
		true_masks = true_masks.to(device=device, dtype=torch.long)

		with torch.no_grad():
			mask_pred = model(imgs)

		tot_loss += criterion(mask_pred, true_masks).item()
		tot_iou += iou.IoU(mask_pred, true_masks, device=device).item()

	val_score = tot_loss / n_val
	iou_score = tot_iou / n_val

	scheduler.step(iou_score)

	metrics["val_loss"].append((global_step, val_score))
	metrics["val_iou"].append((global_step, iou_score))

	logging.info(f'Validation cross entropy : {val_score}')
	logging.info(f'Validation IoU           : {iou_score}')


def train_net(model: nn.Module,
			  device,
			  dataset_train: Dataset,
			  dataset_val: Dataset,
			  epochs=10,
			  batch_size=1,
			  lr=0.01,
			  checkpoint_dir=None):
	# Load the network to the device
	model.to(device=device)

	# Define loaders for each split
	train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
	val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False,
							drop_last=True)

	# Logging
	logging.info(
		f"Starting training with params:\n\t- Epochs = {epochs}\n\t- Batch Size = {batch_size}\n\t- Learnig Rate = {lr}")

	# Define our optimization strategy
	optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
	#optimizer = optim.Adam((model.parameters()), lr=lr, weight_decay=1e-8)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, verbose=True, threshold=0.001)
	criterion = nn.CrossEntropyLoss()

	# Count steps
	global_step = 0

	# Collect metrics from the model
	metrics = {"train_iou": [], "val_iou": [], "train_loss": [], "val_loss": []}

	for epoch in range(epochs):
		global_step = train(model, dataset_train, dataset_val, epoch, epochs, train_loader, val_loader, device,
							criterion, metrics, optimizer, scheduler, global_step)

		# Save checkpoint
		if checkpoint_dir is not None:
			os.makedirs(checkpoint_dir, exist_ok=True)
			torch.save(model.state_dict(),
					   os.path.join(checkpoint_dir, f"{epoch + 1}.pth"))

			logging.info(f'Checkpoint {epoch + 1} saved !')

	return metrics


def get_args():
	parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-i', "--input-dir",
						metavar="INPUT",
						type=str,
						required=True)
	parser.add_argument('-d', "--metrics-dir",
						metavar="METRICS_DIR",
						type=str,
						default=None)
	parser.add_argument('-e', '--epochs',
						metavar='E',
						type=int,
						default=10,
						help='Number of epochs',
						dest='epochs')
	parser.add_argument('-b', '--batch-size',
						metavar='B',
						type=int, nargs='?',
						default=1,
						help='Batch size',
						dest='batchsize')
	parser.add_argument('-l', '--learning-rate',
						metavar='LR',
						type=float, nargs='?',
						default=0.001,
						help='Learning rate',
						dest='lr')
	parser.add_argument('-m', '--model',
						dest='model',
						type=str,
						default=False,
						help='Model .pth file')
	parser.add_argument('-s', '--scale',
						dest='scale',
						type=float,
						default=0.5,
						help='Downscaling factor of the images')

	return parser.parse_args()


def save_csv(path: str, l: List):
	with open(path, 'w', newline='') as f:
		w = csv.writer(f)
		w.writerow(("global_step", "value"))
		w.writerows(l)


def main():
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

	# Parse the CLI arguments
	args = get_args()

	# Create the network
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logging.info(f'Using device {device}')

	net = networks.UNet(n_channels=3, n_classes=len(data.Cityscapes.classes), bilinear=False)

	print(net)

	if args.model:
		net.load_state_dict(
			torch.load(args.model, map_location=device)
		)
		logging.info(f'Model loaded from {args.model}')

	cudnn.benchmark = True

	# Load the validating and training dataset
	dataset_train = data.Cityscapes(
		os.sep.join((args.input_dir, "leftImg8bit", "train")),
		os.sep.join((args.input_dir, "gtFine", "train")),
		args.scale
	)
	dataset_val = data.Cityscapes(
		os.sep.join((args.input_dir, "leftImg8bit", "val")),
		os.sep.join((args.input_dir, "gtFine", "val")),
		args.scale
	)

	# Start training
	try:
		metrics = train_net(net,
							device,
							dataset_train,
							dataset_val,
							epochs=args.epochs,
							batch_size=args.batchsize,
							lr=args.lr,
							checkpoint_dir=os.path.join(args.metrics_dir, "cp"))

		if args.metrics_dir is None:
			print("Not saving metrics")
		else:
			os.makedirs(args.metrics_dir, exist_ok=True)
			save_csv(os.path.join(args.metrics_dir, "train_loss.csv"), metrics["train_loss"])
			save_csv(os.path.join(args.metrics_dir, "val_loss.csv"), metrics["val_loss"])
			save_csv(os.path.join(args.metrics_dir, "train_iou.csv"), metrics["train_iou"])
			save_csv(os.path.join(args.metrics_dir, "val_iou.csv"), metrics["val_iou"])

			logging.info(f"Saved metrics in {args.metrics_dir}")
	except KeyboardInterrupt:
		torch.save(net.state_dict(), 'INTERRUPTED.pth')
		logging.info('Saved interrupt')
		try:
			sys.exit(0)
		except SystemExit:
			os._exit(0)


if __name__ == "__main__":
	main()
