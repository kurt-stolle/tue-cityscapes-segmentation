#!/usr/bin/env python3

import argparse

from src import data

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
	parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
						help='Percent of the data that is used as validation (0-100)')

	return parser.parse_args()


if __name__ == "__main__":
	args = get_args()

	data = data.Cityscapes(args.input_dir, args.truth_dir)

	for f in data:
		print(f)
