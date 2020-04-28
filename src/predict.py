#!/usr/bin/env python3

import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F

import torchvision.transforms.functional as TF

from PIL import Image
from torchvision import transforms

from src import networks
from src import data


def predict_img(net,
				full_img,
				device,
				scale_factor=0.2) -> torch.Tensor:
	net.eval()

	img = TF.to_tensor(data.Cityscapes.downscale(full_img, scale_factor))
	img = img.unsqueeze(0)
	img = img.to(device=device, dtype=torch.float32)

	with torch.no_grad():
		output = net(img)

	return output


def get_args():
	parser = argparse.ArgumentParser(description="Predict masks from input images",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--model", "-m", default="MODEL.pth",
						metavar="FILE",
						help="Specify the file in which the model is stored")
	parser.add_argument("--input", "-i", metavar="INPUT", nargs="+",
						help="filenames of input images", required=True)
	parser.add_argument("--output", "-o", metavar="INPUT", nargs="+",
						help="Filenames of ouput images")
	parser.add_argument("--viz", "-v", action="store_true",
						help="Visualize the images as they are processed",
						default=False)
	parser.add_argument("--no-save", "-n", action="store_true",
						help="Do not save the output masks",
						default=False)
	parser.add_argument("--mask-threshold", "-t", type=float,
						help="Minimum probability value to consider a mask pixel white",
						default=0.5)
	parser.add_argument("--scale", "-s", type=float,
						help="Scale factor for the input images",
						default=0.5)

	return parser.parse_args()


def get_output_filenames(args):
	in_files = args.input
	out_files = []

	if not args.output:
		for f in in_files:
			pathsplit = os.path.splitext(f)
			out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
	elif len(in_files) != len(args.output):
		logging.error("Input files and output files are not of the same length")
		raise SystemExit()
	else:
		out_files = args.output

	return out_files


def mask_to_image(mask):
	return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

	# Parse the command-line arguments
	args = get_args()
	in_files = args.input
	out_files = get_output_filenames(args)

	# Load the model
	net = networks.UNet(n_channels=3, n_classes=len(data.Cityscapes.classes))

	logging.info(f"Loading model {args.model}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logging.info(f"Using device {device}")
	net.to(device=device)
	net.load_state_dict(torch.load(args.model, map_location=device))

	# Predicting images
	logging.info("Model loaded !")

	for i, fn in enumerate(in_files):
		logging.info("\nPredicting image {} ...".format(fn))

		img = Image.open(fn)

		mask = predict_img(net=net,
						   full_img=img,
						   scale_factor=args.scale,
						   device=device)

		result = data.Cityscapes.to_image(mask)

		if not args.no_save:
			out_fn = out_files[i]
			result.save(out_files[i])

			logging.info("Mask saved to {}".format(out_files[i]))

	logging.info("Done")

	exit(0)
