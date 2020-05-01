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

	[img, _] = data.Cityscapes.transform(full_img, None)
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
	parser.add_argument("--output", "-o", metavar="OUTPUT_DIR", help="Directory for output images")
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


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

	# Parse the command-line arguments
	args = get_args()
	in_files = args.input

	assert os.path.isdir(args.output), "Output must be a directory"

	os.makedirs(args.output, exist_ok=True)

	# Load the model
	net = networks.UNet(n_channels=3, n_classes=len(data.Cityscapes.classes), bilinear=False)

	logging.info(f"Loading model {args.model}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logging.info(f"Using device {device}")
	net.to(device=device)

	if os.path.isdir(args.model):
		models = [os.path.join(args.model, m) for m in os.listdir(args.model) if m.endswith(".pth")]
	else:
		models = [args.model]

	# Predicting images
	logging.info("Model loaded !")

	for i_model, m in enumerate(models):
		net.load_state_dict(torch.load(m, map_location=device))

		for i_file, fn in enumerate(in_files):
			logging.info("\nPredicting image {} ...".format(fn))

			img = Image.open(fn)

			mask = predict_img(net=net,
							   full_img=img,
							   scale_factor=args.scale,
							   device=device)

			result = data.Cityscapes.to_image(mask)

			fname_out = os.path.join(args.output, f"{i_model+1}_{i_file+1}_result.png")

			result.save(fname_out)

			logging.info("Mask saved to {}".format(fname_out))

	logging.info("Done")

	exit(0)
