import json
import logging
import os
import re

import numpy as np
import torch
import random

from glob import glob
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from typing import Dict

import torchvision.transforms as T


class CityscapesSample():
	"""One sample in the cityscapes dataset"""

	def __init__(self, city: str, seq_id: str, frame_id: str):
		self.city = city
		self.seq_id = seq_id
		self.frame_id = frame_id
		self.id = "_".join([city, seq_id, frame_id])


class Cityscapes(Dataset):
	"""The Cityscapes dataset, see: https://www.cityscapes-dataset.com/"""

	__read_reg = r"^(\w+)_(\d+)_(\d+).*.png$"

	labels = (
		"road", "sidewalk", "parking", "rail track",  # flat
		"person", "rider",  # human
		"car", "truck", "bus", "on rails", "motorcycle", "bicycle", "caravan", "trailer",  # vehicle
		"building", "wall", "fence", "guard rail", "bridge", "tunnel",  # construction
		"pole", "pole group", "traffic sign", "traffic light",  # object
		"vegetation", "terrain",  # nature
		"sky",  # sky
		"ground", "dynamic", "static"  # void
	)

	def __init__(self, input_dir: str, truth_dir: str, scale=1):
		self.input_dir = input_dir
		self.truth_dir = truth_dir
		self.scale = scale

		assert 0 < scale <= 1, "Scale must be between 0 and 1"

		# Walk the inputs directory and all each file to our items list
		self.items = []
		for (_, _, filenames) in os.walk(self.input_dir):
			for filename in filenames:
				match = re.match(self.__read_reg, filename, re.I)

				if match:
					self.items.append(CityscapesSample(match.group(1), match.group(2), match.group(3)))

		random.seed(1958)

		logging.info(f'Loading cityscapes dataset with {len(self.items)} samples')

	def __len__(self):
		return len(self.items)

	def __getitem__(self, i: int) -> (torch.Tensor, torch.Tensor):
		# Get the sample at index i
		sample = self.items[i]

		img = self.load_image_file(sample)
		mask = self.load_polygons_file(sample)

		return self.transform(img, mask)

	def load_polygons_file(self, sample: CityscapesSample) -> Image:
		"""load ground truths from polygons as a NumPY array, taking into account our scaling factor"""
		with open(os.sep.join([self.truth_dir, sample.city, sample.id + "_gtFine_polygons.json"])) as f:
			data = json.load(f)

		# Read the size from the data object
		size = (data["imgWidth"], data["imgHeight"])

		# Create a buffer 3d-array HW
		img_pil = Image.new('L', size, -1)

		# Iterate over the objects in the polygon data
		for obj in data["objects"]:
			try:
				i = self.labels.index(obj["label"])
			except ValueError:
				# logging.warning(f"""Polygons file contains unexpected label {obj["label"]}""")
				continue

			# flatten the array, [[x1,y1],[x2,y2]] -> [x1,y1,x2,x2]
			polygon = [item for sublist in obj["polygon"] for item in sublist]

			# use PIL to make an image on which we can draw the array
			ImageDraw.Draw(img_pil).polygon(polygon, outline=i, fill=i)

		return img_pil

	def load_image_file(self, sample: CityscapesSample) -> Image:
		"""load an image file and parse to a NumPY array, taking into account our scaling factor"""
		input_file = glob(os.sep.join([self.input_dir, sample.city, sample.id + "_leftImg8bit.png"]))
		assert len(input_file) == 1, \
			f'Either no image or multiple images found for the ID {sample.id}: {input_file}'

		return Image.open(input_file[0])

	def transform(self, img: Image, mask: Image) -> (torch.Tensor, torch.Tensor):
		"""perform data augmentation"""

		img = self.downscale(img, self.scale)
		mask = self.downscale(mask, self.scale)

		jitter = T.ColorJitter.get_params((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))
		img = jitter(img)
		#
		# rotation = random.random() * 10
		# img = TF.rotate(img, rotation, expand=True)
		# mask = TF.rotate(img, rotation, expand=True)
		#
		# if random.random() > 0.9:
		# 	img = TF.to_grayscale(img, 3)

		if random.random() > 0.5:
			img = TF.hflip(img)
			mask = TF.hflip(mask)

		img = TF.to_tensor(img)
		mask = torch.from_numpy(np.array(mask))

		return img, mask

	@staticmethod
	def downscale(img_pil: Image, scale: float) -> Image:
		"""downscale a PIL image"""
		# Image scaling
		w, h = img_pil.size
		w, h = int(scale * w), int(scale * h)

		assert w > 0 and h > 0, 'Scale is too small'

		return img_pil.resize((w, h), resample=Image.NEAREST)
