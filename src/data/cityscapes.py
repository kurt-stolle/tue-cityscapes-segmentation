import json
import logging
import os
import re

import numpy as np
import torch

from glob import glob
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from typing import Dict


class CityscapesSample():
	"""One sample in the cityscapes dataset"""

	def __init__(self, city: str, seq_id: str, frame_id: str):
		self.city = city;
		self.seq_id = seq_id;
		self.frame_id = frame_id

	def get_id(self) -> str:
		return "_".join([self.city, self.seq_id, self.frame_id])


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

		self.items = []
		for (_, _, filenames) in os.walk(self.input_dir):
			for filename in filenames:
				match = re.match(self.__read_reg, filename, re.I)

				if match:
					self.items.append(CityscapesSample(match.group(1), match.group(2), match.group(3)))

		logging.info(f'Loading cityscapes dataset with {len(self.items)} samples')

	def __len__(self):
		return len(self.items)

	def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
		# Get the sample at index i
		sample = self.items[i]
		sample_id = sample.get_id()

		# Preprocess the truth
		with open(os.sep.join([self.truth_dir, sample.city, sample_id + "_gtFine_polygons.json"])) as f:
			data = json.load(f)
			mask = self.preprocess_polygons(data, self.scale)

		# Preprocess the image
		input_file = glob(os.sep.join([self.input_dir, sample.city, sample_id + "_leftImg8bit.png"]))
		assert len(input_file) == 1, \
			f'Either no image or multiple images found for the ID {sample_id}: {input_file}'

		img = self.preprocess(Image.open(input_file[0]), self.scale)

		# assert img.size == mask.size, \
		# 	f'Image and mask {sample_id} should be the same size, but are {img.size} and {mask.size}'

		return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

	@staticmethod
	def downscale(img_pil: Image, scale: float) -> Image:
		# Image scaling
		w, h = img_pil.size
		w, h = int(scale * w), int(scale * h)

		assert w > 0 and h > 0, 'Scale is too small'

		return img_pil.resize((w, h))

	@classmethod
	def preprocess_polygons(cls, data: Dict, scale: float) -> (np.ndarray):
		# Read the size from the data object
		size = (data["imgWidth"], data["imgHeight"])

		# Create a buffer 3d-array CHW
		result = np.empty((len(cls.labels), int(size[1] * scale), int(size[0] * scale)))

		# Iterate over the objects in the polygon data
		for obj in data["objects"]:
			try:
				i = cls.labels.index(obj["label"])
			except ValueError:
				#logging.warning(f"""Polygons file contains unexpected label {obj["label"]}""")
				continue

			# flatten the array, [[x1,y1],[x2,y2]] -> [x1,y1,x2,x2]
			polygon = [item for sublist in obj["polygon"] for item in sublist]

			# use PIL to make an image on which we can draw the array
			img_pil = Image.new('L', size, 0)
			ImageDraw.Draw(img_pil).polygon(polygon, outline=1, fill=1)

			# scale the mask
			img_pil = cls.downscale(img_pil, scale)

			# convert to numpy array an add at the correct index
			img_np = np.array(img_pil)

			result[i] = img_np

		return result

	@classmethod
	def preprocess(cls, img_pil: Image, scale: float) -> np.ndarray:
		# Convert to numpy 3d array
		img_np = np.array(cls.downscale(img_pil, scale))

		if len(img_np.shape) == 2:
			img_np = np.expand_dims(img_np, axis=2)

		# HWC to CHW
		img_np = img_np.transpose((2, 0, 1))

		# Scale from 0-255 to 0-1
		if img_np.max() > 1:
			img_np = img_np / 255

		return img_np
