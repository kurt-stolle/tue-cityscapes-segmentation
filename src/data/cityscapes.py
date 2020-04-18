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

	def __init__(self, input_dir: str, truth_dir: str, scale=1, cache_dir: str = None):
		self.input_dir = input_dir
		self.truth_dir = truth_dir
		self.cache_dir = cache_dir
		self.scale = scale

		assert 0 < scale <= 1, "Scale must be between 0 and 1"

		# Create the cache directory if it doesn't exist yet
		if cache_dir is not None:
			os.makedirs(self.cache_dir, exist_ok=True)

		# Walk the inputs directory and all each file to our items list
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

		# Allocate as None to help the linter
		img = None
		mask = None

		# Check if there is a cache directory
		if self.cache_dir is not None:
			fname_cache_img = os.sep.join((self.cache_dir, f"{sample.id}_@{self.scale}x_image.npy"))
			fname_cache_mask = os.sep.join((self.cache_dir, f"{sample.id}_@{self.scale}x_mask.npy"))

			try:
				img = np.load(fname_cache_img, allow_pickle=False)
				mask = np.load(fname_cache_mask, allow_pickle=False)
			except IOError:
				img = self.load_image_file(sample)
				mask = self.load_polygons_file(sample)
			finally:
				np.save(fname_cache_img, img, allow_pickle=False)
				np.save(fname_cache_mask, mask, allow_pickle=False)

			assert img is not None, "Image not loaded correctly, try clearing your cache"
			assert mask is not None, "Masks not loaded correctly, try clearing your cache"
		else:
			img = self.load_image_file(sample)
			mask = self.load_polygons_file(sample)

		assert img[0].shape == mask.shape, \
			"Image and Masks are not the same shape. Check the ground truth and input image dimensions"

		return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

	def load_polygons_file(self, sample: CityscapesSample) -> np.ndarray:
		"""load ground truths from polygons as a NumPY array, taking into account our scaling factor"""
		with open(os.sep.join([self.truth_dir, sample.city, sample.id + "_gtFine_polygons.json"])) as f:
			data = json.load(f)

		return self.preprocess_polygons(data, self.scale)

	def load_image_file(self, sample: CityscapesSample) -> np.ndarray:
		"""load an image file and parse to a NumPY array, taking into account our scaling factor"""
		input_file = glob(os.sep.join([self.input_dir, sample.city, sample.id + "_leftImg8bit.png"]))
		assert len(input_file) == 1, \
			f'Either no image or multiple images found for the ID {sample.id}: {input_file}'

		return self.preprocess(Image.open(input_file[0]), self.scale)

	@staticmethod
	def downscale(img_pil: Image, scale: float) -> Image:
		"""downscale a PIL image"""
		# Image scaling
		w, h = img_pil.size
		w, h = int(scale * w), int(scale * h)

		assert w > 0 and h > 0, 'Scale is too small'

		return img_pil.resize((w, h))

	@classmethod
	def preprocess_polygons(cls, data: Dict, scale: float) -> (np.ndarray):
		"""preprocess a polygons file to an array of masks"""

		# Read the size from the data object
		size = (data["imgWidth"], data["imgHeight"])

		# Create a buffer 3d-array HW
		img_pil = Image.new('L', size, -1)

		# Iterate over the objects in the polygon data
		for obj in data["objects"]:
			try:
				i = cls.labels.index(obj["label"])
			except ValueError:
				# logging.warning(f"""Polygons file contains unexpected label {obj["label"]}""")
				continue

			# flatten the array, [[x1,y1],[x2,y2]] -> [x1,y1,x2,x2]
			polygon = [item for sublist in obj["polygon"] for item in sublist]

			# use PIL to make an image on which we can draw the array
			ImageDraw.Draw(img_pil).polygon(polygon, outline=i, fill=i)

		return np.array(cls.downscale(img_pil, scale))

	@classmethod
	def preprocess(cls, img_pil: Image, scale: float) -> np.ndarray:
		"""preprocess an image file to an array of channels"""

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
