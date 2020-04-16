import logging
import os
import re

import numpy as np
import torch

from glob import glob
from torch.utils.data import Dataset
from PIL import Image
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

		# Read the truth and input files
		truth_file = glob(os.sep.join([self.truth_dir, sample.city, sample_id + "_gtFine_color.png"]))
		input_file = glob(os.sep.join([self.input_dir, sample.city, sample_id + "_leftImg8bit.png"]))

		assert len(truth_file) == 1, \
			f'Either no truth mask or multiple truth masks found for the ID {sample_id}: {truth_file}'
		assert len(input_file) == 1, \
			f'Either no image or multiple images found for the ID {sample_id}: {input_file}'

		# Open with PIL
		mask = Image.open(truth_file[0])
		img = Image.open(input_file[0])

		assert img.size == mask.size, \
			f'Image and mask {sample_id} should be the same size, but are {img.size} and {mask.size}'

		# Preprocess the truth
		# Fixme

		# Preprocess the image
		img = self.preprocess(img, self.scale)

		return {'input': torch.from_numpy(img), 'truth': torch.from_numpy(mask)}

	@staticmethod
	def downscale(img_pil: Image, scale: float):
		# Image scaling
		w, h = img_pil.size
		w, h = int(scale * w), int(scale * h)

		assert w > 0 and h > 0, 'Scale is too small'

		return img_pil.resize((w, h))

	@classmethod
	def color_to_truth(cls, img_pil: Image, scale: float) -> np.ndarray:
		# Downscale the image
		img_pil = cls.downscale(img_pil, scale)

		# Convert to 2d array
		img_np = np.ndarray()

		return img_np

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
