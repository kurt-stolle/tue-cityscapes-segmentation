import json
import logging
import os
import re

import numpy as np
import torch
import random

from glob import glob
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFilter
import torchvision.transforms.functional as TF
from typing import Dict, Tuple

import torchvision.transforms as T

from torchvision.datasets import Cityscapes as C


class CityscapesSample():
	"""One sample in the cityscapes dataset"""

	def __init__(self, city: str, seq_id: str, frame_id: str):
		self.city = city
		self.seq_id = seq_id
		self.frame_id = frame_id
		self.id = "_".join([city, seq_id, frame_id])


class Label():
	def __init__(self, name, color=(0, 0, 0)):
		self.name = name
		self.color = color

	def __eq__(self, other):
		return other == self.name


class Cityscapes(Dataset):
	"""The Cityscapes dataset, see: https://www.cityscapes-dataset.com/"""

	__read_reg = r"^(\w+)_(\d+)_(\d+).*.png$"

	classes = [c for i, c in enumerate(C.classes) if i == 0 or not c.ignore_in_eval]

	sample_size = (int(256), int(512))
	crop_padding = int(10)

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

		assert len(self.items) > 0, f"No items found in {self.input_dir}"

		random.seed(1958)

		logging.info(f'Loading cityscapes dataset with {len(self.items)} samples')

	def __len__(self):
		return len(self.items)

	def __getitem__(self, i: int) -> (torch.Tensor, torch.Tensor):
		# Get the sample at index i
		sample = self.items[i]

		img = self.load_image_file(sample)
		mask = self.load_colored_image(sample)
		#mask = self.load_polygons_file(sample)

		return self.transform(img, mask)

	def load_colored_image(self, sample:CityscapesSample) -> Image:
		input_file = glob(os.sep.join([self.truth_dir, sample.city, sample.id + "_gtFine_color.png"]))
		assert len(input_file) == 1, \
			f'Either no image or multiple images found for the ID {sample.id}: {input_file}'

		return Image.open(input_file[0])

	def load_polygons_file(self, sample: CityscapesSample) -> Image:
		"""load ground truths from polygons as a NumPY array, taking into account our scaling factor"""
		with open(os.sep.join([self.truth_dir, sample.city, sample.id + "_gtFine_polygons.json"])) as f:
			data = json.load(f)

		# Read the size from the data object
		size = (data["imgWidth"], data["imgHeight"])

		# Create a buffer 3d-array HW
		img_pil = Image.new('L', size, 0)
		img_pil_draw = ImageDraw.Draw(img_pil)

		# Iterate over the objects in the polygon data
		for obj in data["objects"]:
			for i, c in enumerate(self.classes):
				if c.name != obj["label"]:
					continue

				# flatten the array, [[x1,y1],[x2,y2]] -> [x1,y1,x2,x2]
				polygon = [item for sublist in obj["polygon"] for item in sublist]

				# use PIL to make an image on which we can draw the array
				img_pil_draw.polygon(polygon, outline=i, fill=i)

				break

		return img_pil

	def load_image_file(self, sample: CityscapesSample) -> Image:
		"""load an image file and parse to a NumPY array, taking into account our scaling factor"""
		input_file = glob(os.sep.join([self.input_dir, sample.city, sample.id + "_leftImg8bit.png"]))
		assert len(input_file) == 1, \
			f'Either no image or multiple images found for the ID {sample.id}: {input_file}'

		return Image.open(input_file[0])

	@classmethod
	def transform(cls, img: Image, mask: Image) -> (torch.Tensor, torch.Tensor):
		"""perform data augmentation"""

		img = img.convert("RGB")
		img = img.filter(ImageFilter.SHARPEN)
		#  img.putalpha(img.filter(ImageFilter.FIND_EDGES).convert("L"))

		if mask is None:
			return TF.to_tensor(TF.resize(img, cls.sample_size)), None

		mask = mask.convert("RGB")

		img = TF.resize(img,cls.sample_size, interpolation=Image.BILINEAR)
		mask = TF.resize(mask, cls.sample_size, interpolation=Image.NEAREST)

		if random.random() > 0.5:
			img = TF.hflip(img)
			mask = TF.hflip(mask)

		# Transform the Img to a CHW-dimensional Tensor
		img = TF.to_tensor(img)

		# Transform the mask from an image with RGB-colors to an 1-channel image with the index of the class as value
		mask_size = [s for s in cls.sample_size]
		mask = TF.resize(mask, mask_size, Image.NEAREST)
		mask = torch.from_numpy(np.array(mask)).permute((2,0,1))
		target = torch.zeros(mask_size, dtype=torch.uint8)
		for i,c in enumerate(cls.classes):
			eq = mask[0].eq(c.color[0]) & mask[1].eq(c.color[1]) & mask[2].eq(c.color[2])
			target += eq * i

		return img, target

	@staticmethod
	def masks_to_indices(masks: torch.Tensor) -> torch.Tensor:
		vals, indices = masks.softmax(dim=1).max(dim=1)
		#_, indices = masks.max(dim=1)

		return vals.gt(0.1) * indices

	@classmethod
	def to_image(cls, masks: torch.Tensor) -> Image:
		"""Converts a tensor([1, class_index, with, height] = logit) to an image"""

		assert masks.shape[0] == 1, f"Image conversion only works on a single masks collection (shape = {masks.shape})"
		assert masks.shape[1] == len(cls.classes), f"The masks Tensor's first dimension (shape = {masks.shape}) " \
												   f"does not match the amount of labels ({len(cls.classes)})"

		indices = cls.masks_to_indices(masks).squeeze(0)

		target = torch.zeros((3, masks.shape[2], masks.shape[3]),
							 dtype=torch.uint8, device=indices.device, requires_grad=False)

		print("matching pixels with classes")
		for i, lbl in enumerate(cls.classes):
			eq = indices.eq(i)

			target[0] += eq * lbl.color[0]
			target[1] += eq * lbl.color[1]
			target[2] += eq * lbl.color[2]

		print("converting to PIL image")

		return TF.to_pil_image(target.cpu(), 'RGB')
