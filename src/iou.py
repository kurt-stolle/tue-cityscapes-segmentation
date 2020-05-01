import torch

from src import data


def IoU(pred_masks, true_masks, device=None) -> torch.Tensor:
	labels = data.Cityscapes.masks_to_indices(pred_masks)

	correct = labels.eq(true_masks)

	# The true positive (tp) are calculated by taking the correct predictions times the positive preditions
	tp = (correct * labels.gt(0)).sum()

	# Since this is a 2D-grid, we have tp + fp + fn + tn = (total)
	# then tp + fp + fn = (total) - tn
	den = labels.numel() - (correct * labels.eq(0)).sum()

	return tp / den.to(torch.float)
