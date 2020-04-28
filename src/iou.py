import torch

from src import data


def IoU(pred_masks, true_masks, device=None) -> torch.Tensor:
	labels = data.Cityscapes.masks_to_indices(pred_masks)

	positive = labels.gt(0)
	negative = labels.eq(0)

	true_positive = positive.eq(true_masks)
	true_negative = negative.eq(true_masks)

	false_positive = positive & ~true_positive
	false_negative = negative & ~true_negative

	score_tp = true_positive.sum()
	score_fp = false_positive.sum()
	score_fn = false_negative.sum()

	den = score_tp + score_fp + score_fn

	return score_fp / den.to(torch.float)
