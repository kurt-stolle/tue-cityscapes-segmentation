import os
import torch
import torchvision.transforms.functional as TF
from src import data

if __name__ != "__main__":
	exit(1)

path_in ="D:\Cityscapes\leftImg8bit"
path_gt ="D:\Cityscapes\gtFine"
mode = "train"

print("opening Cityscapes dataset")
ds= data.Cityscapes(os.path.join(path_in,mode), os.path.join(path_gt, mode))

print("interating over set")
input, truth = ds.__getitem__(0)

device = torch.device("cuda")

truth = truth.to(device)
target = torch.zeros((3, truth.shape[0], truth.shape[1]),
					 dtype=torch.uint8, device=device, requires_grad=False)

print("matching pixels with classes")
for i, lbl in enumerate(ds.classes):
	eq = truth.eq(i)

	target[0] += eq * lbl.color[0]
	target[1] += eq * lbl.color[1]
	target[2] += eq * lbl.color[2]

	print(f"found {eq.sum().item()} pixels with label {lbl.name}")

print("converting to PIL image")

img = TF.to_pil_image(target.cpu(), 'RGB')
img.show()


print("done")