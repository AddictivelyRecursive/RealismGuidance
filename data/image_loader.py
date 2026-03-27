import numpy as np
import torch
from PIL import Image


def read_image(img_path: str, device, dest_size=(256, 256), grayscale=False):
    if grayscale:
        image = Image.open(img_path).convert("L")
        n_channels = 1
    else:
        image = Image.open(img_path).convert("RGB")
        n_channels = 3

    image = image.resize(dest_size, Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0

    if n_channels == 1:
        image = image[None, None, :, :]
    else:
        image = image[None].transpose(0, 3, 1, 2)

    image = torch.from_numpy(image).to(device)
    image = image * 2.0 - 1.0
    return image