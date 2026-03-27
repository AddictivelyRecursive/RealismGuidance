import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation


def read_mask(
    mask_path: str,
    device,
    dilation_iterations: int = 0,
    dest_size=(32, 32),
    img_size=(256, 256),
):
    org_mask = Image.open(mask_path).convert("L")
    mask = org_mask.resize(dest_size, Image.NEAREST)
    mask = np.array(mask) / 255

    masks_array = []
    for i in reversed(range(dilation_iterations)):
        k_size = 3 + 2 * i
        masks_array.append(binary_dilation(mask, structure=np.ones((k_size, k_size))))
    masks_array.append(mask)

    masks_array = np.array(masks_array).astype(np.float32)
    masks_array = masks_array[:, np.newaxis, :]
    masks_array = torch.from_numpy(masks_array).to(device)

    org_mask = org_mask.resize(img_size, Image.LANCZOS)
    org_mask = np.array(org_mask).astype(np.float32) / 255.0
    org_mask = org_mask[None, None]
    org_mask[org_mask < 0.5] = 0
    org_mask[org_mask >= 0.5] = 1
    org_mask = torch.from_numpy(org_mask).to(device)

    return masks_array, org_mask