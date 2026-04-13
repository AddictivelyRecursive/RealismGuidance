from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from external.MST_plus_plus.train_code.architecture.MST_Plus_Plus import MST_Plus_Plus

class FrozenMSTPlusPlus(nn.Module):
    """
    Frozen RGB -> HSI reconstructor using official MST++ weights.

    Important:
    - Parameters are frozen.
    - We DO NOT wrap forward in torch.no_grad(), because we still need
      gradients to flow from the HSI loss back to the input image.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        input_size: Tuple[int, int] = (128, 128),
        samplewise_minmax: bool = True,
    ):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.samplewise_minmax = samplewise_minmax

        model = MST_Plus_Plus().to(device)

        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model

    def _normalize_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Official MST++ test code normalizes RGB into [0,1].
        Since our decoded image is already clamped into [0,1],
        this mainly adds optional sample-wise min/max stabilization.
        """
        x = x.clamp(0.0, 1.0)

        if not self.samplewise_minmax:
            return x

        b = x.shape[0]
        flat = x.view(b, -1)
        x_min = flat.min(dim=1)[0].view(b, 1, 1, 1)
        x_max = flat.max(dim=1)[0].view(b, 1, 1, 1)
        return (x - x_min) / (x_max - x_min + 1e-6)

    def forward(self, img_rgb: torch.Tensor) -> torch.Tensor:
        """
        img_rgb: [B, 3, H, W] in [0,1]
        returns: [B, 31, H', W'] HSI cube
        """
        x = img_rgb.to(self.device)
        x = F.interpolate(
            x,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )
        x = self._normalize_rgb(x)
        hsi = self.model(x)
        return hsi


def spectral_curvature_loss(hsi: torch.Tensor) -> torch.Tensor:
    """
    L_curv:
    second difference along spectral dimension.

    hsi: [B, C, H, W]
    """
    if hsi.shape[1] < 3:
        return hsi.new_tensor(0.0)

    d2 = hsi[:, 2:, :, :] - 2.0 * hsi[:, 1:-1, :, :] + hsi[:, :-2, :, :]
    return d2.abs().mean()


def interband_edge_consistency_loss(hsi: torch.Tensor) -> torch.Tensor:
    """
    L_edge:
    consistency of spatial gradients across adjacent spectral bands.

    hsi: [B, C, H, W]
    """
    if hsi.shape[1] < 2:
        return hsi.new_tensor(0.0)

    # Spatial gradients per band
    dx = hsi[:, :, :, 1:] - hsi[:, :, :, :-1]   # [B, C, H, W-1]
    dy = hsi[:, :, 1:, :] - hsi[:, :, :-1, :]   # [B, C, H-1, W]

    # Compare gradients between adjacent spectral bands
    edge_x = dx[:, 1:, :, :] - dx[:, :-1, :, :]
    edge_y = dy[:, 1:, :, :] - dy[:, :-1, :, :]

    return edge_x.abs().mean() + edge_y.abs().mean()


class HSIGuidance:
    """
    Computes training-free HSI artifact energy:
        hsi_loss = curv_coeff * L_curv + edge_coeff * L_edge
    """

    def __init__(
        self,
        mstpp: FrozenMSTPlusPlus,
        curv_coeff: float = 1.0,
        edge_coeff: float = 0.25,
        interval: int = 2,
    ):
        self.mstpp = mstpp
        self.curv_coeff = curv_coeff
        self.edge_coeff = edge_coeff
        self.interval = max(1, int(interval))

    def compute_hsi_loss(
        self,
        img: torch.Tensor,
        step: Optional[int] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = img.new_tensor(0.0)

        # Skip some steps to reduce compute.
        if step is not None and self.interval > 1 and (step % self.interval) != 0:
            return zero, {
                "curv_loss": zero,
                "edge_loss": zero,
                "hsi_loss": zero,
            }

        hsi = self.mstpp(img)  # [B, 31, H, W]

        curv_loss = spectral_curvature_loss(hsi)
        edge_loss = interband_edge_consistency_loss(hsi)

        hsi_loss = self.curv_coeff * curv_loss + self.edge_coeff * edge_loss

        return hsi_loss, {
            "curv_loss": curv_loss,
            "edge_loss": edge_loss,
            "hsi_loss": hsi_loss,
        }