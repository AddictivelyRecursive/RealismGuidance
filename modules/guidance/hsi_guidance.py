from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from external.MST_plus_plus.train_code.architecture.MST_Plus_Plus import MST_Plus_Plus


class FrozenMSTPlusPlus(nn.Module):
    """
    Frozen RGB -> HSI reconstructor using official MST++ weights.

    Parameters are frozen, but the forward pass remains differentiable.
    This allows gradients from HSI-space losses to flow back to the RGB image.
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
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model

    def _normalize_rgb(self, x: torch.Tensor) -> torch.Tensor:
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
        Args:
            img_rgb: [B, 3, H, W], expected in [0, 1]

        Returns:
            hsi: [B, 31, H_hsi, W_hsi]
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
    Curvature loss over adjacent spectral bands.

    Args:
        hsi: [B, C, H, W]
    """
    if hsi.shape[1] < 3:
        return hsi.new_tensor(0.0)

    d2 = hsi[:, 2:, :, :] - 2.0 * hsi[:, 1:-1, :, :] + hsi[:, :-2, :, :]

    return d2.abs().mean()


def interband_edge_consistency_loss(hsi: torch.Tensor) -> torch.Tensor:
    """
    Edge consistency loss over adjacent spectral bands.

    Args:
        hsi: [B, C, H, W]
    """
    if hsi.shape[1] < 2:
        return hsi.new_tensor(0.0)

    dx = hsi[:, :, :, 1:] - hsi[:, :, :, :-1]
    dy = hsi[:, :, 1:, :] - hsi[:, :, :-1, :]

    edge_x = dx[:, 1:, :, :] - dx[:, :-1, :, :]
    edge_y = dy[:, 1:, :, :] - dy[:, :-1, :, :]

    return edge_x.abs().mean() + edge_y.abs().mean()


class HSIGuidance:
    """
    HSI guidance module.

    Provides:
    1. MST++ RGB-to-HSI reconstruction.
    2. Existing HSI artifact loss.
    3. HSI cube access for experimental spectral-window guidance.
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

    def should_run(self, step: Optional[int] = None) -> bool:
        if step is None:
            return True

        if self.interval <= 1:
            return True

        return (step % self.interval) == 0

    def reconstruct_hsi(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: [B, 3, H, W], expected in [0, 1]

        Returns:
            hsi: [B, 31, H_hsi, W_hsi]
        """
        return self.mstpp(img)

    def compute_hsi_loss_from_cube(
        self,
        hsi: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        curv_loss = spectral_curvature_loss(hsi)
        edge_loss = interband_edge_consistency_loss(hsi)

        hsi_loss = self.curv_coeff * curv_loss + self.edge_coeff * edge_loss

        return hsi_loss, {
            "curv_loss": curv_loss,
            "edge_loss": edge_loss,
            "hsi_loss": hsi_loss,
        }

    def compute_hsi_loss(
        self,
        img: torch.Tensor,
        step: Optional[int] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = img.new_tensor(0.0)

        if not self.should_run(step):
            return zero, {
                "curv_loss": zero,
                "edge_loss": zero,
                "hsi_loss": zero,
            }

        hsi = self.reconstruct_hsi(img)

        return self.compute_hsi_loss_from_cube(hsi)