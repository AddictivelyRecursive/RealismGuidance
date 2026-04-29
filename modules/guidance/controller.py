from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch

from utils.losses import d_loss


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) == "1"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _normalize_hsi_triplet(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize one HSI 3-band window into [0, 1] sample-wise.

    Args:
        x: [B, 3, H, W]

    Returns:
        x_norm: [B, 3, H, W]
    """
    x_min = x.amin(dim=(1, 2, 3), keepdim=True)
    x_max = x.amax(dim=(1, 2, 3), keepdim=True)

    return (x - x_min) / (x_max - x_min + eps)


def _make_hsi_triplets(
    hsi: torch.Tensor,
    window_size: int = 3,
    stride: int = 1,
) -> List[torch.Tensor]:
    """
    Args:
        hsi: [B, C, H, W]

    Returns:
        list of [B, 3, H, W] spectral windows
    """
    c = hsi.shape[1]

    window_size = max(1, int(window_size))
    stride = max(1, int(stride))

    if c < window_size:
        return []

    triplets = []

    for start in range(0, c - window_size + 1, stride):
        end = start + window_size
        triplet = hsi[:, start:end, :, :]

        if triplet.shape[1] == 3:
            triplets.append(triplet)

    return triplets


def _aggregate_losses(
    losses: List[torch.Tensor],
    mode: str = "mean",
    topk: int = 5,
) -> torch.Tensor | None:
    if len(losses) == 0:
        return None

    stacked = torch.stack(losses)

    if mode == "sum":
        return stacked.sum()

    if mode == "min":
        return stacked.min()

    if mode == "topk":
        k = min(max(1, int(topk)), stacked.numel())
        values, _ = torch.topk(stacked.flatten(), k=k, largest=False)
        return values.mean()

    return stacked.mean()


class GuidanceController:
    """
    Multi-objective guidance controller for diffusion sampling.

    Supports:
    - RGB ArcFace identity guidance
    - RGB face parser / segmentation guidance
    - HSI artifact guidance
    - experimental HSI-window ArcFace guidance
    - experimental HSI-window segmentation guidance
    """

    def __init__(
        self,
        arcface_model,
        face_parser,
        hsi_guidance,
        target_embed: torch.Tensor,
        target_seg: torch.Tensor,
        total_steps: int = 50,
    ):
        self.arcface_model = arcface_model
        self.face_parser = face_parser
        self.hsi_guidance = hsi_guidance
        self.target_embed = target_embed
        self.target_seg = target_seg
        self.total_steps = total_steps

    def _compute_guidance_weights(self, step: int) -> Tuple[float, float, float]:
        denom = max(self.total_steps - 1, 1)
        t_frac = step / float(denom)

        arc_guidance = 20.0 * (1.0 - t_frac) ** 2 + 3.0

        if t_frac < 0.25:
            seg_guidance = 1.0
        elif t_frac < 0.75:
            seg_guidance = 4.0
        else:
            seg_guidance = 1.0

        if t_frac < 0.20:
            hsi_guidance = 0.0
        elif t_frac < 0.80:
            hsi_guidance = 0.05
        else:
            hsi_guidance = 0.02

        return arc_guidance, seg_guidance, hsi_guidance

    def _compute_hsi_window_losses(
        self,
        hsi_cube: torch.Tensor,
        enable_window_arc: bool,
        enable_window_seg: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = hsi_cube.device

        hsi_window_arc_loss = hsi_cube.new_tensor(0.0)
        hsi_window_seg_loss = hsi_cube.new_tensor(0.0)

        window_size = _env_int("RG_HSI_WINDOW_SIZE", 3)
        stride = _env_int("RG_HSI_WINDOW_STRIDE", 1)
        agg_mode = os.getenv("RG_HSI_WINDOW_AGG", "mean").strip().lower()
        topk = _env_int("RG_HSI_WINDOW_TOPK", 5)

        triplets = _make_hsi_triplets(
            hsi=hsi_cube,
            window_size=window_size,
            stride=stride,
        )

        arc_losses: List[torch.Tensor] = []
        seg_losses: List[torch.Tensor] = []

        for triplet in triplets:
            triplet = _normalize_hsi_triplet(triplet)

            if enable_window_arc:
                x_arc = self.arcface_model.arc_embedding(triplet)
                arc_loss = d_loss(self.target_embed, x_arc, type="cosine")
                arc_losses.append(arc_loss)

            if enable_window_seg:
                x_seg = self.face_parser.segmentation_embedding(triplet)
                seg_loss = d_loss(self.target_seg, x_seg, type="cosine")
                seg_losses.append(seg_loss)

        arc_agg = _aggregate_losses(
            losses=arc_losses,
            mode=agg_mode,
            topk=topk,
        )

        seg_agg = _aggregate_losses(
            losses=seg_losses,
            mode=agg_mode,
            topk=topk,
        )

        if arc_agg is not None:
            hsi_window_arc_loss = arc_agg.to(device)

        if seg_agg is not None:
            hsi_window_seg_loss = seg_agg.to(device)

        return hsi_window_arc_loss, hsi_window_seg_loss

    def compute_losses(
        self,
        x_in: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x_in: decoded image tensor, expected [B, 3, H, W]
            step: current diffusion timestep index

        Returns:
            total_loss: scalar tensor
            loss_dict: individual losses and weights
        """
        device = x_in.device

        total_loss = x_in.new_tensor(0.0)

        arc_loss = x_in.new_tensor(0.0)
        seg_loss = x_in.new_tensor(0.0)

        hsi_loss = x_in.new_tensor(0.0)
        curv_loss = x_in.new_tensor(0.0)
        edge_loss = x_in.new_tensor(0.0)

        hsi_window_arc_loss = x_in.new_tensor(0.0)
        hsi_window_seg_loss = x_in.new_tensor(0.0)

        arc_guidance, seg_guidance, hsi_guidance_weight = self._compute_guidance_weights(step)

        arc_enabled = _env_bool("RG_ENABLE_ARC", "1")
        seg_enabled = _env_bool("RG_ENABLE_SEG", "1")
        hsi_enabled = _env_bool("RG_ENABLE_HSI", "1")

        window_arc_enabled = _env_bool("RG_ENABLE_HSI_WINDOW_ARC", "0")
        window_seg_enabled = _env_bool("RG_ENABLE_HSI_WINDOW_SEG", "0")

        hsi_window_arc_weight = _env_float("RG_HSI_WINDOW_ARC_WEIGHT", 0.05)
        hsi_window_seg_weight = _env_float("RG_HSI_WINDOW_SEG_WEIGHT", 0.05)

        hsi_window_interval = _env_int("RG_HSI_WINDOW_INTERVAL", 1)
        hsi_window_should_run = (step % max(1, hsi_window_interval)) == 0

        if not arc_enabled:
            arc_guidance = 0.0

        if not seg_enabled:
            seg_guidance = 0.0

        if not hsi_enabled:
            hsi_guidance_weight = 0.0

        if not window_arc_enabled:
            hsi_window_arc_weight = 0.0

        if not window_seg_enabled:
            hsi_window_seg_weight = 0.0

        if not hsi_window_should_run:
            hsi_window_arc_weight = 0.0
            hsi_window_seg_weight = 0.0

        needs_hsi_cube = (
            hsi_guidance_weight > 0.0
            or hsi_window_arc_weight > 0.0
            or hsi_window_seg_weight > 0.0
        )

        # ---------------- RGB ArcFace loss ----------------
        if arc_guidance > 0.0:
            x_arc = self.arcface_model.arc_embedding(x_in)
            arc_loss = d_loss(self.target_embed, x_arc, type="cosine")
            total_loss = total_loss + arc_guidance * arc_loss

        # ---------------- RGB Segmentation loss ----------------
        if seg_guidance > 0.0:
            x_seg = self.face_parser.segmentation_embedding(x_in)
            seg_loss = d_loss(self.target_seg, x_seg, type="cosine")
            total_loss = total_loss + seg_guidance * seg_loss

        # ---------------- HSI reconstruction once ----------------
        hsi_cube = None

        if needs_hsi_cube:
            if self.hsi_guidance is None:
                raise RuntimeError("HSI guidance module not initialized")

            x_hsi = x_in

            if x_hsi.min() < 0 or x_hsi.max() > 1:
                x_hsi = (x_hsi + 1.0) / 2.0

            x_hsi = x_hsi.clamp(0.0, 1.0)

            hsi_cube = self.hsi_guidance.reconstruct_hsi(x_hsi)

        # ---------------- Existing HSI artifact loss ----------------
        if hsi_guidance_weight > 0.0:
            hsi_loss, hsi_parts = self.hsi_guidance.compute_hsi_loss_from_cube(hsi_cube)

            curv_loss = hsi_parts["curv_loss"]
            edge_loss = hsi_parts["edge_loss"]

            total_loss = total_loss + hsi_guidance_weight * hsi_loss

        # ---------------- Experimental HSI-window Arc/Seg loss ----------------
        if hsi_window_arc_weight > 0.0 or hsi_window_seg_weight > 0.0:
            hsi_window_arc_loss, hsi_window_seg_loss = self._compute_hsi_window_losses(
                hsi_cube=hsi_cube,
                enable_window_arc=hsi_window_arc_weight > 0.0,
                enable_window_seg=hsi_window_seg_weight > 0.0,
            )

            total_loss = total_loss + hsi_window_arc_weight * hsi_window_arc_loss
            total_loss = total_loss + hsi_window_seg_weight * hsi_window_seg_loss

        loss_dict = {
            "arc_loss": arc_loss,
            "seg_loss": seg_loss,
            "hsi_loss": hsi_loss,
            "curv_loss": curv_loss,
            "edge_loss": edge_loss,
            "hsi_window_arc_loss": hsi_window_arc_loss,
            "hsi_window_seg_loss": hsi_window_seg_loss,
            "arc_weight": torch.tensor(arc_guidance, device=device),
            "seg_weight": torch.tensor(seg_guidance, device=device),
            "hsi_weight": torch.tensor(hsi_guidance_weight, device=device),
            "hsi_window_arc_weight": torch.tensor(hsi_window_arc_weight, device=device),
            "hsi_window_seg_weight": torch.tensor(hsi_window_seg_weight, device=device),
        }

        return total_loss, loss_dict