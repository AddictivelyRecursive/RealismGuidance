from __future__ import annotations

from typing import Dict, Tuple

import torch

from utils.losses import d_loss


class GuidanceController:
    """
    Multi-objective guidance controller for diffusion sampling.

    Preserves the original logic:
    - ArcFace identity guidance
    - Face parser / segmentation guidance
    - Patch-forensics guidance
    - timestep-dependent scalar schedules
    """

    def __init__(
        self,
        arcface_model,
        face_parser,
        patch_detector,
        target_embed: torch.Tensor,
        target_seg: torch.Tensor,
        total_steps: int = 50,
    ):
        self.arcface_model = arcface_model
        self.face_parser = face_parser
        self.patch_detector = patch_detector
        self.target_embed = target_embed
        self.target_seg = target_seg
        self.total_steps = total_steps

    def _compute_guidance_weights(self, step: int) -> Tuple[float, float, float]:
        """
        Same scheduling logic as the original cond_fn.
        Original code normalized by 49.0 assuming 50 steps.
        Here we generalize but preserve equivalent behavior.
        """
        denom = max(self.total_steps - 1, 1)
        t_frac = step / float(denom)

        # ARC (IDENTITY)
        arc_guidance = 20 * (1 - t_frac) ** 2 + 3

        # SEG (POSE / GEOMETRY)
        if t_frac < 0.25:
            seg_guidance = 1.0
        elif t_frac < 0.75:
            seg_guidance = 4.0
        else:
            seg_guidance = 1.0

        # PATCH (DETAIL / REALISM)
        patch_guidance = 0.12 * (1 - t_frac)

        return arc_guidance, seg_guidance, patch_guidance

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
            loss_dict: contains individual losses and weights
        """
        device = x_in.device

        total_loss = torch.tensor(0.0, device=device)
        arc_loss = torch.tensor(0.0, device=device)
        seg_loss = torch.tensor(0.0, device=device)
        patch_loss = torch.tensor(0.0, device=device)

        arc_guidance, seg_guidance, patch_guidance = self._compute_guidance_weights(step)

        # ---------------- ArcFace loss ----------------
        x_arc = self.arcface_model.arc_embedding(x_in)
        arc_loss = d_loss(self.target_embed, x_arc, type="cosine")
        total_loss = total_loss + arc_loss * arc_guidance

        # ---------------- Segmentation loss ----------------
        x_seg = self.face_parser.segmentation_embedding(x_in)
        seg_loss = d_loss(self.target_seg, x_seg, type="cosine")
        total_loss = total_loss + seg_loss * seg_guidance

        # ---------------- Patch-forensics loss ----------------
        if self.patch_detector is None:
            raise RuntimeError("Patch-Forensics detector not initialized")

        x_pf = x_in
        if x_pf.min() < 0 or x_pf.max() > 1:
            x_pf = (x_pf + 1) / 2
        x_pf = x_pf.clamp(0, 1)

        patch_loss = self.patch_detector.compute_patch_loss(x_pf)
        total_loss = total_loss + patch_guidance * patch_loss

        loss_dict = {
            "arc_loss": arc_loss,
            "seg_loss": seg_loss,
            "patch_loss": patch_loss,
            "arc_weight": torch.tensor(arc_guidance, device=device),
            "seg_weight": torch.tensor(seg_guidance, device=device),
            "patch_weight": torch.tensor(patch_guidance, device=device),
        }

        return total_loss, loss_dict