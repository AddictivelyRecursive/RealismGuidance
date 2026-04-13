from __future__ import annotations

import os
from typing import Dict, Tuple

import torch

from utils.losses import d_loss


class GuidanceController:
    """
    Multi-objective guidance controller for diffusion sampling.

    Preserves the original logic:
    - ArcFace identity guidance
    - Face parser / segmentation guidance
    - HSI artifact guidance
    - timestep-dependent scalar schedules
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
        """
        Keep ARC and SEG schedules close to current behavior.
        Use a safer HSI schedule:
        - off early
        - moderate in middle
        - small at end
        """
        denom = max(self.total_steps - 1, 1)
        t_frac = step / float(denom)

        # ARC (IDENTITY)
        arc_guidance = 20.0 * (1.0 - t_frac) ** 2 + 3.0

        # SEG (POSE / GEOMETRY)
        if t_frac < 0.25:
            seg_guidance = 1.0
        elif t_frac < 0.75:
            seg_guidance = 4.0
        else:
            seg_guidance = 1.0

        # HSI (REALISM)
        if t_frac < 0.20:
            hsi_guidance = 0.0
        elif t_frac < 0.80:
            hsi_guidance = 0.05
        else:
            hsi_guidance = 0.02

        return arc_guidance, seg_guidance, hsi_guidance

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

        total_loss = x_in.new_tensor(0.0)
        arc_loss = x_in.new_tensor(0.0)
        seg_loss = x_in.new_tensor(0.0)
        hsi_loss = x_in.new_tensor(0.0)
        curv_loss = x_in.new_tensor(0.0)
        edge_loss = x_in.new_tensor(0.0)

        arc_guidance, seg_guidance, hsi_guidance_weight = self._compute_guidance_weights(step)

        arc_enabled = os.getenv("RG_ENABLE_ARC", "1") == "1"
        seg_enabled = os.getenv("RG_ENABLE_SEG", "1") == "1"
        hsi_enabled = os.getenv("RG_ENABLE_HSI", "1") == "1"

        if not arc_enabled:
            arc_guidance = 0.0
        if not seg_enabled:
            seg_guidance = 0.0
        if not hsi_enabled:
            hsi_guidance_weight = 0.0

        # ---------------- ArcFace loss ----------------
        if arc_guidance > 0.0:
            x_arc = self.arcface_model.arc_embedding(x_in)
            arc_loss = d_loss(self.target_embed, x_arc, type="cosine")
            total_loss = total_loss + arc_guidance * arc_loss

        # ---------------- Segmentation loss ----------------
        if seg_guidance > 0.0:
            x_seg = self.face_parser.segmentation_embedding(x_in)
            seg_loss = d_loss(self.target_seg, x_seg, type="cosine")
            total_loss = total_loss + seg_guidance * seg_loss

        # ---------------- HSI guidance loss ----------------
        if hsi_guidance_weight > 0.0:
            if self.hsi_guidance is None:
                raise RuntimeError("HSI guidance module not initialized")

            x_hsi = x_in
            if x_hsi.min() < 0 or x_hsi.max() > 1:
                x_hsi = (x_hsi + 1.0) / 2.0
            x_hsi = x_hsi.clamp(0.0, 1.0)

            hsi_loss, hsi_parts = self.hsi_guidance.compute_hsi_loss(
                img=x_hsi,
                step=step,
            )
            curv_loss = hsi_parts["curv_loss"]
            edge_loss = hsi_parts["edge_loss"]

            total_loss = total_loss + hsi_guidance_weight * hsi_loss

        loss_dict = {
            "arc_loss": arc_loss,
            "seg_loss": seg_loss,
            "hsi_loss": hsi_loss,
            "curv_loss": curv_loss,
            "edge_loss": edge_loss,
            "arc_weight": torch.tensor(arc_guidance, device=device),
            "seg_weight": torch.tensor(seg_guidance, device=device),
            "hsi_weight": torch.tensor(hsi_guidance_weight, device=device),
        }
        return total_loss, loss_dict
    
# Older version without HSI parts:

# class GuidanceController:
#     """
#     Multi-objective guidance controller for diffusion sampling.

#     Preserves the original logic:
#     - ArcFace identity guidance
#     - Face parser / segmentation guidance
#     - Patch-forensics guidance
#     - timestep-dependent scalar schedules
#     """

#     def __init__(
#         self,
#         arcface_model,
#         face_parser,
#         patch_detector,
#         target_embed: torch.Tensor,
#         target_seg: torch.Tensor,
#         total_steps: int = 50,
#     ):
#         self.arcface_model = arcface_model
#         self.face_parser = face_parser
#         self.patch_detector = patch_detector
#         self.target_embed = target_embed
#         self.target_seg = target_seg
#         self.total_steps = total_steps

#     def _compute_guidance_weights(self, step: int) -> Tuple[float, float, float]:
#         """
#         Same scheduling logic as the original cond_fn.
#         Original code normalized by 49.0 assuming 50 steps.
#         Here we generalize but preserve equivalent behavior.
#         """
#         denom = max(self.total_steps - 1, 1)
#         t_frac = step / float(denom)

#         # ARC (IDENTITY)
#         arc_guidance = 20 * (1 - t_frac) ** 2 + 3

#         # SEG (POSE / GEOMETRY)
#         if t_frac < 0.25:
#             seg_guidance = 1.0
#         elif t_frac < 0.75:
#             seg_guidance = 4.0
#         else:
#             seg_guidance = 1.0

#         # PATCH (DETAIL / REALISM)
#         patch_guidance = 0.12 * (1 - t_frac)

#         return arc_guidance, seg_guidance, patch_guidance
    
#     def compute_losses(
#         self,
#         x_in: torch.Tensor,
#         step: int,
#     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         """
#         Args:
#             x_in: decoded image tensor, expected [B, 3, H, W]
#             step: current diffusion timestep index

#         Returns:
#             total_loss: scalar tensor
#             loss_dict: contains individual losses and weights
#         """
#         device = x_in.device

#         total_loss = torch.tensor(0.0, device=device)
#         arc_loss = torch.tensor(0.0, device=device)
#         seg_loss = torch.tensor(0.0, device=device)
#         patch_loss = torch.tensor(0.0, device=device)

#         arc_guidance, seg_guidance, patch_guidance = self._compute_guidance_weights(step)
        
#         arc_enabled = os.getenv("RG_ENABLE_ARC", "1") == "1"
#         seg_enabled = os.getenv("RG_ENABLE_SEG", "1") == "1"
#         patch_enabled = os.getenv("RG_ENABLE_PATCH", "1") == "1"

#         if not arc_enabled:
#             arc_guidance = 0.0
#         if not seg_enabled:
#             seg_guidance = 0.0
#         if not patch_enabled:
#             patch_guidance = 0.0

#         # ---------------- ArcFace loss ----------------
#         x_arc = self.arcface_model.arc_embedding(x_in)
#         arc_loss = d_loss(self.target_embed, x_arc, type="cosine")
#         total_loss = total_loss + arc_loss * arc_guidance

#         # ---------------- Segmentation loss ----------------
#         x_seg = self.face_parser.segmentation_embedding(x_in)
#         seg_loss = d_loss(self.target_seg, x_seg, type="cosine")
#         total_loss = total_loss + seg_loss * seg_guidance

#         # ---------------- Patch-forensics loss ----------------
#         if self.patch_detector is None:
#             raise RuntimeError("Patch-Forensics detector not initialized")

#         x_pf = x_in
#         if x_pf.min() < 0 or x_pf.max() > 1:
#             x_pf = (x_pf + 1) / 2
#         x_pf = x_pf.clamp(0, 1)

#         patch_loss = self.patch_detector.compute_patch_loss(x_pf)
#         total_loss = total_loss + patch_guidance * patch_loss

#         loss_dict = {
#             "arc_loss": arc_loss,
#             "seg_loss": seg_loss,
#             "patch_loss": patch_loss,
#             "arc_weight": torch.tensor(arc_guidance, device=device),
#             "seg_weight": torch.tensor(seg_guidance, device=device),
#             "patch_weight": torch.tensor(patch_guidance, device=device),
#         }

#         return total_loss, loss_dict