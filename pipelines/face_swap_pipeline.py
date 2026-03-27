from __future__ import annotations

from typing import Dict, Optional

import time
import torch


class FaceSwapPipeline:
    """
    Clean pipeline for face swapping using diffusion + guidance.

    Responsibilities:
    - Prepare latent shape
    - Call DDIM sampler
    - Decode outputs
    - Return structured logs

    Does NOT:
    - save images
    - handle CSV
    - do plotting
    """

    def __init__(self, model, sampler):
        self.model = model
        self.sampler = sampler

    @torch.no_grad()
    def sample(
        self,
        *,
        batch_size: int,
        steps: int,
        eta: float,
        init_image: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        org_mask: Optional[torch.Tensor],
        target_image_path: Optional[str],
        run_tests: bool = False,
    ) -> Dict:

        log: Dict = {}

        # -------------------------------
        # Build latent shape
        # -------------------------------
        shape = [
            batch_size,
            self.model.model.diffusion_model.in_channels,
            self.model.model.diffusion_model.image_size,
            self.model.model.diffusion_model.image_size,
        ]

        # -------------------------------
        # Sampling
        # -------------------------------
        with self.model.ema_scope("Sampling"):
            t0 = time.time()

            samples, intermediates, loss = self.sampler.sample(
                S=steps,
                batch_size=batch_size,
                shape=shape[1:],  # sampler expects (C,H,W)
                eta=eta,
                mask=mask,
                org_mask=org_mask,
                init_image=init_image,
            )

            t1 = time.time()

        # -------------------------------
        # Decode final outputs
        # -------------------------------
        x_sample = self.model.decode_first_stage(samples)

        # -------------------------------
        # Optional: process intermediates
        # -------------------------------
        if not run_tests:
            decoded_intermediates = self._decode_intermediates(intermediates)

            log["intermediates"] = decoded_intermediates

        # -------------------------------
        # Logging
        # -------------------------------
        log["sample"] = x_sample
        log["time"] = t1 - t0
        log["throughput"] = samples.shape[0] / (t1 - t0)
        log["cos_dist"] = loss

        # NOTE: keep paths, not PIL objects
        log["target_image_path"] = target_image_path

        return log

    def _decode_intermediates(self, intermediates):
        """
        Decode predicted x0 at each timestep.

        Equivalent to original:
        pred_x0_intermediates = torch.stack(...)
        then decode batch-wise
        """

        pred_x0 = torch.stack(intermediates["pred_x0"], dim=0)

        # take first sample only (same as original code)
        pred_x0 = pred_x0[:, 0, ...]

        decoded = []

        for i in range(len(pred_x0)):
            batch = pred_x0[i : i + 1]
            decoded_img = self.model.decode_first_stage(batch)
            decoded.append(decoded_img)

        return decoded