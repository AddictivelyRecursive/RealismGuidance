"""DDIM sampler with externalized multi-guidance controller.

This preserves the core logic of the original sampler:
- DDIM schedule creation
- latent initialization from init_image or noise
- latent-level blending with mask
- guidance through cond_fn
- optional pixel-level blending with org_mask
- returning intermediates and final loss tuple

The main cleanup is architectural:
- ArcFace / FaceParser / PatchForensics are NOT initialized here
- plotting is removed from the sampler
- hardcoded checkpoint paths are removed from the sampler
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
)


class DDIMSampler:
    def __init__(
        self,
        model,
        guidance_controller=None,
        schedule: str = "linear",
    ):
        super().__init__()
        self.model = model
        self.guidance_controller = guidance_controller
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name: str, attr: torch.Tensor) -> None:
        if isinstance(attr, torch.Tensor) and attr.device != self.model.device:
            attr = attr.to(self.model.device)
        setattr(self, name, attr)

    def make_schedule(
        self,
        ddim_num_steps: int,
        ddim_discretize: str = "uniform",
        ddim_eta: float = 0.0,
        verbose: bool = True,
    ) -> None:
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )

        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, (
            "alphas have to be defined for each timestep"
        )

        def to_torch(x):
            return x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod",
            to_torch(np.log(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),
        )

        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))

        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer(
            "ddim_sigmas_for_original_num_steps",
            sigmas_for_original_sampling_steps,
        )

    @torch.no_grad()
    def sample(
        self,
        S: int,
        batch_size: int,
        shape: Tuple[int, int, int],
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0: bool = False,
        eta: float = 0.0,
        mask=None,
        org_mask=None,
        x0=None,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose: bool = True,
        x_T=None,
        log_every_t: int = 100,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning=None,
        skip_steps: int = 0,
        init_image=None,
        percentage_of_pixel_blending: float = 0.0,
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
                    )

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)

        c, h, w = shape
        size = (batch_size, c, h, w)
        print(f"Data shape for DDIM sampling is {size}, eta {eta}")

        samples, intermediates, loss = self.ddim_sampling(
            cond=conditioning,
            shape=size,
            x_T=x_T,
            ddim_use_original_steps=False,
            callback=callback,
            timesteps=None,
            quantize_denoised=quantize_x0,
            mask=mask,
            org_mask=org_mask,
            x0=x0,
            img_callback=img_callback,
            log_every_t=log_every_t,
            temperature=temperature,
            noise_dropout=noise_dropout,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            skip_steps=skip_steps,
            init_image=init_image,
            percentage_of_pixel_blending=percentage_of_pixel_blending,
        )
        return samples, intermediates, loss

    @torch.no_grad()
    def ddim_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps: bool = False,
        callback=None,
        timesteps=None,
        quantize_denoised: bool = False,
        mask=None,
        org_mask=None,
        x0=None,
        img_callback=None,
        log_every_t: int = 100,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning=None,
        skip_steps: int = 0,
        init_image=None,
        percentage_of_pixel_blending: float = 0.0,
    ):
        device = self.model.betas.device
        batch_size = shape[0]

        if timesteps is None:
            timesteps = (
                self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
            )
        elif not ddim_use_original_steps:
            subset_end = (
                int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0])
                - 1
            )
            timesteps = self.ddim_timesteps[:subset_end]

        if skip_steps != 0:
            timesteps = timesteps[:-skip_steps]

        time_range = (
            reversed(range(0, timesteps))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]

        print(f"Running DDIM Sampling with {total_steps} timesteps")

        if init_image is not None:
            assert x0 is None and x_T is None, (
                "Trying to infer x0 and x_t from init_image, but x0/x_T were already provided"
            )

            encoder_posterior = self.model.encode_first_stage(init_image)
            x0 = self.model.get_first_stage_encoding(encoder_posterior)
            last_ts = torch.full((1,), time_range[0], device=device, dtype=torch.long)
            x_T = torch.cat([self.model.q_sample(x0, last_ts) for _ in range(batch_size)])
            img = x_T
        elif x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)

        cutoff_point = int(len(time_range) * (1 - percentage_of_pixel_blending))

        last_loss_tuple = (
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
        )

        for i, step in enumerate(progress_bar := iterator):
            index = total_steps - i - 1
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # latent-level blending
            if mask is not None and i < cutoff_point:
                num_masks = mask.shape[0]
                masks_interval = len(time_range) // num_masks + 1
                curr_mask = mask[i // masks_interval].unsqueeze(0)

                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * (1 - curr_mask) + curr_mask * img

            img, pred_x0, loss_tuple = self.p_sample_ddim(
                x=img,
                c=cond,
                t=ts,
                index=index,
                repeat_noise=False,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                x0=x0,
            )
            last_loss_tuple = loss_tuple

            arc_loss, seg_loss, patch_loss = loss_tuple
            progress_bar.set_description(
                f"Arc Loss: {arc_loss}, Seg Loss: {seg_loss}, Patch Loss: {patch_loss}"
            )

            # pixel-level blending
            if org_mask is not None and i >= cutoff_point:
                foreground_pixels = self.model.decode_first_stage(pred_x0)
                background_pixels = init_image
                pixel_blended = foreground_pixels * org_mask + background_pixels * (1 - org_mask)

                img_x0 = self.model.get_first_stage_encoding(
                    self.model.encode_first_stage(pixel_blended)
                )
                img = self.model.q_sample(img_x0, ts)

            if callback is not None:
                callback(i)
            if img_callback is not None:
                img_callback(pred_x0, i)

            intermediates["x_inter"].append(img)
            intermediates["pred_x0"].append(pred_x0)

        return img, intermediates, last_loss_tuple[-1]

    def cond_fn(
        self,
        x,
        t,
        c,
        a_t,
        sqrt_one_minus_at,
        quantize_denoised: bool = False,
    ):
        if self.guidance_controller is None:
            raise RuntimeError("guidance_controller must be provided for conditional guidance")

        with torch.enable_grad():
            x = x.detach().requires_grad_()
            e_t = self.model.apply_model(x, t, c)

            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

            # preserve original logic exactly
            fac = self.sqrt_one_minus_alphas_cumprod[t[0].item()]
            x_in = pred_x0 * fac + x * (1 - fac)
            x_in = self.model.decode_first_stage(x_in)

            total_loss, loss_dict = self.guidance_controller.compute_losses(
                x_in=x_in,
                step=t[0].item(),
            )

        grad = -torch.autograd.grad(total_loss, x)[0]
        e_t = e_t - sqrt_one_minus_at * grad

        return e_t, (
            loss_dict["arc_loss"],
            loss_dict["seg_loss"],
            loss_dict["patch_loss"],
        )

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index: int,
        repeat_noise: bool = False,
        use_original_steps: bool = False,
        quantize_denoised: bool = False,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning=None,
        x0=None,
    ):
        batch_size, *_, device = *x.shape, x.device
        conditional_guidance = True

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )

        a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (batch_size, 1, 1, 1),
            sqrt_one_minus_alphas[index],
            device=device,
        )

        if conditional_guidance:
            e_t, loss = self.cond_fn(
                x=x,
                t=t,
                c=c,
                a_t=a_t,
                sqrt_one_minus_at=sqrt_one_minus_at,
                quantize_denoised=quantize_denoised,
            )
        else:
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            loss = (
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
            )

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = noise_like(x.shape, device, repeat_noise) * temperature

        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise * sigma_t

        if conditional_guidance:
            return x_prev, pred_x0, loss
        return x_prev, pred_x0