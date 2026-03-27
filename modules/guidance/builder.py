from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from face_parser import FaceParser
from face_vit.evaluate import ViTWrapper

from .controller import GuidanceController


class MinimalPatchForensics:
    """
    Lightweight loader for the patch-forensics discriminator.

    Expected input to compute_patch_loss:
        img: torch.Tensor [B, 3, H, W] in [0, 1]
    """

    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        input_size: Tuple[int, int] = (128, 128),
    ):
        opt = SimpleNamespace()

        # minimal attributes expected by the original patch_forensics repo
        opt.seed = 0
        opt.isTrain = False
        opt.gpu_ids = []
        opt.which_model_netD = "xception_block3"
        opt.init_type = "normal"
        opt.lr = 0.0002
        opt.beta1 = 0.5
        opt.fake_class_id = 1
        opt.epoch = "latest"

        opt.checkpoints_dir = "."
        opt.name = "dummy"
        opt.model = "patch_discriminator"
        opt.dataset_mode = "single"
        opt.verbose = False
        opt.load_iter = 0

        from patch_forensics.models.patch_discriminator_model import PatchDiscriminatorModel

        wrapper = PatchDiscriminatorModel(opt)
        if not hasattr(wrapper, "net_D"):
            raise RuntimeError("PatchDiscriminatorModel does not expose net_D")

        net = wrapper.net_D.to(device)
        net.eval()

        state_dict = torch.load(ckpt_path, map_location=device)
        loaded = False

        if isinstance(state_dict, dict):
            for key in ("state_dict", "model", "net_D", "network"):
                if key in state_dict:
                    try:
                        net.load_state_dict(state_dict[key], strict=False)
                        loaded = True
                        break
                    except Exception:
                        pass

            if not loaded:
                try:
                    net.load_state_dict(state_dict, strict=False)
                    loaded = True
                except Exception:
                    loaded = False

        if not loaded:
            print("[PatchForensics] WARNING: checkpoint partially loaded; continuing.")

        self.device = device
        self.net = net
        self.input_size = input_size

    @torch.no_grad()
    def compute_patch_loss(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: [B, 3, H, W], values in [0, 1]
        returns: scalar tensor
        """
        img = img.to(self.device)
        x = F.interpolate(
            img,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )

        out = self.net(x)

        if out.dim() == 4:
            probs = torch.softmax(out, dim=1)[:, 1, :, :]
            return probs.mean()

        if out.dim() == 2:
            probs = torch.softmax(out, dim=1)[:, 1]
            return probs.mean()

        logits_flat = out.view(out.shape[0], -1)
        return torch.sigmoid(logits_flat).mean()


def build_guidance_controller(
    *,
    device: torch.device,
    target_image_path: str,
    total_steps: int = 50,
    vit_weight_path: str,
    face_parser_ckpt_path: str,
    patch_forensics_ckpt_path: str,
) -> GuidanceController:
    """
    Build all guidance components outside DDIMSampler.

    Args:
        device: torch device
        target_image_path: path to target image used for target identity / segmentation
        total_steps: number of DDIM steps, used only for weight schedules
        vit_weight_path: ArcFace / ViT checkpoint path
        face_parser_ckpt_path: face parser checkpoint path
        patch_forensics_ckpt_path: patch discriminator checkpoint path
    """

    arcface_model = ViTWrapper(
        weight_path=vit_weight_path,
        device=device,
    )

    face_parser = FaceParser(
        face_parser_ckpt_path,
        device,
    )

    patch_detector = MinimalPatchForensics(
        ckpt_path=patch_forensics_ckpt_path,
        device=device,
        input_size=(128, 128),
    )

    target_embed = arcface_model.inference(target_image_path)
    target_seg = face_parser.inference(target_image_path)

    return GuidanceController(
        arcface_model=arcface_model,
        face_parser=face_parser,
        patch_detector=patch_detector,
        target_embed=target_embed,
        target_seg=target_seg,
        total_steps=total_steps,
    )