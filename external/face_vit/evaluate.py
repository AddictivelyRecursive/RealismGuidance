import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from face_vit.models.vit_model_face import ViT_face_model
from face_vit.models.resnet import resnet_face18


class ViTWrapper:
    """
    A wrapper around the Hybrid H2L ViT-Face model to replace CNN-based identity guidance.
    Designed to mimic the ArcLoss API: .inference() and .arc_embedding()
    """

    def __init__(self, weight_path, device):
        self.device = device

        # ---- Initialize model (same params as your working code) ----
        self.model = ViT_face_model(
            loss_type='ArcFace',
            GPU_ID=['0'],
            num_class=10575,
            use_cls=False,
            use_face_loss=True,
            no_face_model=False,
            image_size=128,
            patch_size=8,
            ac_patch_size=12,
            pad=4,
            dim=512,
            depth=1,
            heads=2,
            mlp_dim=2048,
            dropout=0.0,
            emb_dropout=0.1,
            out_dim=1024,
            singleMLP=False,
            remove_sep=False,
            remove_pos=False
        )

        # Attach ResNet face feature extractor (same as pretrained config)
        facemodel = resnet_face18(False, use_reduce_pool=False, grayscale=True)
        self.model.face_model = facemodel

        # Load pretrained weights
        state_dict = torch.load(weight_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        # Freeze params but allow gradients w.r.t. input (for diffusion guidance)
        for p in self.model.parameters():
            p.requires_grad = False

        # Define preprocessing
        self.transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    # -----------------------------------------------------------------
    def preprocess(self, image_path):
        """Load image from path and preprocess for ViT input."""
        image = Image.open(image_path).convert('L')
        image = self.transform(image).unsqueeze(0)  # (1, 1, 128, 128)
        return image.to(self.device)

    # -----------------------------------------------------------------
    @torch.no_grad()
    def inference(self, img_path):
        """
        ArcFace-style inference using image path.
        Returns the L2-normalized ViT embedding.
        """
        x = self.preprocess(img_path)
        x = torch.cat([x,x],dim=0)  # ViT model quirk: needs batch size >=2
        embedding, _ = self.model(x, fea=True)
        embedding = F.normalize(embedding, p=2, dim=1) # take only the first
        return embedding  # shape: [1, 1024]

    # -----------------------------------------------------------------
    # def arc_embedding(self, img_tensor):
    #     """
    #     Direct embedding from already-preprocessed image tensor.
    #     Keeps gradient flow active for guidance loss.
    #     Expects tensor in range [-1,1], shape [B, 1, H, W].
    #     """
    #     if img_tensor.ndim == 3:
    #         img_tensor = img_tensor.unsqueeze(0)
    #     img_tensor = F.interpolate(img_tensor, size=(128, 128), mode='bilinear', align_corners=False)
    #     embedding, _ = self.model(img_tensor.to(self.device), fea=True)
    #     embedding = F.normalize(embedding, p=2, dim=1)
    #     return embedding
    
    def arc_embedding(self, img_tensor):
        """
        Direct embedding from already-preprocessed image tensor.
        Keeps gradient flow active for guidance loss.
        Expects tensor in range [-1,1], shape [B, 1, H, W] or [B, 3, H, W].
        If RGB input is given, it's converted to grayscale before embedding.
        """
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)  # [C,H,W] → [1,C,H,W]
            
            # If RGB, convert to grayscale using luminance-preserving weights
        if img_tensor.shape[1] == 3:
            r, g, b = img_tensor[:, 0:1, :, :], img_tensor[:, 1:2, :, :], img_tensor[:, 2:3, :, :]
            img_tensor = 0.2989 * r + 0.5870 * g + 0.1140 * b  # grayscale conversion

        # Resize to model’s expected input size
        img_tensor = F.interpolate(img_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        
        img_tensor = torch.cat([img_tensor,img_tensor],dim=0)
        # Get embedding
        embedding, _ = self.model(img_tensor.to(self.device), fea=True)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding