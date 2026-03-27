import torch
from evaluate import ViTWrapper   # <- make sure vitwrapper.py is in the same folder

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit_guidance = ViTWrapper(
        weight_path="ViT-face_webface_arcface_resnet18_gray_depth_1_head_2_lr_1e-5_fc1024_dropout_0_LFW-20250926T190632Z-1-001/best.pth",
        device=device
    )

    feat1 = vit_guidance.inference("obama.jpg")
    feat2 = vit_guidance.inference("modi.jpg")

    cos_sim = torch.nn.functional.cosine_similarity(feat1, feat2)
    print("Cosine Similarity:", cos_sim.item())