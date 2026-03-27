import torch
from ldm.util import instantiate_from_config


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    if torch.cuda.is_available():
        model.cuda()
        print("Model moved to GPU.")
    else:
        model.cpu()
        print("GPU not available. Model will remain on CPU.")
    model.eval()
    return model


def load_model(config, ckpt, gpu=True, eval_mode=True):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd.get("global_step", None)
    else:
        pl_sd = {"state_dict": None}
        global_step = None

    model = load_model_from_config(config.model, pl_sd["state_dict"])
    return model, global_step