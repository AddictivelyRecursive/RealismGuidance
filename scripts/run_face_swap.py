import sys
import os

ROOT = os.getcwd()

sys.path.extend([
    os.path.join(ROOT, "external/ldm_repo"),
    os.path.join(ROOT, "external/face_vit"),
    os.path.join(ROOT, "external/face_parser"),
    os.path.join(ROOT, "external/patch_forensics"),
])
from __future__ import annotations

import argparse
import csv
import datetime
import glob
from typing import Optional

import yaml
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from data.image_loader import read_image
from data.mask_loader import read_mask
from models.diffusion.ddim_sampler import DDIMSampler
from models.diffusion.model_loader import load_model
from modules.guidance.builder import build_guidance_controller
from pipelines.face_swap_pipeline import FaceSwapPipeline
from utils.image_utils import custom_to_np
from visualization.save_outputs import save_logs


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        required=True,
        help="Path to checkpoint file or logdir containing model checkpoint/config.",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=1,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        default=1.0,
        help="Eta for DDIM sampling.",
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        action="store_true",
        default=False,
        help="Use vanilla sampling instead of DDIM. Currently not supported in clean pipeline.",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="none",
        help="Optional override for output logdir root.",
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        default=50,
        help="Number of DDIM steps.",
    )
    parser.add_argument(
        "--run_tests",
        action="store_true",
        default=False,
        help="Run batch evaluation from CSV of source-target-mask triplets.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for sampling.",
    )
    parser.add_argument(
        "--init_image",
        type=str,
        default=None,
        help="Path to source image.",
    )
    parser.add_argument(
        "--target_image",
        type=str,
        default=None,
        help="Path to target image.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to face mask.",
    )

    # Guidance checkpoint paths
    parser.add_argument(
        "--vit_weight_path",
        type=str,
        required=True,
        help="Path to ViT / ArcFace checkpoint.",
    )
    parser.add_argument(
        "--face_parser_ckpt_path",
        type=str,
        required=True,
        help="Path to face parser checkpoint.",
    )
    parser.add_argument(
        "--patch_forensics_ckpt_path",
        type=str,
        required=True,
        help="Path to patch-forensics discriminator checkpoint.",
    )

    # Batch-test dataset paths
    parser.add_argument(
        "--test_base_dir",
        type=str,
        default=None,
        help="Base directory for batch test mode.",
    )
    parser.add_argument(
        "--test_pairs_csv",
        type=str,
        default=None,
        help="CSV containing source_image, target_image, merged_mask columns for batch test mode.",
    )

    return parser


def resolve_resume_paths(resume_path: str):
    """
    Resolve checkpoint path and corresponding logdir/config location.
    Preserves original behavior from old script.
    """
    if not os.path.exists(resume_path):
        raise ValueError(f"Cannot find {resume_path}")

    if os.path.isfile(resume_path):
        try:
            logdir = "/".join(resume_path.split("/")[:-1])
        except ValueError:
            parts = resume_path.split("/")
            logdir = "/".join(parts[:-2])
        ckpt = resume_path
    else:
        if not os.path.isdir(resume_path):
            raise ValueError(f"{resume_path} is neither a file nor a directory")
        logdir = resume_path.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    config_paths = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    if not config_paths:
        raise FileNotFoundError(f"No config.yaml found in {logdir}")

    return ckpt, logdir, config_paths


def build_output_dirs(base_logdir: str, global_step: Optional[int], timestamp: str):
    global_step_str = "00000000" if global_step is None else f"{global_step:08d}"
    run_logdir = os.path.join(base_logdir, "samples", global_step_str, timestamp)
    imglogdir = os.path.join(run_logdir, "img")
    numpylogdir = os.path.join(run_logdir, "numpy")

    os.makedirs(imglogdir, exist_ok=True)
    os.makedirs(numpylogdir, exist_ok=True)

    return run_logdir, imglogdir, numpylogdir


def write_sampling_config(logdir: str, opt) -> None:
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    with open(sampling_file, "w") as f:
        yaml.dump(vars(opt), f, default_flow_style=False)


def build_pipeline(model, opt, target_image_path: str):
    guidance_controller = build_guidance_controller(
        device=model.device,
        target_image_path=target_image_path,
        total_steps=opt.custom_steps,
        vit_weight_path=opt.vit_weight_path,
        face_parser_ckpt_path=opt.face_parser_ckpt_path,
        patch_forensics_ckpt_path=opt.patch_forensics_ckpt_path,
    )

    sampler = DDIMSampler(
        model=model,
        guidance_controller=guidance_controller,
    )

    pipeline = FaceSwapPipeline(
        model=model,
        sampler=sampler,
    )
    return pipeline


def run_single_sample(
    *,
    model,
    pipeline,
    imglogdir: str,
    numpylogdir: str,
    opt,
    init_image,
    mask,
    org_mask,
    target_image_path: str,
    init_image_path: Optional[str] = None,
    csv_file: Optional[str] = None,
):
    logs = pipeline.sample(
        batch_size=opt.batch_size,
        steps=opt.custom_steps,
        eta=opt.eta,
        init_image=init_image,
        mask=mask,
        org_mask=org_mask,
        target_image_path=target_image_path,
        run_tests=False,
    )

    # add source image path for saving module
    logs["init_image_path"] = init_image_path
    logs["target_image_path"] = target_image_path

    n_saved = save_logs(
        logs=logs,
        path=imglogdir,
        n_saved=0,
        key="sample",
        csv_file=csv_file,
        init_image_path=init_image_path,
        target_image_path=target_image_path,
    )

    all_img = custom_to_np(logs["sample"])
    all_img = all_img[: opt.n_samples]
    shape_str = "x".join([str(x) for x in all_img.shape])
    nppath = os.path.join(numpylogdir, f"{shape_str}-samples.npz")
    np.savez(nppath, all_img)

    print(f"Saved {n_saved} sample(s) to {imglogdir}")
    print(f"Final guidance metric (cos_dist): {logs['cos_dist']}")
    print(f"Throughput: {logs['throughput']:.4f} samples/sec")

    return logs["cos_dist"]


def run_batch_tests(
    *,
    model,
    opt,
    run_logdir: str,
):
    if opt.test_base_dir is None or opt.test_pairs_csv is None:
        raise ValueError(
            "Batch test mode requires --test_base_dir and --test_pairs_csv."
        )

    base_dir = opt.test_base_dir
    csv_path = opt.test_pairs_csv

    source_dir = os.path.join(base_dir, "data/source")
    target_dir = os.path.join(base_dir, "data/target")

    os.makedirs(run_logdir, exist_ok=True)
    csv_file_path = os.path.join(run_logdir, "generated_samples_paths.csv")

    image_pairs = []
    with open(csv_path, mode="r") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            merged_mask_path = os.path.join(base_dir, row["merged_mask"])
            image_pairs.append(
                (row["source_image"], row["target_image"], merged_mask_path)
            )

    total_cos_dist = 0.0

    for i, (source_image, target_image, merged_mask) in enumerate(tqdm(image_pairs)):
        source_path = os.path.join(source_dir, source_image)
        target_path = os.path.join(target_dir, target_image)

        init_image = read_image(
            source_path,
            device=model.device,
        )

        mask, org_mask = read_mask(
            merged_mask,
            device=model.device,
            dilation_iterations=0,
            dest_size=(64, 64),
        )

        pair_imglogdir = os.path.join(run_logdir, f"{i}", "img")
        pair_numpylogdir = os.path.join(run_logdir, f"{i}", "numpy")
        os.makedirs(pair_imglogdir, exist_ok=True)
        os.makedirs(pair_numpylogdir, exist_ok=True)

        pipeline = build_pipeline(model, opt, target_image_path=target_path)

        print(f"Running for {source_image} -> {target_image} (mask: {merged_mask})")

        cos_dist = run_single_sample(
            model=model,
            pipeline=pipeline,
            imglogdir=pair_imglogdir,
            numpylogdir=pair_numpylogdir,
            opt=opt,
            init_image=init_image,
            mask=mask,
            org_mask=org_mask,
            target_image_path=target_path,
            init_image_path=source_path,
            csv_file=csv_file_path,
        )

        total_cos_dist += float(cos_dist)

    avg_cos_dist = total_cos_dist / max(len(image_pairs), 1)
    print("AVG COS DIST:", avg_cos_dist)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.vanilla_sample:
        raise NotImplementedError(
            "Clean rewrite currently supports DDIM path only. Vanilla path can be added separately."
        )

    ckpt, base_logdir, base_configs = resolve_resume_paths(opt.resume)

    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = torch.cuda.is_available()
    eval_mode = True

    if opt.logdir != "none":
        local_log_name = base_logdir.split(os.sep)[-1]
        if local_log_name == "":
            local_log_name = base_logdir.split(os.sep)[-2]
        print(
            f"Switching logdir from '{base_logdir}' to '{os.path.join(opt.logdir, local_log_name)}'"
        )
        base_logdir = os.path.join(opt.logdir, local_log_name)

    model, global_step = load_model(config, ckpt, gpu, eval_mode)

    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")

    run_logdir, imglogdir, numpylogdir = build_output_dirs(
        base_logdir=base_logdir,
        global_step=global_step,
        timestamp=now,
    )

    print(run_logdir)
    print(75 * "=")

    write_sampling_config(run_logdir, opt)

    if opt.run_tests:
        run_batch_tests(
            model=model,
            opt=opt,
            run_logdir=run_logdir,
        )
        return

    if opt.init_image is None:
        raise ValueError("Single-image mode requires --init_image")
    if opt.target_image is None:
        raise ValueError("Single-image mode requires --target_image")

    print("Reading source image -", opt.init_image)
    init_image = read_image(opt.init_image, device=model.device)

    mask = None
    org_mask = None
    if opt.mask is not None:
        print("Reading mask -", opt.mask)
        mask, org_mask = read_mask(
            opt.mask,
            device=model.device,
            dilation_iterations=0,
            dest_size=(64, 64),
        )

    pipeline = build_pipeline(
        model=model,
        opt=opt,
        target_image_path=opt.target_image,
    )

    run_single_sample(
        model=model,
        pipeline=pipeline,
        imglogdir=imglogdir,
        numpylogdir=numpylogdir,
        opt=opt,
        init_image=init_image,
        mask=mask,
        org_mask=org_mask,
        target_image_path=opt.target_image,
        init_image_path=opt.init_image,
        csv_file=None,
    )


if __name__ == "__main__":
    main()