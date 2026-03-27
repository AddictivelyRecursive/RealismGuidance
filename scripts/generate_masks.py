from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DIFFSWAP_ROOT = ROOT / "external" / "DiffSwap"
if DIFFSWAP_ROOT.exists() and str(DIFFSWAP_ROOT) not in sys.path:
    sys.path.append(str(DIFFSWAP_ROOT))

from modules.masking.generator import LandmarkMaskGenerator


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate landmark-based masks from image_pairs.csv"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/image_pairs.csv",
        help="CSV containing source_image,target_image",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="data/source",
        help="Directory containing source images",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="data/target",
        help="Directory containing target images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Base output directory for align/ and mask/",
    )
    parser.add_argument(
        "--predictor_path",
        type=str,
        default="checkpoints/shape_predictor_68_face_landmarks.dat",
        help="Path to dlib 68-landmark predictor",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=256,
        help="Aligned output size",
    )
    parser.add_argument(
        "--mask_blur_kernel",
        type=int,
        default=15,
        help="Gaussian blur kernel for mask smoothing",
    )
    parser.add_argument(
        "--crop_margin",
        type=float,
        default=0.35,
        help="Extra crop margin around detected face",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    generator = LandmarkMaskGenerator(
        predictor_path=args.predictor_path,
        output_size=args.output_size,
        mask_blur_kernel=args.mask_blur_kernel,
        crop_margin=args.crop_margin,
    )

    output_csv = generator.run_from_csv(
        csv_path=args.csv,
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        output_dir=args.output_dir,
    )

    print(f"[INFO] Finished generating masks.")
    print(f"[INFO] Output CSV: {output_csv}")


if __name__ == "__main__":
    main()
    
''' We generate face-region masks using a landmark-based preprocessing pipeline. 
For each source-target pair, we detect facial landmarks using a 68-point dlib predictor, 
crop each face around the largest detected face region with a fixed margin, 
resize to a common spatial resolution, 
and construct a smooth face mask from the convex hull of facial landmarks. 
The final swap mask is obtained by a union-style merge of the source and 
target landmark-derived masks.'''