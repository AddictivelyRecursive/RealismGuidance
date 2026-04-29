# evaluation/pose_expression.py

from __future__ import annotations

import os
import faulthandler

faulthandler.enable()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import bz2
import urllib.request
import argparse
from typing import Optional

import cv2
import dlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from tqdm import tqdm

cv2.setNumThreads(0)


# ---------------------------------------------------------------------------
# Model download helper
# ---------------------------------------------------------------------------

MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"


def ensure_model() -> None:
    if os.path.exists(MODEL_PATH):
        return

    print("Downloading dlib shape predictor model (~100 MB)...")
    bz2_path = MODEL_PATH + ".bz2"

    urllib.request.urlretrieve(MODEL_URL, bz2_path)

    with bz2.open(bz2_path, "rb") as f_in, open(MODEL_PATH, "wb") as f_out:
        f_out.write(f_in.read())

    os.remove(bz2_path)
    print("Model ready.\n")


# ---------------------------------------------------------------------------
# 3-D canonical face model matched to dlib 68-point indices
# ---------------------------------------------------------------------------

MODEL_POINTS_3D = np.array(
    [
        [0.0, 0.0, 0.0],          # 30 – Nose tip
        [0.0, -330.0, -65.0],     # 8  – Chin
        [-225.0, 170.0, -135.0],  # 36 – Left eye outer corner
        [225.0, 170.0, -135.0],   # 45 – Right eye outer corner
        [-150.0, -150.0, -125.0], # 48 – Left mouth corner
        [150.0, -150.0, -125.0],  # 54 – Right mouth corner
    ],
    dtype=np.float64,
)

POSE_LM_INDICES = [30, 8, 36, 45, 48, 54]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _camera_matrix(w: int, h: int) -> np.ndarray:
    f = float(w)
    return np.array(
        [
            [f, 0.0, w / 2.0],
            [0.0, f, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def landmarks_to_array(shape: dlib.full_object_detection) -> np.ndarray:
    return np.array(
        [[shape.part(i).x, shape.part(i).y] for i in range(68)],
        dtype=np.float64,
    )


def extract_attributes(
    image_path: str,
    predictor: dlib.shape_predictor,
) -> Optional[dict]:
    """
    Skips face detection entirely.
    Passes a rectangle covering the full image to dlib's shape predictor.

    Returns:
        {
            "pose_deg": np.array([pitch, yaw, roll]) in degrees,
            "expr": normalized 68-point landmark vector, shape (136,)
        }

    Returns None on failure.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray)

    # Use the whole image as the face bounding box.
    full_rect = dlib.rectangle(left=0, top=0, right=w - 1, bottom=h - 1)

    try:
        shape = predictor(gray, full_rect)
    except Exception:
        return None

    lms = landmarks_to_array(shape)

    if np.allclose(lms, 0.0):
        return None

    # Pose estimation using 6 canonical landmarks.
    img_pts = lms[POSE_LM_INDICES]
    cam = _camera_matrix(w, h)
    dist = np.zeros((4, 1), dtype=np.float64)

    try:
        ok, rvec, _ = cv2.solvePnP(
            MODEL_POINTS_3D,
            img_pts,
            cam,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except Exception:
        return None

    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = cv2.RQDecomp3x3(rmat)[0]

    pose_deg = np.array([pitch, yaw, roll], dtype=np.float64)

    # Expression using translation-scale normalized landmarks.
    nose = lms[30]
    left_eye = lms[36]
    right_eye = lms[45]

    iod = np.linalg.norm(left_eye - right_eye)
    if iod < 1e-6:
        return None

    lms_norm = (lms - nose) / iod
    expr = lms_norm.flatten()

    return {
        "pose_deg": pose_deg,
        "expr": expr,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pose_error_deg(target_attrs: dict, output_attrs: dict) -> float:
    """
    Raw mean absolute pose error in degrees.
    Kept for debugging/comparison, but not the main reported metric.
    """
    return float(
        np.mean(
            np.abs(target_attrs["pose_deg"] - output_attrs["pose_deg"])
        )
    )


def pose_error_norm(
    target_attrs: dict,
    output_attrs: dict,
    max_pose_deg: float = 90.0,
) -> float:
    """
    Normalized pose error in [0, 1].

    0 = perfect pose match.
    1 = pose mismatch >= max_pose_deg.

    This avoids reporting very large raw degree values.
    """
    deg_err = pose_error_deg(target_attrs, output_attrs)
    norm_err = deg_err / max_pose_deg
    return float(np.clip(norm_err, 0.0, 1.0))


def expression_similarity(source_attrs: dict, output_attrs: dict) -> float:
    """
    Cosine similarity between normalized landmark vectors.

    Output should preserve source expression.
    Higher is better.
    """
    sim = 1.0 - cosine(source_attrs["expr"], output_attrs["expr"])
    return float(np.clip(sim, -1.0, 1.0))


def combined_score(
    pose_err_norm: float,
    expr_sim: float,
    pose_w: float = 0.5,
    expr_w: float = 0.5,
) -> float:
    """
    Scalar used only to select the best output among n generated samples.

    Higher is better.

    pose_err_norm is already normalized to [0, 1].
    expr_sim is cosine similarity, usually around [-1, 1].
    """
    return float(expr_w * expr_sim - pose_w * pose_err_norm)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def evaluate(
    csv_path: str,
    output_csv: str = "faceswap_eval_results.csv",
    source_col: str = "source_image",
    target_col: str = "target_image",
    output_col: str = "output_image",
    pose_w: float = 0.5,
    expr_w: float = 0.5,
    max_pose_deg: float = 90.0,
    save_raw_pose_deg: bool = True,
) -> pd.DataFrame:
    ensure_model()
    predictor = dlib.shape_predictor(MODEL_PATH)

    df = pd.read_csv(csv_path)

    missing = {source_col, target_col, output_col} - set(df.columns)
    if missing:
        raise ValueError(
            "CSV missing columns: {}. Available: {}".format(
                missing,
                list(df.columns),
            )
        )

    pairs = df.groupby([source_col, target_col], sort=False)

    print(
        "Found {} unique source-target pairs ({} rows total)\n".format(
            len(pairs),
            len(df),
        )
    )

    records = []
    failed_pairs = []

    for (src_path, tgt_path), group in tqdm(pairs, desc="Pairs"):
        src_attrs = extract_attributes(src_path, predictor)
        tgt_attrs = extract_attributes(tgt_path, predictor)

        if src_attrs is None or tgt_attrs is None:
            failed_pairs.append(
                (src_path, tgt_path, "anchor landmark extraction failed")
            )
            continue

        best_score = -np.inf
        best_metrics = None
        best_out_path = None
        n_valid = 0

        for _, row in group.iterrows():
            out_path = row[output_col]
            out_attrs = extract_attributes(out_path, predictor)

            if out_attrs is None:
                continue

            n_valid += 1

            p_deg = pose_error_deg(tgt_attrs, out_attrs)
            p_norm = float(np.clip(p_deg / max_pose_deg, 0.0, 1.0))
            e_sim = expression_similarity(src_attrs, out_attrs)

            score = combined_score(
                pose_err_norm=p_norm,
                expr_sim=e_sim,
                pose_w=pose_w,
                expr_w=expr_w,
            )

            if score > best_score:
                best_score = score
                best_out_path = out_path
                best_metrics = {
                    "pose_error_deg": p_deg,
                    "pose_error_norm": p_norm,
                    "expression_sim": e_sim,
                    "combined_score": score,
                }

        if best_metrics is None:
            failed_pairs.append((src_path, tgt_path, "no valid output"))
            continue

        record = {
            "source_image": src_path,
            "target_image": tgt_path,
            "best_output_image": best_out_path,
            "pose_error_norm": round(best_metrics["pose_error_norm"], 4),
            "expression_sim": round(best_metrics["expression_sim"], 4),
            "combined_score": round(best_metrics["combined_score"], 4),
            "valid_outputs": n_valid,
            "total_outputs_for_pair": int(len(group)),
        }

        if save_raw_pose_deg:
            record["pose_error_deg_raw"] = round(
                best_metrics["pose_error_deg"],
                4,
            )

        records.append(record)

    results_df = pd.DataFrame(records)
    results_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 60)
    print("  Pairs evaluated : {} / {}".format(len(results_df), len(pairs)))
    print("=" * 60)

    if len(results_df) > 0:
        pe = results_df["pose_error_norm"]
        es = results_df["expression_sim"]
        cs = results_df["combined_score"]

        print(
            "  Pose Error Norm (↓) : {:.4f} ± {:.4f}  [min {:.4f}  max {:.4f}]".format(
                pe.mean(),
                pe.std(),
                pe.min(),
                pe.max(),
            )
        )

        if save_raw_pose_deg and "pose_error_deg_raw" in results_df.columns:
            ped = results_df["pose_error_deg_raw"]
            print(
                "  Pose Error Deg Raw  : {:.4f} ± {:.4f}  [min {:.4f}  max {:.4f}]".format(
                    ped.mean(),
                    ped.std(),
                    ped.min(),
                    ped.max(),
                )
            )

        print(
            "  Expr Sim      (↑) : {:.4f} ± {:.4f}  [min {:.4f}  max {:.4f}]".format(
                es.mean(),
                es.std(),
                es.min(),
                es.max(),
            )
        )

        print(
            "  Combined      (↑) : {:.4f} ± {:.4f}  [min {:.4f}  max {:.4f}]".format(
                cs.mean(),
                cs.std(),
                cs.min(),
                cs.max(),
            )
        )

    print("=" * 60)
    print("\nResults saved -> {}".format(output_csv))

    if failed_pairs:
        print("\nFailed pairs ({}):".format(len(failed_pairs)))
        for s, t, reason in failed_pairs:
            print(
                "  [{}]  {} <-> {}".format(
                    reason,
                    os.path.basename(s),
                    os.path.basename(t),
                )
            )

    return results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Face-swap evaluation: normalized pose error and expression similarity"
        )
    )

    parser.add_argument(
        "csv",
        help="Path to input CSV",
    )

    parser.add_argument(
        "--out",
        default="faceswap_eval_results.csv",
        help="Path to output CSV",
    )

    parser.add_argument(
        "--source-col",
        default="source_image",
        help="Column containing source image paths",
    )

    parser.add_argument(
        "--target-col",
        default="target_image",
        help="Column containing target image paths",
    )

    parser.add_argument(
        "--output-col",
        default="output_image",
        help="Column containing generated output image paths",
    )

    parser.add_argument(
        "--pose-w",
        type=float,
        default=0.5,
        help="Weight for normalized pose error in combined score",
    )

    parser.add_argument(
        "--expr-w",
        type=float,
        default=0.5,
        help="Weight for expression similarity in combined score",
    )

    parser.add_argument(
        "--max-pose-deg",
        type=float,
        default=90.0,
        help="Degree value mapped to normalized pose error 1.0",
    )

    parser.add_argument(
        "--no-raw-pose-deg",
        action="store_true",
        help="Do not save raw degree pose error column",
    )

    args = parser.parse_args()

    evaluate(
        csv_path=args.csv,
        output_csv=args.out,
        source_col=args.source_col,
        target_col=args.target_col,
        output_col=args.output_col,
        pose_w=args.pose_w,
        expr_w=args.expr_w,
        max_pose_deg=args.max_pose_deg,
        save_raw_pose_deg=not args.no_raw_pose_deg,
    )