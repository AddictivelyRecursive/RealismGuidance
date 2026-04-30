from __future__ import annotations

import os
import json
import bz2
import argparse
import urllib.request
import faulthandler
from typing import Dict, List, Optional, Tuple, Any

faulthandler.enable()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import dlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as scipy_cosine
from tqdm import tqdm

cv2.setNumThreads(0)


MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"


def ensure_landmark_model() -> None:
    if os.path.exists(MODEL_PATH):
        return

    print("Downloading dlib shape predictor model...")
    bz2_path = MODEL_PATH + ".bz2"
    urllib.request.urlretrieve(MODEL_URL, bz2_path)

    with bz2.open(bz2_path, "rb") as f_in, open(MODEL_PATH, "wb") as f_out:
        f_out.write(f_in.read())

    os.remove(bz2_path)


def load_recognition_model():
    try:
        from insightface.model_zoo import get_model
    except ImportError:
        raise ImportError(
            "insightface is not installed.\n"
            "Install with: pip install insightface onnxruntime"
        )

    model_path = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Recognition model not found at:\n{}".format(model_path))

    providers = ["CPUExecutionProvider"]

    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass

    rec_model = get_model(model_path, providers=providers)
    rec_model.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1)

    print("[INFO] ArcFace model loaded from: {}".format(model_path))
    print("[INFO] Providers: {}".format(providers))

    return rec_model


MODEL_POINTS_3D = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
    ],
    dtype=np.float64,
)

POSE_LM_INDICES = [30, 8, 36, 45, 48, 54]


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
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        print("[WARN] Cannot read image: {}".format(image_path))
        return None

    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]

    gray = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    full_rect = dlib.rectangle(left=0, top=0, right=w - 1, bottom=h - 1)

    try:
        shape = predictor(gray, full_rect)
    except Exception:
        return None

    lms = landmarks_to_array(shape)

    if np.allclose(lms, 0.0):
        return None

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

    nose = lms[30]
    left_eye = lms[36]
    right_eye = lms[45]

    iod = np.linalg.norm(left_eye - right_eye)

    if iod < 1e-6:
        return None

    expr = ((lms - nose) / iod).flatten()

    return {
        "pose_deg": pose_deg,
        "expr": expr,
    }


def pose_error_norm(
    target_attrs: dict,
    output_attrs: dict,
    max_pose_deg: float = 90.0,
) -> float:
    deg_err = float(
        np.mean(np.abs(target_attrs["pose_deg"] - output_attrs["pose_deg"]))
    )

    return float(np.clip(deg_err / max_pose_deg, 0.0, 1.0))


def expression_similarity(
    source_attrs: dict,
    output_attrs: dict,
) -> float:
    sim = 1.0 - scipy_cosine(source_attrs["expr"], output_attrs["expr"])
    return float(np.clip(sim, -1.0, 1.0))


def preprocess_full_image(
    image_path: str,
    input_size: Tuple[int, int] = (112, 112),
) -> Optional[np.ndarray]:
    img = cv2.imread(image_path)

    if img is None:
        print("[WARN] Cannot read image: {}".format(image_path))
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)

    return img


def get_embedding(
    rec_model,
    image_path: str,
) -> Optional[np.ndarray]:
    img = preprocess_full_image(image_path)

    if img is None:
        return None

    try:
        emb = rec_model.get_feat(img)
    except Exception as e:
        print("[WARN] Embedding failed for {} : {}".format(image_path, e))
        return None

    if emb is None:
        return None

    emb = np.asarray(emb).reshape(-1).astype(np.float32)
    norm = np.linalg.norm(emb)

    if norm < 1e-12:
        return None

    return emb / norm


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return float(np.dot(a, b))


def load_csv(
    csv_path: str,
    source_col: str,
    target_col: str,
    output_col: str,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if top_k is not None:
        df = df.head(top_k).copy()
        print("[INFO] Using only top {} rows from CSV.".format(top_k))

    missing = {source_col, target_col, output_col} - set(df.columns)

    if missing:
        raise ValueError(
            "CSV missing columns: {}. Available: {}".format(
                sorted(missing),
                list(df.columns),
            )
        )

    df = df.rename(
        columns={
            source_col: "source_image",
            target_col: "target_image",
            output_col: "output_image",
        }
    )

    print("Loaded {} rows from CSV.".format(len(df)))

    return df


def evaluate(
    csv_path: str,
    output_csv: str = "faceswap_eval_results.csv",
    summary_json: str = "faceswap_eval_summary.json",
    source_col: str = "source_image",
    target_col: str = "target_image",
    output_col: str = "output_image",
    max_pose_deg: float = 90.0,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    ensure_landmark_model()

    predictor = dlib.shape_predictor(MODEL_PATH)
    rec_model = load_recognition_model()

    df = load_csv(csv_path, source_col, target_col, output_col, top_k=top_k)
    pairs = df.groupby(["source_image", "target_image"], sort=False)

    print(
        "\nFound {} unique (source, target) pairs ({} rows total)\n".format(
            len(pairs),
            len(df),
        )
    )

    records: List[Dict[str, Any]] = []
    failed_pairs: List[Tuple[str, str, str]] = []

    for (src_path, tgt_path), group in tqdm(pairs, desc="Pairs"):
        src_attrs = extract_attributes(src_path, predictor)
        tgt_attrs = extract_attributes(tgt_path, predictor)
        src_emb = get_embedding(rec_model, src_path)

        if src_attrs is None or tgt_attrs is None:
            failed_pairs.append((src_path, tgt_path, "anchor landmark extraction failed"))
            continue

        if src_emb is None:
            failed_pairs.append((src_path, tgt_path, "source ArcFace embedding failed"))
            continue

        output_records: List[Dict[str, Any]] = []

        for _, row in group.iterrows():
            out_path = row["output_image"]

            out_attrs = extract_attributes(out_path, predictor)
            out_emb = get_embedding(rec_model, out_path)

            if out_attrs is None or out_emb is None:
                continue

            pose_val = pose_error_norm(
                tgt_attrs,
                out_attrs,
                max_pose_deg=max_pose_deg,
            )

            expr_val = expression_similarity(src_attrs, out_attrs)
            id_val = cosine_similarity(src_emb, out_emb)

            output_records.append(
                {
                    "output_image": out_path,
                    "pose_error_norm": float(pose_val),
                    "expression_sim": float(expr_val),
                    "id_similarity": float(id_val),
                }
            )

        if not output_records:
            failed_pairs.append((src_path, tgt_path, "no valid outputs"))
            continue

        best_pose = min(output_records, key=lambda x: x["pose_error_norm"])
        best_expr = max(output_records, key=lambda x: x["expression_sim"])
        best_id = max(output_records, key=lambda x: x["id_similarity"])

        records.append(
            {
                "source_image": src_path,
                "target_image": tgt_path,
                "pose_error_norm": round(float(best_pose["pose_error_norm"]), 4),
                "pose_best_output": best_pose["output_image"],
                "expression_sim": round(float(best_expr["expression_sim"]), 4),
                "expression_best_output": best_expr["output_image"],
                "id_similarity": round(float(best_id["id_similarity"]), 4),
                "id_best_output": best_id["output_image"],
                "valid_outputs": int(len(output_records)),
                "total_outputs": int(len(group)),
            }
        )

    results_df = pd.DataFrame(records)
    results_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY  ({} / {} pairs)".format(len(results_df), len(pairs)))
    print("=" * 60)

    summary: Dict[str, Any] = {
        "pairs_evaluated": int(len(results_df)),
        "pairs_total": int(len(pairs)),
        "max_pose_deg": float(max_pose_deg),
    }

    if len(results_df) > 0:
        metrics_info = [
            ("pose_error_norm", "Pose Error Norm     (↓ better)"),
            ("expression_sim", "Expression Sim      (↑ better)"),
            ("id_similarity", "ID Similarity       (↑ better)"),
        ]

        for col, label in metrics_info:
            s = results_df[col]

            print(
                "  {:30s}  mean={:.4f}  std={:.4f}  min={:.4f}  max={:.4f}".format(
                    label,
                    s.mean(),
                    s.std(),
                    s.min(),
                    s.max(),
                )
            )

            summary[col] = {
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
            }

    print("=" * 60)
    print("\nPer-pair results saved -> {}".format(output_csv))

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("Summary saved          -> {}".format(summary_json))

    if failed_pairs:
        print("\nFailed pairs ({}):".format(len(failed_pairs)))

        for s, t, reason in failed_pairs:
            print(
                "  [{}] {} <-> {}".format(
                    reason,
                    os.path.basename(s),
                    os.path.basename(t),
                )
            )

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("csv")
    parser.add_argument("--out", default="faceswap_eval_results.csv")
    parser.add_argument("--summary-json", default="faceswap_eval_summary.json")
    parser.add_argument("--source-col", default="source_image")
    parser.add_argument("--target-col", default="target_image")
    parser.add_argument("--output-col", default="output_image")
    parser.add_argument("--max-pose-deg", type=float, default=90.0)
    parser.add_argument("--top-k", type=int, default=None)

    args = parser.parse_args()

    evaluate(
        csv_path=args.csv,
        output_csv=args.out,
        summary_json=args.summary_json,
        source_col=args.source_col,
        target_col=args.target_col,
        output_col=args.output_col,
        max_pose_deg=args.max_pose_deg,
        top_k=args.top_k,
    )
    
# from __future__ import annotations

# import os
# import json
# import bz2
# import argparse
# import urllib.request
# import faulthandler
# from typing import Dict, List, Optional, Tuple, Any

# faulthandler.enable()

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# import cv2
# import dlib
# import numpy as np
# import pandas as pd
# from scipy.spatial.distance import cosine as scipy_cosine
# from tqdm import tqdm

# cv2.setNumThreads(0)


# MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
# MODEL_PATH = "shape_predictor_68_face_landmarks.dat"


# def ensure_landmark_model() -> None:
#     if os.path.exists(MODEL_PATH):
#         return

#     print("Downloading dlib shape predictor model...")
#     bz2_path = MODEL_PATH + ".bz2"
#     urllib.request.urlretrieve(MODEL_URL, bz2_path)

#     with bz2.open(bz2_path, "rb") as f_in, open(MODEL_PATH, "wb") as f_out:
#         f_out.write(f_in.read())

#     os.remove(bz2_path)


# def load_recognition_model():
#     try:
#         from insightface.model_zoo import get_model
#     except ImportError:
#         raise ImportError(
#             "insightface is not installed.\n"
#             "Install with: pip install insightface onnxruntime"
#         )

#     model_path = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")

#     if not os.path.exists(model_path):
#         raise FileNotFoundError("Recognition model not found at:\n{}".format(model_path))

#     providers = ["CPUExecutionProvider"]

#     try:
#         import onnxruntime as ort

#         if "CUDAExecutionProvider" in ort.get_available_providers():
#             providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
#     except Exception:
#         pass

#     rec_model = get_model(model_path, providers=providers)
#     rec_model.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1)

#     print("[INFO] ArcFace model loaded from: {}".format(model_path))
#     print("[INFO] Providers: {}".format(providers))

#     return rec_model


# MODEL_POINTS_3D = np.array(
#     [
#         [0.0, 0.0, 0.0],
#         [0.0, -330.0, -65.0],
#         [-225.0, 170.0, -135.0],
#         [225.0, 170.0, -135.0],
#         [-150.0, -150.0, -125.0],
#         [150.0, -150.0, -125.0],
#     ],
#     dtype=np.float64,
# )

# POSE_LM_INDICES = [30, 8, 36, 45, 48, 54]


# def _camera_matrix(w: int, h: int) -> np.ndarray:
#     f = float(w)
#     return np.array(
#         [
#             [f, 0.0, w / 2.0],
#             [0.0, f, h / 2.0],
#             [0.0, 0.0, 1.0],
#         ],
#         dtype=np.float64,
#     )


# def landmarks_to_array(shape: dlib.full_object_detection) -> np.ndarray:
#     return np.array(
#         [[shape.part(i).x, shape.part(i).y] for i in range(68)],
#         dtype=np.float64,
#     )


# def extract_attributes(
#     image_path: str,
#     predictor: dlib.shape_predictor,
# ) -> Optional[dict]:
#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)

#     if img is None:
#         print("[WARN] Cannot read image: {}".format(image_path))
#         return None

#     img = np.ascontiguousarray(img)
#     h, w = img.shape[:2]

#     gray = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#     full_rect = dlib.rectangle(left=0, top=0, right=w - 1, bottom=h - 1)

#     try:
#         shape = predictor(gray, full_rect)
#     except Exception:
#         return None

#     lms = landmarks_to_array(shape)

#     if np.allclose(lms, 0.0):
#         return None

#     img_pts = lms[POSE_LM_INDICES]
#     cam = _camera_matrix(w, h)
#     dist = np.zeros((4, 1), dtype=np.float64)

#     try:
#         ok, rvec, _ = cv2.solvePnP(
#             MODEL_POINTS_3D,
#             img_pts,
#             cam,
#             dist,
#             flags=cv2.SOLVEPNP_ITERATIVE,
#         )
#     except Exception:
#         return None

#     if not ok:
#         return None

#     rmat, _ = cv2.Rodrigues(rvec)
#     pitch, yaw, roll = cv2.RQDecomp3x3(rmat)[0]

#     pose_deg = np.array([pitch, yaw, roll], dtype=np.float64)

#     nose = lms[30]
#     left_eye = lms[36]
#     right_eye = lms[45]

#     iod = np.linalg.norm(left_eye - right_eye)

#     if iod < 1e-6:
#         return None

#     expr = ((lms - nose) / iod).flatten()

#     return {
#         "pose_deg": pose_deg,
#         "expr": expr,
#     }


# def pose_error_norm(
#     target_attrs: dict,
#     output_attrs: dict,
#     max_pose_deg: float = 90.0,
# ) -> float:
#     deg_err = float(
#         np.mean(np.abs(target_attrs["pose_deg"] - output_attrs["pose_deg"]))
#     )

#     return float(np.clip(deg_err / max_pose_deg, 0.0, 1.0))


# def expression_similarity(
#     source_attrs: dict,
#     output_attrs: dict,
# ) -> float:
#     sim = 1.0 - scipy_cosine(source_attrs["expr"], output_attrs["expr"])
#     return float(np.clip(sim, -1.0, 1.0))


# def preprocess_full_image(
#     image_path: str,
#     input_size: Tuple[int, int] = (112, 112),
# ) -> Optional[np.ndarray]:
#     img = cv2.imread(image_path)

#     if img is None:
#         print("[WARN] Cannot read image: {}".format(image_path))
#         return None

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)

#     return img


# def get_embedding(
#     rec_model,
#     image_path: str,
# ) -> Optional[np.ndarray]:
#     img = preprocess_full_image(image_path)

#     if img is None:
#         return None

#     try:
#         emb = rec_model.get_feat(img)
#     except Exception as e:
#         print("[WARN] Embedding failed for {} : {}".format(image_path, e))
#         return None

#     if emb is None:
#         return None

#     emb = np.asarray(emb).reshape(-1).astype(np.float32)
#     norm = np.linalg.norm(emb)

#     if norm < 1e-12:
#         return None

#     return emb / norm


# def cosine_similarity(
#     a: np.ndarray,
#     b: np.ndarray,
# ) -> float:
#     return float(np.dot(a, b))


# def load_csv(
#     csv_path: str,
#     source_col: str,
#     target_col: str,
#     output_col: str,
# ) -> pd.DataFrame:
#     df = pd.read_csv(csv_path)

#     missing = {source_col, target_col, output_col} - set(df.columns)

#     if missing:
#         raise ValueError(
#             "CSV missing columns: {}. Available: {}".format(
#                 sorted(missing),
#                 list(df.columns),
#             )
#         )

#     df = df.rename(
#         columns={
#             source_col: "source_image",
#             target_col: "target_image",
#             output_col: "output_image",
#         }
#     )

#     print("Loaded {} rows from CSV.".format(len(df)))

#     return df


# def evaluate(
#     csv_path: str,
#     output_csv: str = "faceswap_eval_results.csv",
#     summary_json: str = "faceswap_eval_summary.json",
#     source_col: str = "source_image",
#     target_col: str = "target_image",
#     output_col: str = "output_image",
#     max_pose_deg: float = 90.0,
# ) -> pd.DataFrame:
#     ensure_landmark_model()

#     predictor = dlib.shape_predictor(MODEL_PATH)
#     rec_model = load_recognition_model()

#     df = load_csv(csv_path, source_col, target_col, output_col)
#     pairs = df.groupby(["source_image", "target_image"], sort=False)

#     print(
#         "\nFound {} unique (source, target) pairs ({} rows total)\n".format(
#             len(pairs),
#             len(df),
#         )
#     )

#     records: List[Dict[str, Any]] = []
#     failed_pairs: List[Tuple[str, str, str]] = []

#     for (src_path, tgt_path), group in tqdm(pairs, desc="Pairs"):
#         src_attrs = extract_attributes(src_path, predictor)
#         tgt_attrs = extract_attributes(tgt_path, predictor)
#         src_emb = get_embedding(rec_model, src_path)

#         if src_attrs is None or tgt_attrs is None:
#             failed_pairs.append((src_path, tgt_path, "anchor landmark extraction failed"))
#             continue

#         if src_emb is None:
#             failed_pairs.append((src_path, tgt_path, "source ArcFace embedding failed"))
#             continue

#         pose_values: List[float] = []
#         expr_values: List[float] = []
#         id_values: List[float] = []

#         for _, row in group.iterrows():
#             out_path = row["output_image"]

#             out_attrs = extract_attributes(out_path, predictor)
#             out_emb = get_embedding(rec_model, out_path)

#             if out_attrs is None or out_emb is None:
#                 continue

#             pose_values.append(
#                 pose_error_norm(
#                     tgt_attrs,
#                     out_attrs,
#                     max_pose_deg=max_pose_deg,
#                 )
#             )

#             expr_values.append(expression_similarity(src_attrs, out_attrs))
#             id_values.append(cosine_similarity(src_emb, out_emb))

#         if not pose_values:
#             failed_pairs.append((src_path, tgt_path, "no valid outputs"))
#             continue

#         records.append(
#             {
#                 "source_image": src_path,
#                 "target_image": tgt_path,
#                 "pose_error_norm": round(float(np.mean(pose_values)), 4),
#                 "expression_sim": round(float(np.mean(expr_values)), 4),
#                 "id_similarity": round(float(np.mean(id_values)), 4),
#                 "valid_outputs": int(len(pose_values)),
#                 "total_outputs": int(len(group)),
#             }
#         )

#     results_df = pd.DataFrame(records)
#     results_df.to_csv(output_csv, index=False)

#     print("\n" + "=" * 60)
#     print("  EVALUATION SUMMARY  ({} / {} pairs)".format(len(results_df), len(pairs)))
#     print("=" * 60)

#     summary: Dict[str, Any] = {
#         "pairs_evaluated": int(len(results_df)),
#         "pairs_total": int(len(pairs)),
#         "max_pose_deg": float(max_pose_deg),
#     }

#     if len(results_df) > 0:
#         metrics_info = [
#             ("pose_error_norm", "Pose Error Norm     (↓ better)"),
#             ("expression_sim", "Expression Sim      (↑ better)"),
#             ("id_similarity", "ID Similarity       (↑ better)"),
#         ]

#         for col, label in metrics_info:
#             s = results_df[col]

#             print(
#                 "  {:30s}  mean={:.4f}  std={:.4f}  min={:.4f}  max={:.4f}".format(
#                     label,
#                     s.mean(),
#                     s.std(),
#                     s.min(),
#                     s.max(),
#                 )
#             )

#             summary[col] = {
#                 "mean": round(float(s.mean()), 4),
#                 "std": round(float(s.std()), 4),
#                 "min": round(float(s.min()), 4),
#                 "max": round(float(s.max()), 4),
#             }

#     print("=" * 60)
#     print("\nPer-pair results saved -> {}".format(output_csv))

#     with open(summary_json, "w") as f:
#         json.dump(summary, f, indent=2)

#     print("Summary saved          -> {}".format(summary_json))

#     if failed_pairs:
#         print("\nFailed pairs ({}):".format(len(failed_pairs)))

#         for s, t, reason in failed_pairs:
#             print(
#                 "  [{}] {} <-> {}".format(
#                     reason,
#                     os.path.basename(s),
#                     os.path.basename(t),
#                 )
#             )

#     return results_df


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument("csv")
#     parser.add_argument("--out", default="faceswap_eval_results.csv")
#     parser.add_argument("--summary-json", default="faceswap_eval_summary.json")
#     parser.add_argument("--source-col", default="source_image")
#     parser.add_argument("--target-col", default="target_image")
#     parser.add_argument("--output-col", default="output_image")
#     parser.add_argument("--max-pose-deg", type=float, default=90.0)

#     args = parser.parse_args()

#     evaluate(
#         csv_path=args.csv,
#         output_csv=args.out,
#         summary_json=args.summary_json,
#         source_col=args.source_col,
#         target_col=args.target_col,
#         output_col=args.output_col,
#         max_pose_deg=args.max_pose_deg,
#     )