from __future__ import annotations

import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# 1. Recognition model only (NO detector)
# ──────────────────────────────────────────────────────────────────────────────

def load_recognition_model():
    """
    Load ONLY the ArcFace recognition model.
    No FaceAnalysis. No face detection. No alignment.
    """
    try:
        from insightface.model_zoo import get_model
    except ImportError:
        raise ImportError(
            "insightface is not installed.\n"
            "Install with: pip install insightface onnxruntime"
        )

    model_path = os.path.expanduser(
        "~/.insightface/models/buffalo_l/w600k_r50.onnx"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Recognition model not found at:\n{}".format(model_path)
        )

    providers = ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass

    rec_model = get_model(model_path, providers=providers)

    # ctx_id=0 for CUDA, -1 for CPU only
    if "CUDAExecutionProvider" in providers:
        rec_model.prepare(ctx_id=0)
    else:
        rec_model.prepare(ctx_id=-1)

    print("[INFO] ArcFace recognition model only loaded from: {}".format(model_path))
    print("[INFO] Providers: {}".format(providers))
    return rec_model


# ──────────────────────────────────────────────────────────────────────────────
# 2. Image preprocessing / embedding
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_full_image(
    image_path: str,
    input_size: Tuple[int, int] = (112, 112),
) -> Optional[np.ndarray]:
    """
    Read the full image and resize directly to recognizer input size.
    No face detection or landmark alignment.
    """
    img = cv2.imread(image_path)
    if img is None:
        print("[WARN] Cannot read image: {}".format(image_path))
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
    return img


def get_embedding(rec_model, image_path: str) -> Optional[np.ndarray]:
    """
    Extract an L2-normalized embedding from the FULL image directly.
    """
    img = preprocess_full_image(image_path, input_size=(112, 112))
    if img is None:
        return None

    try:
        emb = rec_model.get_feat(img)
    except Exception as e:
        print("[WARN] Embedding failed for {} : {}".format(image_path, e))
        return None

    if emb is None:
        print("[WARN] No embedding returned for: {}".format(image_path))
        return None

    emb = np.asarray(emb).reshape(-1).astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm < 1e-12:
        print("[WARN] Near-zero embedding for: {}".format(image_path))
        return None

    emb = emb / norm
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ──────────────────────────────────────────────────────────────────────────────
# 3. CSV loading & grouping
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    col_map: Dict[str, str] = {}
    for col in df.columns:
        lc = col.lower().strip()
        if "source" in lc and "image" in lc:
            col_map["source_image"] = col
        elif "target" in lc and "image" in lc:
            col_map["target_image"] = col
        elif "output" in lc and "image" in lc:
            col_map["output_image"] = col

    required = ["source_image", "target_image", "output_image"]
    missing = [k for k in required if k not in col_map]
    if missing:
        raise ValueError(
            "Could not find columns for: {}\n"
            "Available columns: {}\n"
            "Rename them or adjust the detection logic.".format(
                missing, list(df.columns)
            )
        )

    df = df.rename(columns={v: k for k, v in col_map.items()})

    print("Loaded {} rows from CSV.".format(len(df)))
    print("  source_image -> '{}'".format(col_map["source_image"]))
    print("  target_image -> '{}'".format(col_map["target_image"]))
    print("  output_image -> '{}'".format(col_map["output_image"]))
    return df


def group_outputs(df: pd.DataFrame) -> Dict[Tuple[str, str], List[str]]:
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for _, row in df.iterrows():
        key = (row["source_image"], row["target_image"])
        groups[key].append(row["output_image"])

    print("\nFound {} unique (source, target) pairs.".format(len(groups)))

    lengths = [len(v) for v in groups.values()]
    if lengths:
        print(
            "  Outputs per pair - min:{}  max:{}  mean:{:.1f}".format(
                min(lengths), max(lengths), float(np.mean(lengths))
            )
        )
    else:
        print("  No grouped pairs found.")

    return dict(groups)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Gallery
# ──────────────────────────────────────────────────────────────────────────────

def build_gallery(rec_model, source_paths: List[str]) -> Dict[str, np.ndarray]:
    unique_sources = list(dict.fromkeys(source_paths))
    print("\nBuilding gallery for {} unique source identities".format(len(unique_sources)))

    gallery: Dict[str, np.ndarray] = {}
    for src in tqdm(unique_sources, desc="Gallery"):
        emb = get_embedding(rec_model, src)
        if emb is not None:
            gallery[src] = emb

    print(
        "  Gallery built - {}/{} embeddings extracted.".format(
            len(gallery), len(unique_sources)
        )
    )
    return gallery


# ──────────────────────────────────────────────────────────────────────────────
# 5. Mean aggregation across all outputs for each pair
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_pair_outputs_mean(
    rec_model,
    groups: Dict[Tuple[str, str], List[str]],
    gallery: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """
    For each (source, target) pair:
    - embed all outputs
    - average the embeddings
    - L2-normalize the mean embedding
    """
    results: List[Dict[str, Any]] = []
    skipped = 0

    print("\nAggregating outputs per pair using mean embedding")
    for (src, tgt), outputs in tqdm(groups.items(), desc="Pairs"):
        src_emb = gallery.get(src)
        if src_emb is None:
            skipped += 1
            continue

        out_embs: List[np.ndarray] = []
        for out_path in outputs:
            out_emb = get_embedding(rec_model, out_path)
            if out_emb is not None:
                out_embs.append(out_emb)

        if not out_embs:
            skipped += 1
            continue

        mean_emb = np.mean(np.stack(out_embs, axis=0), axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        if norm < 1e-12:
            skipped += 1
            continue

        mean_emb = mean_emb / norm
        mean_sim = cosine_similarity(src_emb, mean_emb)

        results.append(
            {
                "source": src,
                "target": tgt,
                "mean_output_emb": mean_emb,
                "n_candidates": len(outputs),
                "n_valid": len(out_embs),
                "mean_sim_to_source": mean_sim,
            }
        )

    print("  Aggregated {} pairs ({} skipped).".format(len(results), skipped))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 6. Top-K evaluation from mean pair embeddings
# ──────────────────────────────────────────────────────────────────────────────

def compute_topk_id_similarity_from_mean_pairs(
    aggregated: List[Dict[str, Any]],
    gallery: Dict[str, np.ndarray],
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    if not gallery:
        print("[ERROR] Gallery is empty.")
        return {}

    gallery_keys = list(gallery.keys())
    gallery_embs = np.stack(list(gallery.values()), axis=0)

    hits = {k: 0 for k in ks}
    all_sims: List[float] = []
    all_ranks: List[int] = []
    valid_counts: List[int] = []
    n_valid = 0

    print(
        "\nComputing Top-K ID Similarity from mean pair embeddings over {} pairs".format(
            len(aggregated)
        )
    )

    key_to_index = {k: i for i, k in enumerate(gallery_keys)}

    for record in tqdm(aggregated, desc="Top-K eval"):
        src = record["source"]
        pair_emb = record["mean_output_emb"]

        if src not in key_to_index:
            continue

        sims = gallery_embs @ pair_emb
        ranked_indices = np.argsort(sims)[::-1]

        correct_idx = key_to_index[src]
        rank = int(np.where(ranked_indices == correct_idx)[0][0]) + 1

        all_sims.append(float(sims[correct_idx]))
        all_ranks.append(rank)
        valid_counts.append(int(record["n_valid"]))
        n_valid += 1

        for k in ks:
            if rank <= k:
                hits[k] += 1

    if n_valid == 0:
        print("[ERROR] No valid pairs to evaluate.")
        return {}

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics["top{}".format(k)] = hits[k] / float(n_valid)

    metrics["mean_id_sim"] = float(np.mean(all_sims))
    metrics["mean_rank"] = float(np.mean(all_ranks))
    metrics["mean_valid_outputs_per_pair"] = float(np.mean(valid_counts))
    metrics["n_evaluated"] = float(n_valid)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 7. Reporting
# ──────────────────────────────────────────────────────────────────────────────

def print_metrics(metrics: Dict[str, Any]) -> None:
    if not metrics:
        print("\nNo metrics to print.")
        return

    print("\n" + "=" * 50)
    print("  TOP-K ID SIMILARITY RESULTS")
    print("=" * 50)

    for k, v in metrics.items():
        if k.startswith("top"):
            print("  {:8s}  {:6.2f}%".format(k.upper(), v * 100.0))

    print("  {:8s}  {:.4f}".format("Mean Sim", metrics.get("mean_id_sim", 0.0)))
    print("  {:8s}  {:.2f}".format("Mean Rank", metrics.get("mean_rank", 0.0)))
    print(
        "  {:8s}  {:.2f}".format(
            "MeanValid", metrics.get("mean_valid_outputs_per_pair", 0.0)
        )
    )
    print("  Evaluated on {}".format(int(metrics.get("n_evaluated", 0))))
    print("=" * 50)


def save_results(
    aggregated: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    out_dir: str = ".",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for r in aggregated:
        rows.append(
            {
                "source": r["source"],
                "target": r["target"],
                "mean_id_sim": r["mean_sim_to_source"],
                "n_candidates": r["n_candidates"],
                "n_valid": r["n_valid"],
            }
        )

    df_out = pd.DataFrame(rows)
    pair_csv = os.path.join(out_dir, "aggregated_pair_results.csv")
    df_out.to_csv(pair_csv, index=False)
    print("\nSaved per-pair results -> {}".format(pair_csv))

    metrics_path = os.path.join(out_dir, "topk_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved summary metrics  -> {}".format(metrics_path))


# ──────────────────────────────────────────────────────────────────────────────
# 8. Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Top-K ID Similarity for Face Swap Evaluation (no detector, mean-pair)"
    )
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    parser.add_argument("--ks", default="1,5,10", help="Comma-separated K values")
    parser.add_argument(
        "--out_dir",
        default="./eval_results",
        help="Directory to save outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        ks = tuple(int(k.strip()) for k in args.ks.split(",") if k.strip())
    except ValueError:
        raise ValueError("--ks must be a comma-separated list of integers, e.g. 1,5,10")

    print("Loading ArcFace recognition model")
    rec_model = load_recognition_model()

    df = load_csv(args.csv)
    groups = group_outputs(df)

    all_sources = [src for (src, _) in groups.keys()]
    gallery = build_gallery(rec_model, all_sources)

    aggregated = aggregate_pair_outputs_mean(rec_model, groups, gallery)
    metrics = compute_topk_id_similarity_from_mean_pairs(aggregated, gallery, ks=ks)

    print_metrics(metrics)
    save_results(aggregated, metrics, out_dir=args.out_dir)


if __name__ == "__main__":
    main()