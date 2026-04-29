import pandas as pd
from pathlib import Path

INPUT_CSV = "image_pairs.csv"
OUTPUT_CSV = "LDFaceNet_outputpath.csv"

SOURCE_PREFIX = "/storage/users/multicog/mehul/RealismGuidance/data/source"
TARGET_PREFIX = "/storage/users/multicog/mehul/RealismGuidance/data/target"
OUTPUT_PREFIX = "/storage/users/multicog/mehul/RealismGuidance/comparison_outputs/LDFaceNet"


def main():
    df = pd.read_csv(INPUT_CSV)

    required_cols = {"target_image", "source_image"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    output_rows = []

    for pair_idx, row in df.iterrows():
        target_img = str(row["target_image"]).strip()
        source_img = str(row["source_image"]).strip()

        source_path = f"{SOURCE_PREFIX}/{source_img}"
        target_path = f"{TARGET_PREFIX}/{target_img}"

        img_dir = Path(OUTPUT_PREFIX) / str(pair_idx) / "img"

        if not img_dir.exists():
            print(f"[WARN] Missing output directory: {img_dir}")
            continue

        sample_paths = sorted(
            p for p in img_dir.glob("sample_*.png")
            if not p.name.startswith("combined_sample_")
        )

        if not sample_paths:
            print(f"[WARN] No sample outputs found in: {img_dir}")
            continue

        for sample_path in sample_paths:
            output_rows.append({
                "source_image": source_path,
                "target_image": target_path,
                "output_image": str(sample_path),
                "pair_index": pair_idx,
                "sample_name": sample_path.name,
            })

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"[DONE] Saved: {OUTPUT_CSV}")
    print(f"[INFO] Rows written: {len(out_df)}")
    print(f"[INFO] Unique pairs found: {out_df['pair_index'].nunique() if len(out_df) else 0}")


if __name__ == "__main__":
    main()