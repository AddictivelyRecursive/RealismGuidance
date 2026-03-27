from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import dlib
import numpy as np
from PIL import Image
from imutils import face_utils
from tqdm import tqdm


@dataclass
class ImagePair:
    source_image: str
    target_image: str


@dataclass
class CropResult:
    image_bgr: np.ndarray
    landmarks: np.ndarray
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


class LandmarkMaskGenerator:
    """
    Clean landmark-based masking pipeline.

    Main principles:
    - preserve landmark-based masking for paper narration
    - avoid modifying original dataset
    - crop around largest detected face, then resize to output_size
    - build full-face mask from facial landmarks
    """

    def __init__(
        self,
        predictor_path: str,
        output_size: int = 256,
        mask_blur_kernel: int = 15,
        crop_margin: float = 0.35,
        detector_upsample_levels: int = 2,
    ):
        self.detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(predictor_path)

        self.output_size = output_size
        self.mask_blur_kernel = mask_blur_kernel
        self.crop_margin = crop_margin
        self.detector_upsample_levels = detector_upsample_levels

    @staticmethod
    def load_pairs_from_csv(csv_path: str | Path) -> List[ImagePair]:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        pairs: List[ImagePair] = []
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"source_image", "target_image"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    "CSV must contain header columns: source_image,target_image"
                )

            for row in reader:
                src = (row.get("source_image") or "").strip()
                tgt = (row.get("target_image") or "").strip()
                if src and tgt:
                    pairs.append(ImagePair(source_image=src, target_image=tgt))

        return pairs

    @staticmethod
    def pair_key(source_name: str, target_name: str) -> str:
        src_stem = Path(source_name).stem
        tgt_stem = Path(target_name).stem
        return f"{src_stem}__{tgt_stem}"

    @staticmethod
    def load_image_bgr(image_path: str | Path) -> np.ndarray:
        image_path = str(image_path)
        image = cv2.imread(image_path)
        if image is not None:
            return np.ascontiguousarray(image)

        pil_image = Image.open(image_path).convert("RGB")
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(image)

    def detect_largest_face_and_landmarks(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[dlib.rectangle, np.ndarray]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        faces = []
        for level in range(self.detector_upsample_levels + 1):
            faces = self.detector(gray, level)
            if len(faces) > 0:
                break

        if len(faces) == 0:
            raise RuntimeError("No face detected.")

        face = max(faces, key=lambda r: r.width() * r.height())
        shape = self.landmark_predictor(image_bgr, face)
        landmarks = face_utils.shape_to_np(shape)

        return face, landmarks

    def crop_around_face(
        self,
        image_bgr: np.ndarray,
        face_rect: dlib.rectangle,
    ) -> np.ndarray:
        h, w = image_bgr.shape[:2]

        x1 = face_rect.left()
        y1 = face_rect.top()
        x2 = face_rect.right()
        y2 = face_rect.bottom()

        face_w = x2 - x1
        face_h = y2 - y1

        margin_x = int(face_w * self.crop_margin)
        margin_y = int(face_h * self.crop_margin)

        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(w, x2 + margin_x)
        crop_y2 = min(h, y2 + margin_y)

        cropped = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if cropped.size == 0:
            raise RuntimeError("Computed crop is empty.")

        cropped = cv2.resize(
            cropped,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_CUBIC,
        )
        return cropped

    def prepare_aligned_face(
        self,
        image_path: str | Path,
    ) -> CropResult:
        image_bgr = self.load_image_bgr(image_path)
        face_rect, _ = self.detect_largest_face_and_landmarks(image_bgr)

        cropped = self.crop_around_face(image_bgr, face_rect)
        cropped_face_rect, cropped_landmarks = self.detect_largest_face_and_landmarks(cropped)

        bbox = (
            cropped_face_rect.left(),
            cropped_face_rect.top(),
            cropped_face_rect.right(),
            cropped_face_rect.bottom(),
        )

        return CropResult(
            image_bgr=cropped,
            landmarks=cropped_landmarks,
            bbox=bbox,
        )

    def create_landmark_face_mask(
        self,
        image_bgr: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        """
        Landmark-based full-face mask.

        Uses:
        - jawline
        - eyebrows
        - nose
        - eyes
        - mouth

        This is still landmark-based, but stronger and more complete than
        the original jaw+eyebrow convex hull.
        """
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        jaw = landmarks[0:17]
        brows = landmarks[17:27]
        nose = landmarks[27:36]
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        mouth = landmarks[48:68]

        face_points = np.concatenate(
            [jaw, brows, nose, left_eye, right_eye, mouth],
            axis=0,
        )

        hull = cv2.convexHull(face_points.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)

        # Slightly strengthen mask around the center-face area.
        # This stays landmark-based while making blending less brittle.
        nose_hull = cv2.convexHull(nose.astype(np.int32))
        mouth_hull = cv2.convexHull(mouth.astype(np.int32))
        cv2.fillConvexPoly(mask, nose_hull, 255)
        cv2.fillConvexPoly(mask, mouth_hull, 255)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        blur_k = self.mask_blur_kernel
        if blur_k % 2 == 0:
            blur_k += 1
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

        return mask

    @staticmethod
    def merge_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
        """
        Stable merged mask for swap pipeline.

        Replaces the older Laplacian-mask blending.
        For masks, union is more principled and easier to defend.
        """
        merged = cv2.max(mask_a, mask_b)
        merged = cv2.GaussianBlur(merged, (21, 21), 0)
        return merged

    def generate_pair_masks(
        self,
        source_image_path: str | Path,
        target_image_path: str | Path,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            source_aligned_bgr,
            target_aligned_bgr,
            source_mask,
            target_mask,
            merged_mask
        """
        src = self.prepare_aligned_face(source_image_path)
        tgt = self.prepare_aligned_face(target_image_path)

        source_mask = self.create_landmark_face_mask(src.image_bgr, src.landmarks)
        target_mask = self.create_landmark_face_mask(tgt.image_bgr, tgt.landmarks)
        merged_mask = self.merge_masks(source_mask, target_mask)

        return (
            src.image_bgr,
            tgt.image_bgr,
            source_mask,
            target_mask,
            merged_mask,
        )

    def run_from_csv(
        self,
        csv_path: str | Path,
        source_dir: str | Path,
        target_dir: str | Path,
        output_dir: str | Path,
    ) -> Path:
        """
        Output layout:

        output_dir/
          align/
            source/<pair_key>.png
            target/<pair_key>.png
          mask/
            source/<pair_key>.png
            target/<pair_key>.png
            merged/<pair_key>.png
          image_pairs_with_masks.csv
        """
        csv_path = Path(csv_path)
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        output_dir = Path(output_dir)

        align_source_dir = output_dir / "align" / "source"
        align_target_dir = output_dir / "align" / "target"
        mask_source_dir = output_dir / "mask" / "source"
        mask_target_dir = output_dir / "mask" / "target"
        mask_merged_dir = output_dir / "mask" / "merged"

        for d in [
            align_source_dir,
            align_target_dir,
            mask_source_dir,
            mask_target_dir,
            mask_merged_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        pairs = self.load_pairs_from_csv(csv_path)

        output_rows = []

        for pair in tqdm(pairs, desc="Generating landmark-based masks"):
            pair_name = self.pair_key(pair.source_image, pair.target_image)

            source_path = source_dir / pair.source_image
            target_path = target_dir / pair.target_image

            if not source_path.exists():
                print(f"[WARN] Missing source image: {source_path}")
                continue
            if not target_path.exists():
                print(f"[WARN] Missing target image: {target_path}")
                continue

            try:
                (
                    source_aligned,
                    target_aligned,
                    source_mask,
                    target_mask,
                    merged_mask,
                ) = self.generate_pair_masks(source_path, target_path)

                source_align_out = align_source_dir / f"{pair_name}.png"
                target_align_out = align_target_dir / f"{pair_name}.png"
                source_mask_out = mask_source_dir / f"{pair_name}.png"
                target_mask_out = mask_target_dir / f"{pair_name}.png"
                merged_mask_out = mask_merged_dir / f"{pair_name}.png"

                cv2.imwrite(str(source_align_out), source_aligned)
                cv2.imwrite(str(target_align_out), target_aligned)
                cv2.imwrite(str(source_mask_out), source_mask)
                cv2.imwrite(str(target_mask_out), target_mask)
                cv2.imwrite(str(merged_mask_out), merged_mask)

                output_rows.append(
                    {
                        "source_image": pair.source_image,
                        "target_image": pair.target_image,
                        "source_aligned": str(source_align_out.as_posix()),
                        "target_aligned": str(target_align_out.as_posix()),
                        "source_mask": str(source_mask_out.as_posix()),
                        "target_mask": str(target_mask_out.as_posix()),
                        "merged_mask": str(merged_mask_out.as_posix()),
                    }
                )

            except Exception as e:
                print(f"[WARN] Failed for pair {pair_name}: {e}")

        output_csv = output_dir / "image_pairs_with_masks.csv"
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "source_image",
                    "target_image",
                    "source_aligned",
                    "target_aligned",
                    "source_mask",
                    "target_mask",
                    "merged_mask",
                ],
            )
            writer.writeheader()
            writer.writerows(output_rows)

        return output_csv