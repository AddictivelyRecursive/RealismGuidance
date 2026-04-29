from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .pairs import EvalPair, PairImageReader
from .resolvers import CandidateImage


@dataclass(frozen=True)
class MaterializedSet:
    root: Path
    source_dir: Path
    target_dir: Path
    source_mask_dir: Path
    target_mask_dir: Path
    generated_dir: Path
    manifest_csv: Path


def materialize_pairs(
    pairs: list[EvalPair],
    resolver,
    selected_candidates: dict[int, CandidateImage],
    destination: Union[Path, str],
    reference: str = "align",
    link_mode: str = "symlink",
    overwrite: bool = False,
) -> MaterializedSet:
    """Create REFace-compatible directories for a chosen candidate per pair."""

    destination = Path(destination)
    _prepare_directory(destination, overwrite=overwrite)

    source_dir = destination / "source"
    target_dir = destination / "target"
    source_mask_dir = destination / "source_mask"
    target_mask_dir = destination / "target_mask"
    generated_dir = destination / "generated"
    for directory in [source_dir, target_dir, source_mask_dir, target_mask_dir, generated_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = destination / "manifest.csv"
    rows = []
    eval_index = 0
    for pair in pairs:
        candidate = selected_candidates.get(pair.index)
        if candidate is None:
            continue

        reader = PairImageReader(pair, resolver, reference=reference)
        _require_file(reader.source_path, f"source reference for {pair.pair_key}")
        _require_file(reader.target_path, f"target reference for {pair.pair_key}")
        _require_file(candidate.path, f"generated candidate for {pair.pair_key}")

        eval_name = f"{eval_index:05d}"
        staged_source = _stage_file(reader.source_path, source_dir / f"{eval_name}{reader.source_path.suffix}", link_mode)
        staged_target = _stage_file(reader.target_path, target_dir / f"{eval_name}{reader.target_path.suffix}", link_mode)
        staged_generated = _stage_file(
            candidate.path,
            generated_dir / f"{eval_name}{candidate.path.suffix}",
            link_mode,
        )

        rows.append(
            {
                "eval_index": eval_index,
                "pair_index": pair.index,
                "pair_key": pair.pair_key,
                "source_image": pair.source_image,
                "target_image": pair.target_image,
                "candidate_id": candidate.candidate_id,
                "generated_path": str(candidate.path),
                "staged_source": str(staged_source),
                "staged_target": str(staged_target),
                "staged_generated": str(staged_generated),
            }
        )
        eval_index += 1

    if not rows:
        raise RuntimeError(f"No generated images were staged for {resolver.name}")

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return MaterializedSet(
        root=destination,
        source_dir=source_dir,
        target_dir=target_dir,
        source_mask_dir=source_mask_dir,
        target_mask_dir=target_mask_dir,
        generated_dir=generated_dir,
        manifest_csv=manifest_path,
    )


def first_candidates(pairs: list[EvalPair], resolver) -> dict[int, CandidateImage]:
    selected: dict[int, CandidateImage] = {}
    for pair in pairs:
        candidates = resolver.candidates_for(pair)
        if candidates:
            selected[pair.index] = candidates[0]
    return selected


def candidates_by_slot(pairs: list[EvalPair], resolver, slot: int) -> dict[int, CandidateImage]:
    selected: dict[int, CandidateImage] = {}
    for pair in pairs:
        candidates = resolver.candidates_for(pair)
        if len(candidates) > slot:
            selected[pair.index] = candidates[slot]
    return selected


def max_candidate_count(pairs: list[EvalPair], resolver) -> int:
    max_count = 0
    for pair in pairs:
        max_count = max(max_count, len(resolver.candidates_for(pair)))
    return max_count


def _prepare_directory(destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"{destination} already exists; pass --overwrite or use a new --run-name")
        if destination.resolve() == Path("/"):
            raise ValueError("Refusing to remove filesystem root")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)


def _stage_file(src: Path, dst: Path, link_mode: str) -> Path:
    if link_mode not in {"symlink", "copy"}:
        raise ValueError(f"Unsupported link mode: {link_mode}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)
    return dst


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
