from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

from .pairs import EvalPair, existing_path


IMAGE_EXTENSIONS = {
    ".bmp", ".jpg", ".jpeg", ".pgm", ".png", ".ppm", ".tif", ".tiff", ".webp",
}


def natural_key(value) -> list:
    parts = re.split(r"(-?\d+)", str(value))
    return [int(part) if re.fullmatch(r"-?\d+", part) else part.lower() for part in parts]


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "model"


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def usable_image(path: Path) -> bool:
    name = path.name.lower()
    return is_image(path) and "combined" not in name


@dataclass(frozen=True)
class CandidateImage:
    path: Path
    candidate_id: str
    metadata: dict[str, str] = field(default_factory=dict)


class OutputResolver:
    resolver_type = "base"

    def __init__(self, root: Union[Path, str], name: Optional[str] = None):
        self.root = Path(root)
        self.name = safe_name(name or self.root.name)

    def candidates_for(self, pair: EvalPair) -> list[CandidateImage]:
        raise NotImplementedError

    def count_pairs(self, pairs: Iterable[EvalPair]) -> tuple[int, int]:
        total = 0
        with_candidates = 0
        for pair in pairs:
            total += 1
            if self.candidates_for(pair):
                with_candidates += 1
        return with_candidates, total


class DirectCsvResolver(OutputResolver):
    resolver_type = "direct-csv"

    def __init__(self, pairs: list[EvalPair], name: str = "csv_outputs"):
        super().__init__(Path("."), name=name)
        self._by_index: dict[int, list[CandidateImage]] = {}
        for pair in pairs:
            output_path = existing_path(pair.output_file)
            self._by_index[pair.index] = [
                CandidateImage(
                    path=output_path,
                    candidate_id=output_path.stem or str(pair.index),
                    metadata={"csv_row": str(pair.index)},
                )
            ]

    def candidates_for(self, pair: EvalPair) -> list[CandidateImage]:
        return self._by_index.get(pair.index, [])


def detect_resolvers(output: Union[Path, str], pairs: list[EvalPair]):
    raise RuntimeError(
        "Directory-based resolver discovery is disabled in this CSV-only rewrite. "
        "Use DirectCsvResolver via eval_models.cli with --pairs-csv."
    )
