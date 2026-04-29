from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union


@dataclass(frozen=True)
class EvalPair:
    index: int
    source_image: str
    target_image: str
    output_image: str

    @property
    def source_file(self) -> Path:
        return Path(self.source_image)

    @property
    def target_file(self) -> Path:
        return Path(self.target_image)

    @property
    def output_file(self) -> Path:
        return Path(self.output_image)

    @property
    def source_id(self) -> str:
        return self.source_file.stem

    @property
    def target_id(self) -> str:
        return self.target_file.stem

    @property
    def pair_key(self) -> str:
        return f"{self.source_id}__{self.target_id}"

    def source_reference(self, reference: str = "original") -> Path:
        return self.source_file

    def target_reference(self, reference: str = "original") -> Path:
        return self.target_file


class PairImageReader:
    """Path reader for one pair and one model-output resolver.

    This is the small adapter layer that hides the model-specific output layout.
    It returns source/target references, masks, and all generated candidates for
    a given pair.
    """

    def __init__(self, pair: EvalPair, resolver, reference: str = "align"):
        self.pair = pair
        self.resolver = resolver
        self.reference = reference

    @property
    def source_path(self) -> Path:
        return self.pair.source_reference(self.reference)

    @property
    def target_path(self) -> Path:
        return self.pair.target_reference(self.reference)

    @property
    def source_mask_path(self) -> Path:
        return self.pair.source_mask_file

    @property
    def target_mask_path(self) -> Path:
        return self.pair.target_mask_file

    def generated_candidates(self):
        return self.resolver.candidates_for(self.pair)

    def first_candidate(self):
        candidates = self.generated_candidates()
        return candidates[0] if candidates else None


def read_pairs_csv(
    pairs_csv: Union[Path, str],
    data_root: Optional[Union[Path, str]] = None,
    limit: Optional[int] = None,
) -> list[EvalPair]:
    pairs_csv = Path(pairs_csv)

    pairs: list[EvalPair] = []
    with pairs_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_image", "target_image", "output_image"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{pairs_csv} is missing columns: {sorted(missing)}")

        for row_index, row in enumerate(reader):
            if limit is not None and row_index >= limit:
                break

            pairs.append(
                EvalPair(
                    index=row_index,
                    source_image=row["source_image"],
                    target_image=row["target_image"],
                    output_image=row["output_image"],
                )
            )

    return pairs


def pair_lookup(pairs: Iterable[EvalPair]) -> dict[str, EvalPair]:
    return {pair.pair_key: pair for pair in pairs}
