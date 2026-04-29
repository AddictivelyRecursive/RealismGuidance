from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .materialize import (
    candidates_by_slot,
    first_candidates,
    materialize_pairs,
    max_candidate_count,
)
from .metrics import RefaceMetricRunner
from .pairs import EvalPair, read_pairs_csv
from .resolvers import CandidateImage, DirectCsvResolver, safe_name


DEFAULT_PAIRS_CSV = "output/checkpoints/batch/2026-04-21-23-23-20/generated_samples_paths.csv"
DEFAULT_REFACE_ROOT = "/home/users/multicog/dwij22/REFace"
DEFAULT_OUTPUT_DIR = "eval_runs"
ALL_METRICS = ["fid", "id", "pose", "expression"]
DEFAULT_METRICS = ["fid", "pose", "expression"]


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if "id" in args.metrics and args.selection == "best-id":
        pass
    elif args.selection == "best-id":
        raise ValueError("--selection best-id needs the id metric/masks; use --selection first for CSV-only evaluation")

    if "id" in args.metrics:
        raise ValueError(
            "The id metric in the REFace script requires source/target masks. "
            "This CSV-only rewrite evaluates fid/pose/expression. Remove 'id' from --metrics."
        )

    pairs_csv = Path(args.pairs_csv)
    data_root = Path(args.data_root) if args.data_root else None
    pairs = read_pairs_csv(pairs_csv, data_root=data_root, limit=args.limit)

    resolver = DirectCsvResolver(pairs, name=args.model_name or pairs_csv.stem)
    resolvers = [resolver]

    summaries = [summarize_resolver(resolver, pairs) for resolver in resolvers]
    print_discovery(summaries)
    if args.dry_run:
        return 0

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_dir) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    runner = RefaceMetricRunner(
        reface_root=args.reface_root,
        python=args.python,
        device=args.device,
        cuda_visible_devices=args.cuda_visible_devices,
        batch_size=args.batch_size,
        id_batch_size=args.id_batch_size,
        id_dataset=args.id_dataset,
        keep_going=args.keep_going,
    )

    run_summary = {
        "run_root": str(run_root),
        "pairs_csv": str(pairs_csv),
        "data_root": str(data_root) if data_root else None,
        "models": {},
    }

    for resolver in resolvers:
        print(f"\n== {resolver.name} ==")
        model_root = run_root / resolver.name
        selection = resolve_selection_mode(args.selection, resolver, pairs, args.prepare_only)
        print(f"selection: {selection}")

        if selection == "best-id":
            selected = select_best_by_id(
                pairs=pairs,
                resolver=resolver,
                model_root=model_root,
                runner=runner,
                reference=args.reference,
                link_mode=args.link_mode,
                overwrite=args.overwrite,
            )
        else:
            selected = first_candidates(pairs, resolver)

        selected_stage = materialize_pairs(
            pairs=pairs,
            resolver=resolver,
            selected_candidates=selected,
            destination=model_root / "selected",
            reference=args.reference,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
        )
        print(f"staged: {selected_stage.root}")

        model_summary = {
            "resolver_type": resolver.resolver_type,
            "selection": selection,
            "selected_pairs": len(selected),
            "stage": str(selected_stage.root),
            "manifest": str(selected_stage.manifest_csv),
        }

        if not args.prepare_only:
            result_dir = model_root / "metrics"
            results = runner.run_metrics(
                selected_stage,
                result_dir=result_dir,
                metrics=args.metrics,
                fid_reference=args.fid_reference,
            )
            model_summary["metrics_dir"] = str(result_dir)
            model_summary["metrics"] = {name: result.parsed for name, result in results.items()}
            print_metric_summary(results)

        run_summary["models"][resolver.name] = model_summary

    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2))
    print(f"\nsummary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate face-swap outputs from a CSV containing source_image,target_image,output_image paths."
    )
    parser.add_argument("--pairs-csv", default=DEFAULT_PAIRS_CSV)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional prefix for relative paths inside the CSV. Absolute CSV paths are used as-is.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--reface-root", default=DEFAULT_REFACE_ROOT)
    parser.add_argument("--python", default="python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--reference", choices=["original"], default="original")
    parser.add_argument(
        "--selection",
        choices=["auto", "first", "best-id"],
        default="auto",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=ALL_METRICS,
        default=DEFAULT_METRICS,
    )
    parser.add_argument(
        "--fid-reference",
        default=None,
        help="Optional real-image directory for FID. Defaults to staged target images.",
    )
    parser.add_argument("--id-dataset", default="ffhq")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--id-batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def summarize_resolver(resolver, pairs: list[EvalPair]) -> dict:
    counts = []
    missing = []
    for pair in pairs:
        candidate_count = len(resolver.candidates_for(pair))
        counts.append(candidate_count)
        if candidate_count == 0 and len(missing) < 8:
            missing.append(pair.pair_key)
    return {
        "name": resolver.name,
        "type": resolver.resolver_type,
        "root": str(resolver.root),
        "pairs": len(pairs),
        "found": sum(1 for count in counts if count > 0),
        "missing_examples": missing,
        "min_candidates": min(counts) if counts else 0,
        "max_candidates": max(counts) if counts else 0,
    }


def print_discovery(summaries: list[dict]) -> None:
    print("Discovered output sets:")
    for summary in summaries:
        print(
            f"- {summary['name']} ({summary['type']}): "
            f"{summary['found']}/{summary['pairs']} pairs, "
            f"candidates {summary['min_candidates']}..{summary['max_candidates']}"
        )
        if summary["missing_examples"]:
            print(f"  missing examples: {', '.join(summary['missing_examples'])}")


def resolve_selection_mode(selection: str, resolver, pairs: list[EvalPair], prepare_only: bool) -> str:
    if selection != "auto":
        if selection == "best-id" and prepare_only:
            raise ValueError("--selection best-id needs metric execution; remove --prepare-only")
        return selection
    return "first"


def select_best_by_id(
    pairs: list[EvalPair],
    resolver,
    model_root: Path,
    runner: RefaceMetricRunner,
    reference: str,
    link_mode: str,
    overwrite: bool,
) -> dict[int, CandidateImage]:
    sweep_root = model_root / "candidate_sweep"
    max_slots = max_candidate_count(pairs, resolver)
    if max_slots <= 1:
        return first_candidates(pairs, resolver)

    scores: dict[int, dict[int, float]] = {}
    selected_by_slot: dict[int, dict[int, CandidateImage]] = {}
    for slot in range(max_slots):
        slot_candidates = candidates_by_slot(pairs, resolver, slot)
        if not slot_candidates:
            continue
        selected_by_slot[slot] = slot_candidates
        stage = materialize_pairs(
            pairs=pairs,
            resolver=resolver,
            selected_candidates=slot_candidates,
            destination=sweep_root / f"slot_{slot:02d}",
            reference=reference,
            link_mode=link_mode,
            overwrite=overwrite,
        )
        result_dir = sweep_root / f"slot_{slot:02d}_metrics"
        print(f"scoring candidate slot {slot + 1}/{max_slots}: {stage.generated_dir}")
        results = runner.run_metrics(stage, result_dir=result_dir, metrics=["id"])
        similarities = results["id"].parsed.get("similarities", [])
        if not isinstance(similarities, list):
            similarities = []
        manifest = read_manifest(stage.manifest_csv)
        for eval_index, row in enumerate(manifest):
            if eval_index >= len(similarities):
                continue
            pair_index = int(row["pair_index"])
            scores.setdefault(pair_index, {})[slot] = float(similarities[eval_index])

    selected: dict[int, CandidateImage] = {}
    for pair in pairs:
        pair_scores = scores.get(pair.index)
        if not pair_scores:
            first = resolver.candidates_for(pair)
            if first:
                selected[pair.index] = first[0]
            continue
        best_slot = max(pair_scores, key=lambda slot: pair_scores[slot])
        selected[pair.index] = selected_by_slot[best_slot][pair.index]

    write_selection_csv(model_root / "best_id_selection.csv", pairs, resolver, scores, selected)
    return selected


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_selection_csv(
    path: Path,
    pairs: list[EvalPair],
    resolver,
    scores: dict[int, dict[int, float]],
    selected: dict[int, CandidateImage],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_slots = max_candidate_count(pairs, resolver)
    fieldnames = [
        "pair_index",
        "pair_key",
        "source_image",
        "target_image",
        "selected_candidate_id",
        "selected_path",
    ] + [f"slot_{slot:02d}_id_score" for slot in range(max_slots)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            candidate = selected.get(pair.index)
            row = {
                "pair_index": pair.index,
                "pair_key": pair.pair_key,
                "source_image": pair.source_image,
                "target_image": pair.target_image,
                "selected_candidate_id": candidate.candidate_id if candidate else "",
                "selected_path": str(candidate.path) if candidate else "",
            }
            for slot in range(max_slots):
                row[f"slot_{slot:02d}_id_score"] = scores.get(pair.index, {}).get(slot, "")
            writer.writerow(row)


def print_metric_summary(results) -> None:
    for metric, result in results.items():
        parsed = {key: value for key, value in result.parsed.items() if key != "similarities"}
        if parsed:
            print(f"{metric}: {parsed}")
        else:
            print(f"{metric}: returncode={result.returncode}")


if __name__ == "__main__":
    raise SystemExit(main())
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .materialize import (
    candidates_by_slot,
    first_candidates,
    materialize_pairs,
    max_candidate_count,
)
from .metrics import RefaceMetricRunner
from .pairs import EvalPair, read_pairs_csv
from .resolvers import CandidateImage, DirectCsvResolver, safe_name


DEFAULT_PAIRS_CSV = "output/checkpoints/batch/2026-04-21-23-23-20/generated_samples_paths.csv"
DEFAULT_REFACE_ROOT = "/home/users/multicog/dwij22/REFace"
DEFAULT_OUTPUT_DIR = "eval_runs"
ALL_METRICS = ["fid", "id", "pose", "expression"]
DEFAULT_METRICS = ["fid", "pose", "expression"]


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if "id" in args.metrics and args.selection == "best-id":
        pass
    elif args.selection == "best-id":
        raise ValueError("--selection best-id needs the id metric/masks; use --selection first for CSV-only evaluation")

    if "id" in args.metrics:
        raise ValueError(
            "The id metric in the REFace script requires source/target masks. "
            "This CSV-only rewrite evaluates fid/pose/expression. Remove 'id' from --metrics."
        )

    pairs_csv = Path(args.pairs_csv)
    data_root = Path(args.data_root) if args.data_root else None
    pairs = read_pairs_csv(pairs_csv, data_root=data_root, limit=args.limit)

    resolver = DirectCsvResolver(pairs, name=args.model_name or pairs_csv.stem)
    resolvers = [resolver]

    summaries = [summarize_resolver(resolver, pairs) for resolver in resolvers]
    print_discovery(summaries)
    if args.dry_run:
        return 0

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_dir) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    runner = RefaceMetricRunner(
        reface_root=args.reface_root,
        python=args.python,
        device=args.device,
        cuda_visible_devices=args.cuda_visible_devices,
        batch_size=args.batch_size,
        id_batch_size=args.id_batch_size,
        id_dataset=args.id_dataset,
        keep_going=args.keep_going,
    )

    run_summary = {
        "run_root": str(run_root),
        "pairs_csv": str(pairs_csv),
        "data_root": str(data_root) if data_root else None,
        "models": {},
    }

    for resolver in resolvers:
        print(f"\n== {resolver.name} ==")
        model_root = run_root / resolver.name
        selection = resolve_selection_mode(args.selection, resolver, pairs, args.prepare_only)
        print(f"selection: {selection}")

        if selection == "best-id":
            selected = select_best_by_id(
                pairs=pairs,
                resolver=resolver,
                model_root=model_root,
                runner=runner,
                reference=args.reference,
                link_mode=args.link_mode,
                overwrite=args.overwrite,
            )
        else:
            selected = first_candidates(pairs, resolver)

        selected_stage = materialize_pairs(
            pairs=pairs,
            resolver=resolver,
            selected_candidates=selected,
            destination=model_root / "selected",
            reference=args.reference,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
        )
        print(f"staged: {selected_stage.root}")

        model_summary = {
            "resolver_type": resolver.resolver_type,
            "selection": selection,
            "selected_pairs": len(selected),
            "stage": str(selected_stage.root),
            "manifest": str(selected_stage.manifest_csv),
        }

        if not args.prepare_only:
            result_dir = model_root / "metrics"
            results = runner.run_metrics(
                selected_stage,
                result_dir=result_dir,
                metrics=args.metrics,
                fid_reference=args.fid_reference,
            )
            model_summary["metrics_dir"] = str(result_dir)
            model_summary["metrics"] = {name: result.parsed for name, result in results.items()}
            print_metric_summary(results)

        run_summary["models"][resolver.name] = model_summary

    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2))
    print(f"\nsummary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate face-swap outputs from a CSV containing source_image,target_image,output_image paths."
    )
    parser.add_argument("--pairs-csv", default=DEFAULT_PAIRS_CSV)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional prefix for relative paths inside the CSV. Absolute CSV paths are used as-is.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--reface-root", default=DEFAULT_REFACE_ROOT)
    parser.add_argument("--python", default="python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--reference", choices=["original"], default="original")
    parser.add_argument(
        "--selection",
        choices=["auto", "first", "best-id"],
        default="auto",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=ALL_METRICS,
        default=DEFAULT_METRICS,
    )
    parser.add_argument(
        "--fid-reference",
        default=None,
        help="Optional real-image directory for FID. Defaults to staged target images.",
    )
    parser.add_argument("--id-dataset", default="ffhq")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--id-batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def summarize_resolver(resolver, pairs: list[EvalPair]) -> dict:
    counts = []
    missing = []
    for pair in pairs:
        candidate_count = len(resolver.candidates_for(pair))
        counts.append(candidate_count)
        if candidate_count == 0 and len(missing) < 8:
            missing.append(pair.pair_key)
    return {
        "name": resolver.name,
        "type": resolver.resolver_type,
        "root": str(resolver.root),
        "pairs": len(pairs),
        "found": sum(1 for count in counts if count > 0),
        "missing_examples": missing,
        "min_candidates": min(counts) if counts else 0,
        "max_candidates": max(counts) if counts else 0,
    }


def print_discovery(summaries: list[dict]) -> None:
    print("Discovered output sets:")
    for summary in summaries:
        print(
            f"- {summary['name']} ({summary['type']}): "
            f"{summary['found']}/{summary['pairs']} pairs, "
            f"candidates {summary['min_candidates']}..{summary['max_candidates']}"
        )
        if summary["missing_examples"]:
            print(f"  missing examples: {', '.join(summary['missing_examples'])}")


def resolve_selection_mode(selection: str, resolver, pairs: list[EvalPair], prepare_only: bool) -> str:
    if selection != "auto":
        if selection == "best-id" and prepare_only:
            raise ValueError("--selection best-id needs metric execution; remove --prepare-only")
        return selection
    return "first"


def select_best_by_id(
    pairs: list[EvalPair],
    resolver,
    model_root: Path,
    runner: RefaceMetricRunner,
    reference: str,
    link_mode: str,
    overwrite: bool,
) -> dict[int, CandidateImage]:
    sweep_root = model_root / "candidate_sweep"
    max_slots = max_candidate_count(pairs, resolver)
    if max_slots <= 1:
        return first_candidates(pairs, resolver)

    scores: dict[int, dict[int, float]] = {}
    selected_by_slot: dict[int, dict[int, CandidateImage]] = {}
    for slot in range(max_slots):
        slot_candidates = candidates_by_slot(pairs, resolver, slot)
        if not slot_candidates:
            continue
        selected_by_slot[slot] = slot_candidates
        stage = materialize_pairs(
            pairs=pairs,
            resolver=resolver,
            selected_candidates=slot_candidates,
            destination=sweep_root / f"slot_{slot:02d}",
            reference=reference,
            link_mode=link_mode,
            overwrite=overwrite,
        )
        result_dir = sweep_root / f"slot_{slot:02d}_metrics"
        print(f"scoring candidate slot {slot + 1}/{max_slots}: {stage.generated_dir}")
        results = runner.run_metrics(stage, result_dir=result_dir, metrics=["id"])
        similarities = results["id"].parsed.get("similarities", [])
        if not isinstance(similarities, list):
            similarities = []
        manifest = read_manifest(stage.manifest_csv)
        for eval_index, row in enumerate(manifest):
            if eval_index >= len(similarities):
                continue
            pair_index = int(row["pair_index"])
            scores.setdefault(pair_index, {})[slot] = float(similarities[eval_index])

    selected: dict[int, CandidateImage] = {}
    for pair in pairs:
        pair_scores = scores.get(pair.index)
        if not pair_scores:
            first = resolver.candidates_for(pair)
            if first:
                selected[pair.index] = first[0]
            continue
        best_slot = max(pair_scores, key=lambda slot: pair_scores[slot])
        selected[pair.index] = selected_by_slot[best_slot][pair.index]

    write_selection_csv(model_root / "best_id_selection.csv", pairs, resolver, scores, selected)
    return selected


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_selection_csv(
    path: Path,
    pairs: list[EvalPair],
    resolver,
    scores: dict[int, dict[int, float]],
    selected: dict[int, CandidateImage],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_slots = max_candidate_count(pairs, resolver)
    fieldnames = [
        "pair_index",
        "pair_key",
        "source_image",
        "target_image",
        "selected_candidate_id",
        "selected_path",
    ] + [f"slot_{slot:02d}_id_score" for slot in range(max_slots)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            candidate = selected.get(pair.index)
            row = {
                "pair_index": pair.index,
                "pair_key": pair.pair_key,
                "source_image": pair.source_image,
                "target_image": pair.target_image,
                "selected_candidate_id": candidate.candidate_id if candidate else "",
                "selected_path": str(candidate.path) if candidate else "",
            }
            for slot in range(max_slots):
                row[f"slot_{slot:02d}_id_score"] = scores.get(pair.index, {}).get(slot, "")
            writer.writerow(row)


def print_metric_summary(results) -> None:
    for metric, result in results.items():
        parsed = {key: value for key, value in result.parsed.items() if key != "similarities"}
        if parsed:
            print(f"{metric}: {parsed}")
        else:
            print(f"{metric}: returncode={result.returncode}")


if __name__ == "__main__":
    raise SystemExit(main())
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .materialize import (
    candidates_by_slot,
    first_candidates,
    materialize_pairs,
    max_candidate_count,
)
from .metrics import RefaceMetricRunner
from .pairs import EvalPair, read_pairs_csv
from .resolvers import CandidateImage, DirectCsvResolver, safe_name


DEFAULT_PAIRS_CSV = "output/checkpoints/batch/2026-04-21-23-23-20/generated_samples_paths.csv"
DEFAULT_REFACE_ROOT = "/home/users/multicog/dwij22/REFace"
DEFAULT_OUTPUT_DIR = "eval_runs"
ALL_METRICS = ["fid", "id", "pose", "expression"]
DEFAULT_METRICS = ["fid", "pose", "expression"]


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if "id" in args.metrics and args.selection == "best-id":
        pass
    elif args.selection == "best-id":
        raise ValueError("--selection best-id needs the id metric/masks; use --selection first for CSV-only evaluation")

    if "id" in args.metrics:
        raise ValueError(
            "The id metric in the REFace script requires source/target masks. "
            "This CSV-only rewrite evaluates fid/pose/expression. Remove 'id' from --metrics."
        )

    pairs_csv = Path(args.pairs_csv)
    data_root = Path(args.data_root) if args.data_root else None
    pairs = read_pairs_csv(pairs_csv, data_root=data_root, limit=args.limit)

    resolver = DirectCsvResolver(pairs, name=args.model_name or pairs_csv.stem)
    resolvers = [resolver]

    summaries = [summarize_resolver(resolver, pairs) for resolver in resolvers]
    print_discovery(summaries)
    if args.dry_run:
        return 0

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_dir) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    runner = RefaceMetricRunner(
        reface_root=args.reface_root,
        python=args.python,
        device=args.device,
        cuda_visible_devices=args.cuda_visible_devices,
        batch_size=args.batch_size,
        id_batch_size=args.id_batch_size,
        id_dataset=args.id_dataset,
        keep_going=args.keep_going,
    )

    run_summary = {
        "run_root": str(run_root),
        "pairs_csv": str(pairs_csv),
        "data_root": str(data_root) if data_root else None,
        "models": {},
    }

    for resolver in resolvers:
        print(f"\n== {resolver.name} ==")
        model_root = run_root / resolver.name
        selection = resolve_selection_mode(args.selection, resolver, pairs, args.prepare_only)
        print(f"selection: {selection}")

        if selection == "best-id":
            selected = select_best_by_id(
                pairs=pairs,
                resolver=resolver,
                model_root=model_root,
                runner=runner,
                reference=args.reference,
                link_mode=args.link_mode,
                overwrite=args.overwrite,
            )
        else:
            selected = first_candidates(pairs, resolver)

        selected_stage = materialize_pairs(
            pairs=pairs,
            resolver=resolver,
            selected_candidates=selected,
            destination=model_root / "selected",
            reference=args.reference,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
        )
        print(f"staged: {selected_stage.root}")

        model_summary = {
            "resolver_type": resolver.resolver_type,
            "selection": selection,
            "selected_pairs": len(selected),
            "stage": str(selected_stage.root),
            "manifest": str(selected_stage.manifest_csv),
        }

        if not args.prepare_only:
            result_dir = model_root / "metrics"
            results = runner.run_metrics(
                selected_stage,
                result_dir=result_dir,
                metrics=args.metrics,
                fid_reference=args.fid_reference,
            )
            model_summary["metrics_dir"] = str(result_dir)
            model_summary["metrics"] = {name: result.parsed for name, result in results.items()}
            print_metric_summary(results)

        run_summary["models"][resolver.name] = model_summary

    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2))
    print(f"\nsummary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate face-swap outputs from a CSV containing source_image,target_image,output_image paths."
    )
    parser.add_argument("--pairs-csv", default=DEFAULT_PAIRS_CSV)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional prefix for relative paths inside the CSV. Absolute CSV paths are used as-is.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--reface-root", default=DEFAULT_REFACE_ROOT)
    parser.add_argument("--python", default="python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--reference", choices=["original"], default="original")
    parser.add_argument(
        "--selection",
        choices=["auto", "first", "best-id"],
        default="auto",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=ALL_METRICS,
        default=DEFAULT_METRICS,
    )
    parser.add_argument(
        "--fid-reference",
        default=None,
        help="Optional real-image directory for FID. Defaults to staged target images.",
    )
    parser.add_argument("--id-dataset", default="ffhq")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--id-batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def summarize_resolver(resolver, pairs: list[EvalPair]) -> dict:
    counts = []
    missing = []
    for pair in pairs:
        candidate_count = len(resolver.candidates_for(pair))
        counts.append(candidate_count)
        if candidate_count == 0 and len(missing) < 8:
            missing.append(pair.pair_key)
    return {
        "name": resolver.name,
        "type": resolver.resolver_type,
        "root": str(resolver.root),
        "pairs": len(pairs),
        "found": sum(1 for count in counts if count > 0),
        "missing_examples": missing,
        "min_candidates": min(counts) if counts else 0,
        "max_candidates": max(counts) if counts else 0,
    }


def print_discovery(summaries: list[dict]) -> None:
    print("Discovered output sets:")
    for summary in summaries:
        print(
            f"- {summary['name']} ({summary['type']}): "
            f"{summary['found']}/{summary['pairs']} pairs, "
            f"candidates {summary['min_candidates']}..{summary['max_candidates']}"
        )
        if summary["missing_examples"]:
            print(f"  missing examples: {', '.join(summary['missing_examples'])}")


def resolve_selection_mode(selection: str, resolver, pairs: list[EvalPair], prepare_only: bool) -> str:
    if selection != "auto":
        if selection == "best-id" and prepare_only:
            raise ValueError("--selection best-id needs metric execution; remove --prepare-only")
        return selection
    return "first"


def select_best_by_id(
    pairs: list[EvalPair],
    resolver,
    model_root: Path,
    runner: RefaceMetricRunner,
    reference: str,
    link_mode: str,
    overwrite: bool,
) -> dict[int, CandidateImage]:
    sweep_root = model_root / "candidate_sweep"
    max_slots = max_candidate_count(pairs, resolver)
    if max_slots <= 1:
        return first_candidates(pairs, resolver)

    scores: dict[int, dict[int, float]] = {}
    selected_by_slot: dict[int, dict[int, CandidateImage]] = {}
    for slot in range(max_slots):
        slot_candidates = candidates_by_slot(pairs, resolver, slot)
        if not slot_candidates:
            continue
        selected_by_slot[slot] = slot_candidates
        stage = materialize_pairs(
            pairs=pairs,
            resolver=resolver,
            selected_candidates=slot_candidates,
            destination=sweep_root / f"slot_{slot:02d}",
            reference=reference,
            link_mode=link_mode,
            overwrite=overwrite,
        )
        result_dir = sweep_root / f"slot_{slot:02d}_metrics"
        print(f"scoring candidate slot {slot + 1}/{max_slots}: {stage.generated_dir}")
        results = runner.run_metrics(stage, result_dir=result_dir, metrics=["id"])
        similarities = results["id"].parsed.get("similarities", [])
        if not isinstance(similarities, list):
            similarities = []
        manifest = read_manifest(stage.manifest_csv)
        for eval_index, row in enumerate(manifest):
            if eval_index >= len(similarities):
                continue
            pair_index = int(row["pair_index"])
            scores.setdefault(pair_index, {})[slot] = float(similarities[eval_index])

    selected: dict[int, CandidateImage] = {}
    for pair in pairs:
        pair_scores = scores.get(pair.index)
        if not pair_scores:
            first = resolver.candidates_for(pair)
            if first:
                selected[pair.index] = first[0]
            continue
        best_slot = max(pair_scores, key=lambda slot: pair_scores[slot])
        selected[pair.index] = selected_by_slot[best_slot][pair.index]

    write_selection_csv(model_root / "best_id_selection.csv", pairs, resolver, scores, selected)
    return selected


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_selection_csv(
    path: Path,
    pairs: list[EvalPair],
    resolver,
    scores: dict[int, dict[int, float]],
    selected: dict[int, CandidateImage],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_slots = max_candidate_count(pairs, resolver)
    fieldnames = [
        "pair_index",
        "pair_key",
        "source_image",
        "target_image",
        "selected_candidate_id",
        "selected_path",
    ] + [f"slot_{slot:02d}_id_score" for slot in range(max_slots)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            candidate = selected.get(pair.index)
            row = {
                "pair_index": pair.index,
                "pair_key": pair.pair_key,
                "source_image": pair.source_image,
                "target_image": pair.target_image,
                "selected_candidate_id": candidate.candidate_id if candidate else "",
                "selected_path": str(candidate.path) if candidate else "",
            }
            for slot in range(max_slots):
                row[f"slot_{slot:02d}_id_score"] = scores.get(pair.index, {}).get(slot, "")
            writer.writerow(row)


def print_metric_summary(results) -> None:
    for metric, result in results.items():
        parsed = {key: value for key, value in result.parsed.items() if key != "similarities"}
        if parsed:
            print(f"{metric}: {parsed}")
        else:
            print(f"{metric}: returncode={result.returncode}")


if __name__ == "__main__":
    raise SystemExit(main())
