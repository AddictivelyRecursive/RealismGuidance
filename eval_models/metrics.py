from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .materialize import MaterializedSet


@dataclass
class MetricResult:
    name: str
    command: List[str]
    returncode: int
    stdout_path: Path
    stderr_path: Path
    parsed: Dict[str, Any] = field(default_factory=dict)


class RefaceMetricRunner:
    """Shell wrapper around REFace/eval_tool metric scripts."""

    def __init__(
        self,
        reface_root: Union[Path, str],
        python: str = "python",
        device: str = "cuda",
        cuda_visible_devices: Optional[str] = None,
        batch_size: int = 50,
        id_batch_size: int = 1,
        id_dataset: str = "ffhq",
        keep_going: bool = False,
    ):
        self.reface_root = Path(reface_root)
        self.python = python
        self.device = device
        self.cuda_visible_devices = cuda_visible_devices
        self.batch_size = batch_size
        self.id_batch_size = id_batch_size
        self.id_dataset = id_dataset
        self.keep_going = keep_going

    def run_metrics(
        self,
        stage: MaterializedSet,
        result_dir: Union[Path, str],
        metrics: List[str],
        fid_reference: Optional[Union[Path, str]] = None,
    ) -> Dict[str, MetricResult]:
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, MetricResult] = {}
        for metric in metrics:
            command = self._command_for(metric, stage, fid_reference=fid_reference)
            result = self._run(metric, command, result_dir)
            results[metric] = result
            if result.returncode != 0 and not self.keep_going:
                raise RuntimeError(
                    f"{metric} failed with return code {result.returncode}. "
                    f"See {result.stdout_path} and {result.stderr_path}."
                )

        json_path = result_dir / "metrics.json"
        with json_path.open("w") as handle:
            json.dump(
                {
                    metric: {
                        **asdict(result),
                        "stdout_path": str(result.stdout_path),
                        "stderr_path": str(result.stderr_path),
                    }
                    for metric, result in results.items()
                },
                handle,
                indent=2,
            )
        return results

    def _command_for(
        self,
        metric: str,
        stage: MaterializedSet,
        fid_reference: Optional[Union[Path, str]] = None,
    ) -> List[str]:
        metric = metric.lower()
        if metric == "fid":
            reference = Path(fid_reference) if fid_reference else stage.target_dir
            return [
                self.python,
                "eval_tool/fid/fid_score.py",
                "--device",
                self.device,
                "--batch-size",
                str(self.batch_size),
                str(reference),
                str(stage.generated_dir),
            ]
        if metric == "id":
            return [
                self.python,
                "eval_tool/ID_retrieval/ID_retrieval.py",
                "--device",
                self.device,
                "--batch-size",
                str(self.id_batch_size),
                str(stage.source_dir),
                str(stage.generated_dir),
                str(stage.source_mask_dir),
                str(stage.target_mask_dir),
                "--dataset",
                self.id_dataset,
                "--print_sim",
                "True",
                "--arcface",
                "True",
            ]
        if metric == "pose":
            return [
                self.python,
                "eval_tool/Pose/pose_compare.py",
                "--device",
                self.device,
                "--batch-size",
                str(self.batch_size),
                str(stage.target_dir),
                str(stage.generated_dir),
            ]
        if metric == "expression":
            return [
                self.python,
                "eval_tool/Expression/expression_compare_face_recon.py",
                "--device",
                self.device,
                "--batch-size",
                str(self.batch_size),
                str(stage.target_dir),
                str(stage.generated_dir),
                "--print_sim",
                "True",
            ]
        raise ValueError(f"Unknown metric: {metric}")

    def _run(self, metric: str, command: list[str], result_dir: Path) -> MetricResult:
        env = os.environ.copy()
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices

        completed = subprocess.run(
            command,
            cwd=self.reface_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        stdout_path = result_dir / f"{metric}.stdout.txt"
        stderr_path = result_dir / f"{metric}.stderr.txt"
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        return MetricResult(
            name=metric,
            command=command,
            returncode=completed.returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            parsed=parse_metric_output(metric, completed.stdout),
        )


def parse_metric_output(metric: str, stdout: str) -> Dict[str, Any]:
    metric = metric.lower()
    parsed: Dict[str, Any] = {}
    patterns = {
        "fid": [("fid", r"FID:\s*([-+0-9.eE]+)")],
        "id": [
            ("top1", r"Top-1 accuracy:\s*([-+0-9.eE]+)%"),
            ("top5", r"Top-5 accuracy:\s*([-+0-9.eE]+)%"),
            ("mean_id", r"Mean ID feat:\s*([-+0-9.eE]+)"),
        ],
        "pose": [("pose", r"Pose_value:\s*([-+0-9.eE]+)")],
        "expression": [("expression", r"Expression_value:\s*([-+0-9.eE]+)")],
    }
    for key, pattern in patterns.get(metric, []):
        match = re.search(pattern, stdout)
        if match:
            parsed[key] = float(match.group(1))

    similarities = parse_similarity_lines(stdout)
    if similarities:
        parsed["similarities"] = similarities
    return parsed


def parse_similarity_lines(stdout: str) -> List[float]:
    values = []
    for line in stdout.splitlines():
        match = re.match(r"\s*([0-9]+)\s*:\s*([-+0-9.eE]+)\s*$", line)
        if match:
            values.append((int(match.group(1)), float(match.group(2))))
    values.sort(key=lambda item: item[0])
    return [value for _, value in values]
