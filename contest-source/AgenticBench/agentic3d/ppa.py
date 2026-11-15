"""Deterministic, version-neutral collection of global PPA evidence."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .contracts import STAGES
from .open3dbench import Open3DBenchAdapter


METRIC_ALIASES: Mapping[str, tuple[str, ...]] = {
    "hpwl": ("hpwl", "half_perimeter_wirelength"),
    "wirelength": ("wirelength", "total_wire_length", "routed_wirelength"),
    "overflow": ("overflow", "density_overflow", "congestion_overflow"),
    "max_density": ("max_density", "maximum_density", "peak_density"),
    "wns": ("wns", "worst_negative_slack"),
    "tns": ("tns", "total_negative_slack"),
    "power": ("power", "total_power"),
    "area": ("area", "design_area"),
    "drc": ("drc", "drc_count", "violations"),
    "vias": ("vias", "via_count", "hbt_count"),
    "skew": ("skew", "clock_skew"),
    "runtime": ("runtime", "elapsed", "elapsed_time"),
    "cut": ("cut", "cut_weight", "cut_count"),
    "balance": ("balance", "area_balance", "area_balance_error"),
}

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
def _text_alias(alias: str) -> str:
    return r"[\s_-]+".join(re.escape(part) for part in alias.split("_"))


TEXT_PATTERNS = {
    metric: re.compile(
        rf"(?i)\b(?:{'|'.join(_text_alias(alias) for alias in aliases)})\b"
        rf"[^0-9+\-.]{{0,40}}(?P<value>{NUMBER})"
    )
    for metric, aliases in METRIC_ALIASES.items()
}


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.fullmatch(rf"\s*({NUMBER})\s*", value)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _walk_json(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_json(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


def _canonical_metric(key: str) -> Optional[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    leaf = normalized.split("_")[-1]
    for metric, aliases in METRIC_ALIASES.items():
        if normalized in aliases or leaf in aliases:
            return metric
        if any(normalized.endswith("_" + alias) for alias in aliases):
            return metric
    return None


@dataclass(frozen=True)
class GlobalPPACollector:
    adapter: Open3DBenchAdapter
    max_files_per_stage: int = 80
    max_text_bytes: int = 2_000_000

    def _json_metrics(self, path: Path) -> List[Dict[str, Any]]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []
        metrics: List[Dict[str, Any]] = []
        for key, raw in _walk_json(value):
            metric = _canonical_metric(key)
            numeric = _number(raw)
            if metric is not None and numeric is not None:
                metrics.append({"metric": metric, "value": numeric, "field": key})
        return metrics

    def _csv_metrics(self, path: Path) -> List[Dict[str, Any]]:
        try:
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, UnicodeDecodeError, csv.Error):
            return []
        if not rows:
            return []
        metrics: List[Dict[str, Any]] = []
        for key, raw in rows[-1].items():
            metric = _canonical_metric(str(key))
            numeric = _number(raw)
            if metric is not None and numeric is not None:
                metrics.append({"metric": metric, "value": numeric, "field": key})
        return metrics

    def _text_metrics(self, path: Path) -> List[Dict[str, Any]]:
        try:
            size = path.stat().st_size
            with path.open("rb") as stream:
                if size > self.max_text_bytes:
                    stream.seek(-self.max_text_bytes, 2)
                raw = stream.read(self.max_text_bytes)
            text = raw.decode("utf-8", errors="replace")
        except OSError:
            return []
        metrics: List[Dict[str, Any]] = []
        for metric, pattern in TEXT_PATTERNS.items():
            matches = list(pattern.finditer(text))
            if matches:
                metrics.append(
                    {
                        "metric": metric,
                        "value": float(matches[-1].group("value")),
                        "field": "last_text_match",
                    }
                )
        return metrics

    def _metrics(self, path: Path) -> List[Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._json_metrics(path)
        if suffix == ".csv":
            return self._csv_metrics(path)
        if suffix in {".log", ".rpt", ".txt", ".out"}:
            return self._text_metrics(path)
        return []

    def collect(self) -> Dict[str, Any]:
        stage_reports: Dict[str, Any] = {}
        for stage in STAGES:
            artifacts = self.adapter.collect_artifacts(
                stage, limit=self.max_files_per_stage
            )
            observations: List[Dict[str, Any]] = []
            for artifact in artifacts:
                relative = Path(str(artifact["path"]))
                path = self.adapter.repository / relative
                for metric in self._metrics(path):
                    observations.append(
                        {
                            **metric,
                            "path": str(relative),
                            "mtime_ns": artifact["mtime_ns"],
                        }
                    )
            latest: Dict[str, Dict[str, Any]] = {}
            for observation in sorted(
                observations, key=lambda item: int(item["mtime_ns"])
            ):
                latest[str(observation["metric"])] = observation
            stage_reports[stage] = {
                "artifact_count": len(artifacts),
                "metric_observation_count": len(observations),
                "latest_metrics": latest,
                "source_artifacts": artifacts,
            }
        return {
            "repository": str(self.adapter.repository),
            "adapter": self.adapter.manifest.name,
            "stages": stage_reports,
            "limitations": [
                "Metrics are evidence inventory, not automatic proof of comparability.",
                "Global/Local Analyzers must verify design, seed, units, stage success, and correctness reports.",
            ],
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.collect(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path
