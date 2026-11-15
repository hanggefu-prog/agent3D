"""Conservative change-impact analysis for adaptive stage replay."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

from .model import EvaluationSpec, StageSpec


def _normalized_change(path: str) -> str:
    value = PurePosixPath(path)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError("changed file must be a safe repository-relative path: {}".format(path))
    return value.as_posix()


def _matches(path: str, pattern: str) -> bool:
    # Pure fnmatch treats ``**`` portably enough for repository-relative paths,
    # while the exact-prefix branch makes directory-wide contracts intuitive.
    prefix = pattern[:-3] if pattern.endswith("/**") else None
    return fnmatch.fnmatchcase(path, pattern) or (
        prefix is not None and (path == prefix or path.startswith(prefix + "/"))
    )


@dataclass(frozen=True)
class EvaluationPlan:
    policy: str
    changed_files: Tuple[str, ...]
    claimed_stages: Tuple[str, ...]
    directly_affected: Tuple[str, ...]
    rerun_stages: Tuple[str, ...]
    reused_stages: Tuple[str, ...]
    unknown_files: Tuple[str, ...]
    conservative_fallback: bool
    start_stage: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = 1
        return value


class ImpactResolver:
    """Map source changes to a trusted DAG; candidate claims can only widen it."""

    def __init__(self, evaluation: EvaluationSpec):
        self.evaluation = evaluation
        self._by_id = {stage.stage_id: stage for stage in evaluation.stages}
        self._order = self._topological_order(evaluation.stages)

    @staticmethod
    def _topological_order(stages: Sequence[StageSpec]) -> Tuple[str, ...]:
        by_id = {stage.stage_id: stage for stage in stages}
        ordered = []
        visited: Set[str] = set()

        def visit(stage_id: str) -> None:
            if stage_id in visited:
                return
            for dependency in by_id[stage_id].depends_on:
                visit(dependency)
            visited.add(stage_id)
            ordered.append(stage_id)

        for stage in stages:
            visit(stage.stage_id)
        return tuple(ordered)

    def _downstream(self, roots: Iterable[str]) -> Set[str]:
        result = set(roots)
        changed = True
        while changed:
            changed = False
            for stage in self.evaluation.stages:
                if stage.stage_id not in result and set(stage.depends_on) & result:
                    result.add(stage.stage_id)
                    changed = True
        return result

    def resolve(
        self,
        changed_files: Sequence[str],
        *,
        claimed_stages: Sequence[str] = (),
        policy: Optional[str] = None,
    ) -> EvaluationPlan:
        selected_policy = policy or self.evaluation.default_policy
        if selected_policy not in {"quick", "suffix", "final", "full"}:
            raise ValueError("unsupported evaluation policy: {}".format(selected_policy))
        changes = tuple(dict.fromkeys(_normalized_change(path) for path in changed_files))
        claims = tuple(dict.fromkeys(str(stage) for stage in claimed_stages))
        unknown_claims = set(claims) - set(self._by_id)
        if unknown_claims:
            raise ValueError("candidate claimed unknown stages: {}".format(sorted(unknown_claims)))

        direct: Set[str] = set(claims)
        unknown_files = []
        for path in changes:
            matched = {
                stage.stage_id
                for stage in self.evaluation.stages
                if any(_matches(path, pattern) for pattern in stage.implementation_paths)
            }
            if not matched:
                unknown_files.append(path)
            direct.update(matched)

        conservative = False
        if unknown_files:
            if self.evaluation.unknown_change_policy == "reject":
                raise ValueError(
                    "changed files are outside the stage dependency map: {}".format(unknown_files)
                )
            direct = set(self._order)
            conservative = True

        if selected_policy == "full":
            rerun = set(self._order)
        elif selected_policy == "quick":
            if direct:
                indices = [self._order.index(stage) for stage in direct]
                rerun = set(self._order[min(indices) : max(indices) + 1])
            else:
                rerun = set()
        else:
            rerun = self._downstream(direct)

        ordered_direct = tuple(stage for stage in self._order if stage in direct)
        ordered_rerun = tuple(stage for stage in self._order if stage in rerun)
        reused = tuple(stage for stage in self._order if stage not in rerun)
        return EvaluationPlan(
            policy=selected_policy,
            changed_files=changes,
            claimed_stages=claims,
            directly_affected=ordered_direct,
            rerun_stages=ordered_rerun,
            reused_stages=reused,
            unknown_files=tuple(unknown_files),
            conservative_fallback=conservative,
            start_stage=ordered_rerun[0] if ordered_rerun else None,
        )


def compute_stage_signature(
    *,
    stage_id: str,
    upstream_artifacts: Mapping[str, str],
    resolved_config: Mapping[str, Any],
    implementation_files: Sequence[Path],
    environment_identity: str,
    seed: int,
) -> str:
    """Hash the trusted inputs that decide whether a stage may be reused."""

    files = []
    for path in sorted((item.expanduser().resolve() for item in implementation_files), key=str):
        if not path.is_file() or path.is_symlink():
            raise ValueError("implementation input must be a regular file: {}".format(path))
        files.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload = {
        "stage_id": stage_id,
        "upstream_artifacts": dict(sorted(upstream_artifacts.items())),
        "resolved_config": resolved_config,
        "implementation_files": files,
        "environment_identity": environment_identity,
        "seed": int(seed),
    }
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
