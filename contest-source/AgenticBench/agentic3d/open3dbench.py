"""Version-adaptive Open3DBench repository discovery.

All version-specific paths live in JSON adapter manifests.  Agents receive the
normalized context emitted here and therefore do not need a new harness when
Open3DBench renames or consolidates directories.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .contracts import STAGES, require_stage


class AdapterNotFoundError(ValueError):
    pass


@dataclass(frozen=True)
class AdapterManifest:
    path: Path
    data: Mapping[str, Any]

    @property
    def name(self) -> str:
        return str(self.data["name"])


def _git_value(repo: Path, *args: str) -> Optional[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


class Open3DBenchAdapter:
    def __init__(self, repository: Path, manifest: AdapterManifest) -> None:
        self.repository = repository.expanduser().resolve()
        self.manifest = manifest

    @classmethod
    def detect(
        cls, repository: Path, adapters_dir: Optional[Path] = None
    ) -> "Open3DBenchAdapter":
        repository = repository.expanduser().resolve()
        if not repository.is_dir():
            raise ValueError(f"Open3DBench repository is not a directory: {repository}")
        if adapters_dir is None:
            adapters_dir = Path(__file__).resolve().parents[1] / "adapters"
        matches: List[AdapterManifest] = []
        for path in sorted(adapters_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            sentinels = [repository / item for item in data.get("sentinels", [])]
            if sentinels and all(item.exists() for item in sentinels):
                matches.append(AdapterManifest(path=path, data=data))
        if not matches:
            raise AdapterNotFoundError(
                f"no Open3DBench adapter matches {repository}; "
                f"add a JSON manifest under {adapters_dir}"
            )
        matches.sort(key=lambda item: int(item.data.get("priority", 0)), reverse=True)
        return cls(repository, matches[0])

    def stage(self, stage: str) -> Mapping[str, Any]:
        require_stage(stage)
        value = self.manifest.data["stages"].get(stage)
        if not isinstance(value, dict):
            raise ValueError(f"adapter {self.manifest.name} has no {stage} stage")
        return value

    def _existing(self, values: Iterable[str]) -> List[str]:
        return [
            str(Path(value))
            for value in values
            if (self.repository / value).exists()
        ]

    def context(self, *, artifact_limit: int = 200) -> Dict[str, Any]:
        stages: Dict[str, Any] = {}
        for stage in STAGES:
            profile = self.stage(stage)
            path_groups = {
                "source_paths": list(profile.get("source_paths", [])),
                "config_paths": list(profile.get("config_paths", [])),
                "log_roots": list(profile.get("log_roots", [])),
                "result_roots": list(profile.get("result_roots", [])),
            }
            stages[stage] = {
                "regime": str(profile.get("regime", self.manifest.data.get("regime", "unknown"))),
                **path_groups,
                "path_status": {
                    value: (self.repository / value).exists()
                    for values in path_groups.values()
                    for value in values
                },
                "compile_commands": list(profile.get("compile_commands", [])),
                "preparation_commands": list(
                    profile.get("preparation_commands", [])
                ),
                "evaluation_commands": list(profile.get("evaluation_commands", [])),
                "required_inputs": list(profile.get("required_inputs", [])),
                "optional_tools": list(profile.get("optional_tools", [])),
                "metrics": list(profile.get("metrics", [])),
                "correctness_gates": list(profile.get("correctness_gates", [])),
                "artifact_patterns": list(profile.get("artifact_patterns", [])),
                "artifacts": self.collect_artifacts(stage, limit=artifact_limit),
            }
        return {
            "adapter": self.manifest.name,
            "regime": str(self.manifest.data.get("regime", "unknown")),
            "adapter_manifest": str(self.manifest.path),
            "repository": str(self.repository),
            "git": {
                "commit": _git_value(self.repository, "rev-parse", "HEAD"),
                "branch": _git_value(self.repository, "branch", "--show-current"),
                "status": _git_value(self.repository, "status", "--short"),
            },
            "stages": stages,
        }

    def collect_artifacts(self, stage: str, *, limit: int = 200) -> List[Dict[str, Any]]:
        profile = self.stage(stage)
        roots = list(profile.get("log_roots", [])) + list(
            profile.get("result_roots", [])
        )
        patterns = list(profile.get("artifact_patterns", []))
        found: Dict[Path, Dict[str, Any]] = {}
        for root_value in roots:
            root = self.repository / root_value
            if not root.exists():
                continue
            for pattern in patterns:
                for path in root.glob(pattern):
                    if not path.is_file():
                        continue
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    found[path] = {
                        "path": str(path.relative_to(self.repository)),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
        return sorted(
            found.values(), key=lambda item: int(item["mtime_ns"]), reverse=True
        )[:limit]

    def write_context(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.context(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentic3d-inspect",
        description="Detect an Open3DBench revision and print normalized stage context.",
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-limit", type=int, default=50)
    args = parser.parse_args(argv)
    try:
        adapter = Open3DBenchAdapter.detect(args.repo)
        context = adapter.context(artifact_limit=args.artifact_limit)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"kind": "adapter_error", "error": str(exc)}), file=sys.stderr)
        return 2
    rendered = json.dumps(context, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
