from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic3d.scope_guard import (  # noqa: E402
    IntegrationGate,
    ScopeCompiler,
    ScopeGuard,
    StageScope,
    WorktreeManager,
)


def git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.strip()


class ScopeGuardTests(unittest.TestCase):
    def _repository(self, root: Path) -> Path:
        repository = root / "repo"
        (repository / "Place-3D" / "src" / "partition").mkdir(parents=True)
        (repository / "Place-3D" / "src" / "placement").mkdir(parents=True)
        (repository / "Place-3D" / "config" / "placement").mkdir(parents=True)
        (repository / "OpenROAD-3D" / "flow" / "script_routing").mkdir(
            parents=True
        )
        (repository / "Place-3D" / "src" / "partition" / "partition.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        (repository / "Place-3D" / "src" / "placement" / "placer.py").write_text(
            "DENSITY = 0.70\n", encoding="utf-8"
        )
        (repository / "Place-3D" / "config" / "placement" / "base.json").write_text(
            json.dumps({"target_density": 0.7}) + "\n", encoding="utf-8"
        )
        (repository / "OpenROAD-3D" / "flow" / "script_routing" / "route.py").write_text(
            "PASSES = 1\n", encoding="utf-8"
        )
        git(repository, "init")
        git(repository, "add", ".")
        git(
            repository,
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "baseline",
        )
        return repository

    def _layout_scope(self, repository: Path) -> StageScope:
        return ScopeCompiler.compile(
            stage="layout",
            baseline_commit=git(repository, "rev-parse", "HEAD"),
            adapter_profile={
                "source_paths": ["Place-3D/src/placement"],
                "config_paths": ["Place-3D/config/placement"],
            },
            proposal={
                "source_paths": ["Place-3D/src/placement/placer.py"],
                "config_paths": [],
                "allowed_extensions": [".py"],
                "max_changed_files": 2,
                "max_added_lines": 20,
                "max_deleted_lines": 20,
            },
        )

    def test_compiler_rejects_a_local_analyzer_scope_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repository = self._repository(Path(temp))
            with self.assertRaisesRegex(ValueError, "out-of-stage"):
                ScopeCompiler.compile(
                    stage="layout",
                    baseline_commit=git(repository, "rev-parse", "HEAD"),
                    adapter_profile={
                        "source_paths": ["Place-3D/src/placement"],
                        "config_paths": ["Place-3D/config/placement"],
                    },
                    proposal={
                        "source_paths": ["OpenROAD-3D/flow/script_routing"],
                        "config_paths": [],
                    },
                )

    def test_allowed_change_is_checkpointed_and_integrated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repository = self._repository(root)
            scope = self._layout_scope(repository)
            scope_path = root / "scope.json"
            scope.write(scope_path)
            self.assertEqual(StageScope.read(scope_path).computed_digest(), scope.computed_digest())

            worktree = WorktreeManager(repository).create(root / "layout-worktree")
            target = worktree / "Place-3D" / "src" / "placement" / "placer.py"
            target.write_text("DENSITY = 0.75\n", encoding="utf-8")
            guard = ScopeGuard(worktree, StageScope.read(scope_path))
            baseline = guard.head()
            report = guard.audit(
                expected_head=baseline,
                declared_files=["Place-3D/src/placement/placer.py"],
            )
            self.assertTrue(report.passed, report.violations)
            candidate = guard.checkpoint(message="agentic3d(layout): round 1")
            self.assertIsNotNone(candidate)

            integrated = IntegrationGate(repository).integrate(
                scope=scope,
                worktree=worktree,
                candidate_head=str(candidate),
                message="agentic3d(layout): accepted candidate",
            )
            self.assertEqual(integrated, git(repository, "rev-parse", "HEAD"))
            self.assertEqual(
                (repository / "Place-3D" / "src" / "placement" / "placer.py").read_text(),
                "DENSITY = 0.75\n",
            )

    def test_out_of_scope_change_is_rejected_and_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repository = self._repository(root)
            scope = self._layout_scope(repository)
            worktree = WorktreeManager(repository).create(root / "layout-worktree")
            baseline = git(worktree, "rev-parse", "HEAD")
            illegal = worktree / "OpenROAD-3D" / "flow" / "script_routing" / "route.py"
            illegal.write_text("PASSES = 2\n", encoding="utf-8")
            guard = ScopeGuard(worktree, scope)
            report = guard.audit(
                expected_head=baseline,
                declared_files=["OpenROAD-3D/flow/script_routing/route.py"],
            )
            self.assertFalse(report.passed)
            self.assertIn(
                "path_out_of_scope", {item["type"] for item in report.violations}
            )
            guard.rollback(expected_head=baseline)
            self.assertEqual(illegal.read_text(), "PASSES = 1\n")
            self.assertEqual(
                (repository / "OpenROAD-3D" / "flow" / "script_routing" / "route.py").read_text(),
                "PASSES = 1\n",
            )

    def test_final_audit_enforces_budget_across_multiple_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repository = self._repository(root)
            scope = ScopeCompiler.compile(
                stage="layout",
                baseline_commit=git(repository, "rev-parse", "HEAD"),
                adapter_profile={
                    "source_paths": ["Place-3D/src/placement"],
                    "config_paths": [],
                },
                proposal={
                    "source_paths": ["Place-3D/src/placement/placer.py"],
                    "config_paths": [],
                    "allowed_extensions": [".py"],
                    "max_changed_files": 1,
                    "max_added_lines": 1,
                    "max_deleted_lines": 5,
                },
            )
            worktree = WorktreeManager(repository).create(root / "layout-worktree")
            guard = ScopeGuard(worktree, scope)
            target = worktree / "Place-3D" / "src" / "placement" / "placer.py"
            target.write_text("DENSITY = 0.75\n", encoding="utf-8")
            first = guard.audit(
                expected_head=guard.head(),
                declared_files=["Place-3D/src/placement/placer.py"],
            )
            self.assertTrue(first.passed, first.violations)
            first_head = guard.checkpoint(message="agentic3d(layout): round 1")
            target.write_text("DENSITY = 0.75\nGAMMA = 4.0\n", encoding="utf-8")
            second = guard.audit(
                expected_head=str(first_head),
                declared_files=["Place-3D/src/placement/placer.py"],
            )
            self.assertTrue(second.passed, second.violations)
            candidate = guard.checkpoint(message="agentic3d(layout): round 2")

            final = guard.audit_candidate(candidate_head=str(candidate))
            self.assertFalse(final.passed)
            self.assertIn(
                "added_line_budget_exceeded",
                {item["type"] for item in final.violations},
            )

    def test_integration_refuses_staged_canonical_user_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repository = self._repository(root)
            scope = self._layout_scope(repository)
            worktree = WorktreeManager(repository).create(root / "layout-worktree")
            guard = ScopeGuard(worktree, scope)
            target = worktree / "Place-3D" / "src" / "placement" / "placer.py"
            target.write_text("DENSITY = 0.75\n", encoding="utf-8")
            report = guard.audit(
                expected_head=guard.head(),
                declared_files=["Place-3D/src/placement/placer.py"],
            )
            self.assertTrue(report.passed, report.violations)
            candidate = guard.checkpoint(message="agentic3d(layout): candidate")

            user_file = repository / "Place-3D" / "src" / "partition" / "partition.py"
            user_file.write_text("VALUE = 2\n", encoding="utf-8")
            git(repository, "add", "Place-3D/src/partition/partition.py")
            with self.assertRaisesRegex(RuntimeError, "tracked modifications"):
                IntegrationGate(repository).integrate(
                    scope=scope,
                    worktree=worktree,
                    candidate_head=str(candidate),
                    message="must not integrate",
                )
            self.assertIn(
                "Place-3D/src/partition/partition.py",
                git(repository, "diff", "--cached", "--name-only"),
            )

    def test_partition_layout_and_routing_manifests_compile(self) -> None:
        adapters = Path(__file__).resolve().parents[1] / "adapters"
        for manifest_path in adapters.glob("*.json"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for stage in ("partition", "layout", "routing"):
                profile = manifest["stages"][stage]
                proposal = {
                    "source_paths": [profile["source_paths"][0]],
                    "config_paths": [],
                }
                scope = ScopeCompiler.compile(
                    stage=stage,
                    baseline_commit="a" * 40,
                    adapter_profile=profile,
                    proposal=proposal,
                )
                self.assertEqual(scope.stage, stage)


if __name__ == "__main__":
    unittest.main()
