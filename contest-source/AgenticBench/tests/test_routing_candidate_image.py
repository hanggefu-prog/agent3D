from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from local_bench import workspace

ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "routing_image_builder_test", ROOT / "scripts" / "build_routing_candidate_image.py")
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
builder = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(builder)
BASE = "sha256:" + "a" * 64
IMAGE = "sha256:" + "b" * 64


class RoutingRuntimeReceiptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "candidate"
        self.source = self.repo / workspace.OPENROAD_SUBMODULE
        self.source.mkdir(parents=True)
        (self.source / "CMakeLists.txt").write_text("project(OpenROAD)\n")
        (self.source / "algorithm.cpp").write_text("int policy = 1;\n")
        self.binary = self.root / "installed" / "openroad"
        self.binary.parent.mkdir()
        self.binary.write_bytes(b"image-installed-openroad")
        self.binary.chmod(0o755)
        self.manifest = self.root / "installed" / "runtime.json"
        self.receipt = self.root / "receipt.json"
        self.payload = {
            "schema_version": 1, "kind": "routing-source-installed-openroad",
            "source_fingerprint": workspace.openroad_source_fingerprint(self.source),
            "binary_path": str(self.binary),
            "binary_sha256": hashlib.sha256(self.binary.read_bytes()).hexdigest(),
        }
        self.manifest.write_text(json.dumps(self.payload))
        self.receipt.write_text(json.dumps({**self.payload, "image_id": IMAGE}))
        for patch in (
            mock.patch.object(workspace, "PREBUILT_OPENROAD_BINARY", self.binary),
            mock.patch.object(workspace, "PREBUILT_OPENROAD_MANIFEST", self.manifest),
            mock.patch.object(workspace, "_inside_docker", return_value=True),
            mock.patch.dict(os.environ, {workspace.PREBUILT_OPENROAD_ENV: str(self.receipt),
                                         "OPEN3DBENCH_IMAGE_ID": IMAGE}),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.changed = [workspace.OPENROAD_SUBMODULE + "/algorithm.cpp"]

    def evaluate(self):
        return workspace.prepare_openroad_runtime(self.repo, self.root / "result", self.changed)

    def test_matching_receipt_uses_installed_binary_without_build(self):
        with mock.patch.object(workspace.subprocess, "run") as run:
            self.assertEqual(self.evaluate(), self.binary)
        run.assert_not_called()
        result = json.loads((self.root / "result/openroad-prebuilt-receipt.json").read_text())
        self.assertEqual(result["verification"], "passed")

    def test_changed_source_is_rejected(self):
        (self.source / "algorithm.cpp").write_text("int policy = 2;\n")
        with self.assertRaisesRegex(workspace.WorkspaceError, "source fingerprint"):
            self.evaluate()

    def test_wrong_image_is_rejected(self):
        with mock.patch.dict(os.environ, {"OPEN3DBENCH_IMAGE_ID": BASE}):
            with self.assertRaisesRegex(workspace.WorkspaceError, "image ID"):
                self.evaluate()

    def test_changed_binary_is_rejected(self):
        self.binary.write_bytes(b"mounted-or-mutated")
        with self.assertRaisesRegex(workspace.WorkspaceError, "binary hash"):
            self.evaluate()

    def test_external_receipt_cannot_override_image_manifest(self):
        value = {**self.payload, "image_id": IMAGE, "binary_sha256": "c" * 64}
        self.receipt.write_text(json.dumps(value))
        with self.assertRaisesRegex(workspace.WorkspaceError, "installed manifest"):
            self.evaluate()

    def test_host_execution_is_rejected(self):
        with mock.patch.object(workspace, "_inside_docker", return_value=False):
            with self.assertRaisesRegex(workspace.WorkspaceError, "inside Docker"):
                self.evaluate()

    def test_receipt_without_cpp_candidate_is_rejected(self):
        with self.assertRaisesRegex(workspace.WorkspaceError, "without an OpenROAD"):
            workspace.prepare_openroad_runtime(self.repo, self.root / "result", [])

    def test_no_cpp_candidate_preserves_default_noop(self):
        with mock.patch.dict(os.environ, {workspace.PREBUILT_OPENROAD_ENV: ""}):
            self.assertIsNone(workspace.prepare_openroad_runtime(self.repo, self.root / "result", []))


class RoutingSourceFingerprintTest(unittest.TestCase):
    def test_git_metadata_is_excluded_but_source_and_modes_are_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "CMakeLists.txt").write_text("project(OpenROAD)")
            initial = workspace.openroad_source_fingerprint(root)
            (root / ".git").mkdir()
            (root / ".git/config").write_text("host-specific metadata")
            self.assertEqual(workspace.openroad_source_fingerprint(root), initial)
            (root / "CMakeLists.txt").chmod(0o755)
            self.assertNotEqual(workspace.openroad_source_fingerprint(root), initial)

    def test_external_source_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            root.mkdir()
            (root / "CMakeLists.txt").write_text("project(OpenROAD)")
            (Path(temp) / "secret").write_text("not-a-build-input")
            (root / "outside").symlink_to("../secret")
            with self.assertRaisesRegex(workspace.WorkspaceError, "escaping/broken"):
                workspace.openroad_source_fingerprint(root)


class RoutingCandidateImageBuilderTest(unittest.TestCase):
    def test_only_pinned_local_images_are_accepted(self):
        with mock.patch.object(builder, "_run_json") as docker:
            with self.assertRaisesRegex(ValueError, "pinned"):
                builder._local_image_id("open3dbench:latest")
        docker.assert_not_called()

    def test_dockerfile_installs_binary_and_uses_only_trusted_builder(self):
        text = builder._dockerfile(BASE, 8)
        self.assertIn("FROM " + BASE, text)
        self.assertIn("COPY --from=candidate_builder /candidate/openroad /opt/", text)
        self.assertIn("bash /trusted/build_openroad_candidate.sh", text)
        self.assertNotIn(" -version", text)  # candidate code runs only after sealing in a read-only container
        self.assertNotIn("apt-get", text)
        self.assertNotIn("curl", text)
        self.assertIn("CCACHE_DIR=/build/ccache CCACHE_DISABLE=1", text)
        self.assertIn("/candidate/runtime.json && rm -rf /build/openroad /build/ccache", text)
        self.assertNotIn("rm -rf", builder._dockerfile(BASE, 8, "retain"))
        with self.assertRaises(ValueError):
            builder._dockerfile(BASE, 8, "global-prune")
        with self.assertRaises(ValueError):
            builder._dockerfile(BASE, 17)
        with self.assertRaises(ValueError):
            builder._dockerfile(BASE, True)

    def test_build_handshake_binds_image_receipt_and_resource_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "source", root / "image-build"
            source.mkdir()
            candidate = {"source_patch": "verified patch", "schema_version": 1}
            task = SimpleNamespace(kind="routing_net_ordering_source", task_id="routing-test")

            def clone(_source, dest, **kwargs):
                self.assertEqual(kwargs["mutable_submodules"], [workspace.OPENROAD_SUBMODULE])
                tree = dest / workspace.OPENROAD_SUBMODULE
                tree.mkdir(parents=True)
                (tree / "CMakeLists.txt").write_text("project(OpenROAD)")

            commands = []

            def run(command, **kwargs):
                commands.append(command)
                if command[-2:] == ["rev-parse", "HEAD"]:
                    return subprocess.CompletedProcess(command, 0, stdout="d" * 40 + "\n", stderr="")
                if command[:2] == ["docker", "build"]:
                    self.assertEqual(kwargs["env"]["DOCKER_BUILDKIT"], "0")
                    self.assertIn("--network=none", command)
                    self.assertIn("--pull=false", command)
                    self.assertEqual(command[command.index("--cpu-quota") + 1], "1600000")
                    self.assertEqual(command[command.index("--memory") + 1], "48g")
                    (output / "image.id").write_text(IMAGE)
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            def runtime(_command):
                self.assertIn("--read-only", _command)
                self.assertIn("--network=none", _command)
                if "python3" in _command:
                    return {"binary_sha256": "c" * 64,
                            "binary_path": str(workspace.PREBUILT_OPENROAD_BINARY),
                            "resolved_command": str(workspace.PREBUILT_OPENROAD_BINARY),
                            "dependencies": "fixture dependencies"}
                payload = json.loads((output / "context/runtime-template.json").read_text())
                return {**payload, "binary_sha256": "c" * 64}

            with (mock.patch.object(builder, "load_task", return_value=task),
                  mock.patch.object(builder, "load_candidate", return_value=candidate),
                  mock.patch.object(builder, "verify_candidate", return_value={"patch": {
                      "paths": [workspace.OPENROAD_SUBMODULE + "/src/grt/policy.cpp"]}}),
                  mock.patch.object(builder, "isolated_clone", side_effect=clone),
                  mock.patch.object(builder, "apply_candidate_patch"),
                  mock.patch.object(builder, "_local_image_id", side_effect=[BASE, IMAGE]),
                  mock.patch.object(builder, "_run_json", side_effect=runtime),
                  mock.patch.object(builder.subprocess, "run", side_effect=run)):
                result = builder.build_candidate_image(root / "task", root / "candidate", source, BASE, output)
            self.assertEqual(result["image"], IMAGE)
            receipt = json.loads(Path(result["receipt"]).read_text())
            self.assertEqual(receipt["image_id"], IMAGE)
            self.assertEqual(receipt["base_image_id"], BASE)
            self.assertEqual(receipt["source_commit"], "d" * 40)
            self.assertEqual(receipt["openroad_commit"], "d" * 40)
            self.assertEqual(json.loads((output / "candidate.json").read_text()), candidate)
            self.assertEqual(receipt["build_cache_policy"], "discard")
            self.assertFalse((output / "workspace").exists())
            self.assertFalse((output / "context/openroad").exists())
            self.assertTrue((output / "context/Dockerfile").is_file())
            self.assertTrue((output / "reproduction/Dockerfile").is_file())
            self.assertTrue((output / "candidate.patch").is_file())
            self.assertTrue((output / "image-build.log").is_file())
            ledger = json.loads((output / "cache-retention.json").read_text())
            self.assertEqual(ledger["state"], "completed")
            self.assertEqual(ledger["deleted_paths"], ["workspace", "context/openroad"])
            self.assertEqual(ledger["receipt_sha256"], hashlib.sha256((output / "receipt.json").read_bytes()).hexdigest())
            self.assertEqual(sum(command[:2] == ["docker", "build"] for command in commands), 1)

    def test_missing_verified_receipt_prevents_cache_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "workspace").mkdir()
            (root / "workspace/keep").write_text("not verified")
            (root / "context/openroad").mkdir(parents=True)
            for name in ("Dockerfile", "runtime-template.json", "seal_runtime.py", "build_openroad_candidate.sh"):
                (root / "context" / name).write_text("recipe")
            with self.assertRaises(workspace.WorkspaceError):
                builder._discard_build_copies(root)
            self.assertTrue((root / "workspace/keep").is_file())
            self.assertFalse((root / "reproduction").exists())

    def test_cleanup_failure_is_storage_only_and_does_not_replace_build_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            receipt = {"image_id": IMAGE, "binary_sha256": "c" * 64}
            (output / "receipt.json").write_text(json.dumps(receipt))
            with mock.patch.object(builder, "_discard_build_copies", side_effect=OSError("disk full")):
                builder._finish_build_cache(output, IMAGE)
            self.assertEqual(json.loads((output / "receipt.json").read_text()), receipt)
            error = json.loads((output / "cache-retention-error.json").read_text())
            self.assertTrue(error["installed_image_verified"])
            self.assertEqual(error["state"], "storage-error")
            self.assertEqual(error["image_id"], IMAGE)

    def test_symlinked_cache_root_is_never_followed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "output"
            root.mkdir()
            external = Path(temp) / "external"
            external.mkdir()
            (external / "keep").write_text("preserve")
            (root / "workspace").symlink_to(external, target_is_directory=True)
            (root / "context/openroad").mkdir(parents=True)
            with self.assertRaises(workspace.WorkspaceError):
                builder._discard_build_copies(root)
            self.assertEqual((external / "keep").read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
