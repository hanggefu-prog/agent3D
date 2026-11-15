from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location("test_" + name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cache = load("openroad_compiler_cache")
driver = load("build_openroad_cached")
BASE = "sha256:" + "a" * 64


class CompilerCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "CMakeLists.txt").write_text("project(OpenROAD)\n")
        (self.source / "algorithm.cpp").write_text("int value = 1;\n")
        self.recipe = self.root / "recipe.sh"
        self.recipe.write_text("cmake -S $1 -B $2\ncmake --build $2\n")
        self.output = self.root / "attempt"
        self.shared = self.root / "cache"
        self.calls = []
        self.probe = {"schema_version": 1, "architecture": "x86_64", "tools": {
            key: {"path": "/usr/bin/" + key, "version": "test-1", "sha256": "b" * 64}
            for key in ("c", "cxx", "cmake", "ccache", "make")}}

    def fake_run(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command[-1] == "--probe":
            return SimpleNamespace(returncode=0, stdout=json.dumps(self.probe), stderr="")
        if command[-1] == "--verify":
            mount = next(x for x in command if x.endswith(",target=/candidate/openroad,readonly"))
            binary = Path(mount.split("source=", 1)[1].split(",target=", 1)[0])
            return SimpleNamespace(returncode=0, stdout=json.dumps({
                "binary_sha256": cache._sha(binary), "size_bytes": binary.stat().st_size,
                "version_exit_code": 0, "dependencies_exit_code": 0,
                "version_stdout": "synthetic version", "dependencies_stdout": "synthetic dependency"}), stderr="")
        artifacts = next(x for x in command if x.endswith(",target=/candidate/artifacts"))
        path = Path(artifacts.split("source=", 1)[1].split(",target=", 1)[0])
        binary = path / "openroad"
        binary.write_bytes(b"synthetic ELF: not an actual executable")
        binary.chmod(0o755)
        (path / "binary.json").write_text(json.dumps({
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "size_bytes": binary.stat().st_size, "candidate_executed": False}))
        return SimpleNamespace(returncode=0)

    def build(self, **kwargs):
        params = dict(base_image=BASE, source_tree=self.source, builder=self.recipe,
                      output=self.output, cache_root=self.shared, threads=8)
        params.update(kwargs)
        with mock.patch.object(cache.subprocess, "run", side_effect=self.fake_run):
            return cache.compile_cached(**params)

    def test_safe_mounts_limits_and_explicit_statistics_scope(self):
        result = self.build()
        command = self.calls[1][0]
        for value in ("--network=none", "--read-only", "--pull=never", "--cap-drop=ALL",
                      "--cpus", "16", "--memory", "48g", "--memory-swap", "--pids-limit"):
            self.assertIn(value, command)
        self.assertTrue(any(x.endswith("/build-source,target=/candidate/source") for x in command))
        self.assertFalse(any(f"source={self.source}," in x for x in command))
        self.assertTrue(any(x.endswith("target=/trusted/ccache.conf,readonly") for x in command))
        self.assertFalse(any("docker.sock" in x or ".ssh" in x for x in command))
        self.assertIn("not per-attempt", result["cache_statistics_interpretation"])
        self.assertFalse(result["native_evaluation_performed"])
        self.assertEqual(result["binary_sha256"], cache._sha(Path(result["binary_path"])))
        rows = [json.loads(x) for x in (self.output / "phases.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["state"], "started")
        self.assertEqual(rows[-1]["state"], "completed")
        self.assertGreaterEqual(rows[-1]["duration_seconds"], 0)
        self.assertTrue(all("timestamp_utc" in row for row in rows))

    def test_source_edits_reuse_compatibility_key_but_not_private_objects(self):
        first = self.build()
        (self.source / "algorithm.cpp").write_text("int value = 2;\n")
        second = self.build(output=self.root / "attempt2")
        self.assertEqual(first["cache_key"], second["cache_key"])
        self.assertNotEqual(first["source_fingerprint"], second["source_fingerprint"])
        self.assertNotEqual(first["binary_path"], second["binary_path"])
        self.assertNotEqual(self.calls[1][0], self.calls[4][0])

    def test_candidate_verification_has_no_writable_host_or_cache_mount(self):
        result = self.build()
        command = self.calls[-1][0]
        self.assertEqual(command[-1], "--verify")
        mounts = [command[i + 1] for i, part in enumerate(command) if part == "--mount"]
        self.assertEqual(len(mounts), 2)
        self.assertTrue(all(value.endswith(",readonly") for value in mounts))
        self.assertTrue(any("target=/candidate/openroad," in value for value in mounts))
        self.assertFalse(any("/ccache" in value or "/candidate/logs" in value or
                             "/candidate/source" in value for value in mounts))
        self.assertIn("--network=none", command)
        self.assertIn("--read-only", command)
        self.assertTrue(Path(result["binary_verification"]).is_file())

    def test_helper_code_identity_selects_a_new_namespace(self):
        helper = self.root / "trusted-helper.py"
        helper.write_text("trusted helper version one")
        with mock.patch.object(cache, "HELPER", helper):
            first = self.build()
            helper.write_text("trusted helper version two")
            second = self.build(output=self.root / "attempt2")
        self.assertNotEqual(first["cache_key"], second["cache_key"])

    def test_image_recipe_toolchain_and_budget_get_new_namespaces(self):
        first = self.build()
        image = self.build(output=self.root / "image", base_image="sha256:" + "c" * 64)
        self.recipe.write_text("changed trusted recipe\n")
        recipe = self.build(output=self.root / "recipe-attempt")
        self.probe["tools"]["cxx"]["sha256"] = "d" * 64
        toolchain = self.build(output=self.root / "toolchain")
        budget = self.build(output=self.root / "budget", max_size_gb=4)
        keys = {x["cache_key"] for x in (first, image, recipe, toolchain, budget)}
        self.assertEqual(len(keys), 5)
        self.assertTrue(all((self.shared / key / "identity.json").is_file() for key in keys))

    def test_preexisting_output_is_not_retried(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(self.calls, [])

    def test_nonempty_unmarked_cache_is_not_adopted(self):
        self.shared.mkdir()
        (self.shared / "other-work").write_text("keep")
        with self.assertRaisesRegex(cache.CompilerCacheError, "nonempty"):
            self.build()
        self.assertEqual((self.shared / "other-work").read_text(), "keep")
        self.assertEqual(len(self.calls), 1)

    def test_corrupt_namespace_metadata_is_preserved_and_rejected(self):
        result = self.build()
        metadata = self.shared / result["cache_key"] / "identity.json"
        metadata.write_text("{bad")
        with self.assertRaisesRegex(cache.CompilerCacheError, "Corrupt"):
            self.build(output=self.root / "attempt2")
        self.assertEqual(metadata.read_text(), "{bad")
        self.assertEqual(len(self.calls), 4)  # probe only on second attempt

    def test_namespace_identity_disagreement_is_rejected(self):
        result = self.build()
        metadata = self.shared / result["cache_key"] / "identity.json"
        metadata.write_text("{}")
        with self.assertRaisesRegex(cache.CompilerCacheError, "namespace metadata"):
            self.build(output=self.root / "attempt2")

    def test_symlink_cache_root_ancestor_and_namespace_are_rejected(self):
        actual = self.root / "actual"
        actual.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(actual, target_is_directory=True)
        for path in (alias, alias / "cache"):
            with self.subTest(path=path), self.assertRaisesRegex(cache.CompilerCacheError, "Symlink"):
                self.build(cache_root=path)
        result = self.build()
        objects = self.shared / result["cache_key"] / "objects"
        objects.rmdir()
        objects.symlink_to(actual, target_is_directory=True)
        with self.assertRaisesRegex(cache.CompilerCacheError, "Symlink"):
            self.build(output=self.root / "attempt2")

    def test_path_overlaps_are_rejected_before_docker(self):
        for kwargs in (dict(cache_root=self.source), dict(cache_root=self.source / "cache"),
                       dict(cache_root=self.root), dict(output=self.source / "out"),
                       dict(cache_root=self.output / "cache")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(cache.CompilerCacheError, "overlap"):
                self.build(**kwargs)
        self.assertEqual(self.calls, [])

    def test_unpinned_image_bad_threads_or_bad_budget_never_run(self):
        for kwargs in (dict(base_image="image:latest"), dict(threads=True), dict(threads=17),
                       dict(max_size_gb=0), dict(max_size_gb=65), dict(max_size_gb=True)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)
        self.assertEqual(self.calls, [])

    def test_failed_build_preserves_one_attempt_and_does_not_retry(self):
        original = self.fake_run
        def fail(command, **kwargs):
            if command[-1] == "--probe":
                return original(command, **kwargs)
            self.calls.append((command, kwargs))
            kwargs["stdout"].write("synthetic compile failure\n")
            return SimpleNamespace(returncode=7)
        self.fake_run = fail
        with self.assertRaisesRegex(cache.CompilerCacheError, "no retry"):
            self.build()
        self.assertEqual(len(self.calls), 2)
        self.assertIn("synthetic compile failure", (self.output / "build.log").read_text())
        self.assertTrue((self.output / "cache-identity.json").is_file())
        self.assertFalse((self.output / "result.json").exists())
        self.assertEqual(json.loads((self.output / "phases.jsonl").read_text().splitlines()[-1])["state"],
                         "failed")

    def test_incomplete_probe_fails_before_build(self):
        del self.probe["tools"]["ccache"]
        with self.assertRaisesRegex(cache.CompilerCacheError, "ccache"):
            self.build()
        self.assertEqual(len(self.calls), 1)

    def test_artifact_mutation_receipt_mismatch_is_rejected(self):
        original = self.fake_run
        def mutate(command, **kwargs):
            result = original(command, **kwargs)
            if command[-2] == "--build":
                (self.output / "artifacts/openroad").write_bytes(b"different")
            return result
        self.fake_run = mutate
        with self.assertRaisesRegex(cache.CompilerCacheError, "verification failed"):
            self.build()

    def test_source_mutation_during_build_is_rejected(self):
        original = self.fake_run
        def mutate(command, **kwargs):
            result = original(command, **kwargs)
            if command[-2] == "--build":
                (self.source / "algorithm.cpp").write_text("changed during compile")
            return result
        self.fake_run = mutate
        with self.assertRaisesRegex(cache.CompilerCacheError, "source changed"):
            self.build()

    def test_only_audited_native_generated_files_may_change(self):
        (self.source / "include/ord").mkdir(parents=True)
        (self.source / "src/grt").mkdir(parents=True)
        (self.source / "src/sta/doc").mkdir(parents=True)
        (self.source / "src/grt/messages.txt").write_text("original messages")
        original = self.fake_run
        def generate(command, **kwargs):
            result = original(command, **kwargs)
            if command[-2] == "--build":
                copied = self.output / "build-source"
                (copied / "include/ord/Version.hh").write_text("#define VERSION test\n")
                (copied / "src/grt/messages.txt").write_text("new generated messages")
                (copied / "src/grt/messages.checked").touch()
                (copied / "src/sta/doc/messages.txt").write_text("STA messages")
            return result
        self.fake_run = generate
        result = self.build()
        self.assertNotEqual(result["source_fingerprint"], result["build_source_fingerprint"])
        self.assertFalse((self.source / "include/ord/Version.hh").exists())
        self.assertEqual((self.source / "src/grt/messages.txt").read_text(), "original messages")
        generation = result["source_generation"]
        self.assertEqual(generation["unexpected_paths"], [])
        self.assertEqual(len(generation["changed_paths"]), 4)
        version = next(x for x in generation["generated_files"] if x["path"] == "include/ord/Version.hh")
        self.assertIsNone(version["before"])
        self.assertEqual(version["after"]["mode"], 0o644)
        self.assertEqual(version["after"]["sha256"], cache._sha(self.output / version["retained_path"]))

    def test_existing_generated_version_header_changes_are_recorded(self):
        (self.source / "include/ord").mkdir(parents=True)
        (self.source / "include/ord/Version.hh").write_text("old generated version")
        original = self.fake_run
        def generate(command, **kwargs):
            result = original(command, **kwargs)
            if command[-2] == "--build":
                (self.output / "build-source/include/ord/Version.hh").write_text("new generated version")
            return result
        self.fake_run = generate
        result = self.build()
        version = result["source_generation"]["generated_files"][0]
        self.assertNotEqual(version["before"]["sha256"], version["after"]["sha256"])
        self.assertEqual((self.source / "include/ord/Version.hh").read_text(), "old generated version")

    def test_unexpected_build_source_mutation_preserves_audit_and_fails(self):
        original = self.fake_run
        def mutate(command, **kwargs):
            result = original(command, **kwargs)
            if command[-2] == "--build":
                (self.output / "build-source/algorithm.cpp").write_text("unexpected mutation")
            return result
        self.fake_run = mutate
        with self.assertRaisesRegex(cache.CompilerCacheError, "Unexpected source mutation"):
            self.build()
        audit = json.loads((self.output / "source-generation.json").read_text())
        self.assertEqual(audit["unexpected_paths"], ["algorithm.cpp"])
        self.assertFalse((self.output / "result.json").exists())
        self.assertEqual((self.source / "algorithm.cpp").read_text(), "int value = 1;\n")

    def test_generated_path_removal_is_not_allowed(self):
        (self.source / "include/ord").mkdir(parents=True)
        version = self.source / "include/ord/Version.hh"
        version.write_text("original")
        original = self.fake_run
        def remove(command, **kwargs):
            result = original(command, **kwargs)
            if command[-2] == "--build":
                (self.output / "build-source/include/ord/Version.hh").unlink()
            return result
        self.fake_run = remove
        with self.assertRaisesRegex(cache.CompilerCacheError, "Unexpected source mutation"):
            self.build()

    def test_generated_header_symlink_is_not_archived_or_accepted(self):
        (self.source / "include/ord").mkdir(parents=True)
        original = self.fake_run
        def link(command, **kwargs):
            result = original(command, **kwargs)
            if command[-2] == "--build":
                (self.output / "build-source/include/ord/Version.hh").symlink_to("../../algorithm.cpp")
            return result
        self.fake_run = link
        with self.assertRaisesRegex(cache.CompilerCacheError, "Unexpected source mutation"):
            self.build()
        self.assertFalse((self.output / "generated-source/include/ord/Version.hh").exists())

    def test_generated_git_metadata_is_not_silently_excluded(self):
        original = self.fake_run
        def add_git(command, **kwargs):
            result = original(command, **kwargs)
            if command[-2] == "--build":
                (self.output / "build-source/.git").write_text("changed version identity")
            return result
        self.fake_run = add_git
        with self.assertRaisesRegex(cache.CompilerCacheError, "Unexpected source mutation.*\\.git"):
            self.build()

    def _generated_header_fixture(self):
        (self.source / "include/ord").mkdir(parents=True)
        (self.source / "include/ord/Version.hh").write_text("native generated header")
        outside = self.root / "outside"
        (outside / "include/ord").mkdir(parents=True)
        sentinel = outside / "include/ord/Version.hh"
        sentinel.write_text("external sentinel")
        return outside, sentinel

    def test_container_artifact_archive_symlink_cannot_redirect_host_archive(self):
        outside, sentinel = self._generated_header_fixture()
        original = self.fake_run
        def link(command, **kwargs):
            result = original(command, **kwargs)
            if command[-2] == "--build":
                (self.output / "artifacts/generated-source").symlink_to(outside, target_is_directory=True)
            return result
        self.fake_run = link
        result = self.build()
        self.assertEqual(sentinel.read_text(), "external sentinel")
        generated = result["source_generation"]["generated_files"][0]
        self.assertEqual(generated["retained_path"], "generated-source/include/ord/Version.hh")
        self.assertEqual((self.output / generated["retained_path"]).read_text(), "native generated header")

    def test_host_archive_directory_symlink_is_rejected_without_touching_external_file(self):
        outside, sentinel = self._generated_header_fixture()
        original = self.fake_run
        def link(command, **kwargs):
            result = original(command, **kwargs)
            if command[-2] == "--build":
                (self.output / "generated-source").symlink_to(outside, target_is_directory=True)
            return result
        self.fake_run = link
        with self.assertRaisesRegex(cache.CompilerCacheError, "Symlink"):
            self.build()
        self.assertEqual(sentinel.read_text(), "external sentinel")

    def test_host_archive_leaf_symlink_is_rejected_without_touching_external_file(self):
        _, sentinel = self._generated_header_fixture()
        original = self.fake_run
        def link(command, **kwargs):
            result = original(command, **kwargs)
            if command[-2] == "--build":
                parent = self.output / "generated-source/include/ord"
                parent.mkdir(parents=True)
                (parent / "Version.hh").symlink_to(sentinel)
            return result
        self.fake_run = link
        with self.assertRaisesRegex(cache.CompilerCacheError, "Symlink"):
            self.build()
        self.assertEqual(sentinel.read_text(), "external sentinel")

    def test_failed_isolated_verification_is_not_retried(self):
        original = self.fake_run
        def fail(command, **kwargs):
            result = original(command, **kwargs)
            if command[-1] == "--verify":
                return SimpleNamespace(returncode=9, stdout="", stderr="synthetic runtime failure")
            return result
        self.fake_run = fail
        with self.assertRaisesRegex(cache.CompilerCacheError, "verification failed; no retry"):
            self.build()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual((self.output / "verify.stderr").read_text(), "synthetic runtime failure")
        self.assertTrue((self.output / "artifacts/openroad").exists())

    def test_source_symlink_escape_fails_before_docker(self):
        (self.source / "escape").symlink_to(self.recipe)
        with self.assertRaisesRegex(Exception, "source link"):
            self.build()
        self.assertEqual(self.calls, [])


class TrustedDriverTest(unittest.TestCase):
    def test_staging_does_not_execute_the_candidate_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            binary = root / "input-openroad"
            binary.write_bytes(b"not a real executable")
            binary.chmod(0o755)
            target = root / "staged-openroad"
            receipt = root / "binary.json"
            paths = {"/candidate/build/bin/openroad": binary,
                     "/candidate/artifacts/openroad": target,
                     "/candidate/artifacts/binary.json": receipt}
            with mock.patch.object(driver, "Path", side_effect=lambda value: paths[value]), \
                    mock.patch.object(driver, "event"), \
                    mock.patch.object(driver.subprocess, "run") as run:
                driver.stage_artifact()
            run.assert_not_called()
            self.assertEqual(target.read_bytes(), binary.read_bytes())
            self.assertIs(json.loads(receipt.read_text())["candidate_executed"], False)

    def test_inherited_ccache_shortcuts_removed_but_image_compiler_preserved(self):
        with mock.patch.dict(os.environ, {"CCACHE_SLOPPINESS": "time_macros", "CCACHE_DISABLE": "1",
                                          "CCACHE_CONFIGPATH": "/unsafe", "CC": "/usr/bin/gcc-11"}):
            env = driver.clean_environment()
        self.assertFalse(any(key.startswith("CCACHE_") for key in env))
        self.assertEqual(env["CC"], "/usr/bin/gcc-11")
        self.assertEqual(env["HOME"], "/tmp/openroad-home")

    def test_cmake_interceptor_preserves_arguments_and_exit_code(self):
        for arguments, phase in ((["-S", "/candidate/source", "-B", "/candidate/build",
                                    "-DCMAKE_BUILD_TYPE=Release"], "configure"),
                                 (["--build", "/candidate/build", "--target", "openroad", "-j", "8"],
                                  "compile_link")):
            with self.subTest(phase=phase), mock.patch.dict(os.environ, {
                    "OPENROAD_TRUSTED_CMAKE": "/usr/bin/cmake"}), \
                    mock.patch.object(driver, "event") as events, \
                    mock.patch.object(driver.subprocess, "run", return_value=SimpleNamespace(returncode=3)) as run:
                self.assertEqual(driver.cmake(arguments), 3)
                run.assert_called_once_with(["/usr/bin/cmake", *arguments], check=False)
                self.assertEqual(events.call_args_list[0].args, (phase, "started"))
                self.assertEqual(events.call_args_list[1].kwargs["exit_code"], 3)

    def test_compiler_command_wrappers_are_not_silently_accepted(self):
        with self.assertRaisesRegex(RuntimeError, "one executable"):
            driver.tool("ccache g++", {})


if __name__ == "__main__":
    unittest.main()
