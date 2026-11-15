"""Exercise manual GRT dispatch with a fake Docker/CLI, never EDA tools."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/manual_flow.sh"


class ManualFlowGRTTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.work = self.root / "work"
        self.work.mkdir()
        for directory in ("source", "logs", "tmp"):
            (self.work / directory).mkdir()
        for name, value in {
            ".manual-flow": "1", "source-root.txt": str(self.root),
            "image.txt": "test-image", "gpu.txt": "none", "config.yaml": "{}",
        }.items():
            (self.work / name).write_text(value + "\n")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.calls = self.root / "calls.jsonl"
        docker = self.bin / "docker"
        docker.write_text("#!" + sys.executable + "\n" + '''
import json, os, subprocess, sys
if sys.argv[1:3] == ["image", "inspect"]:
    print("sha256:test-image")
else:
    with open(os.environ["FAKE_DOCKER_LOG"], "a") as stream:
        stream.write(json.dumps(sys.argv[1:]) + "\\n")
    image = sys.argv.index("test-image")
    raise SystemExit(subprocess.call(sys.argv[image + 1:]))
''')
        docker.chmod(0o755)
        python = self.bin / "python3"
        python.write_text("#!" + sys.executable + "\n" + '''
import json, os, subprocess, sys
if sys.argv[1:2] == ["-c"]:
    raise SystemExit(subprocess.call([sys.executable] + sys.argv[1:]))
if "--help" in sys.argv:
    print("{plan,run,global-route}" if os.environ["FAKE_CLI"] == "new" else "{plan,run}")
else:
    with open(os.environ["FAKE_CALL_LOG"], "a") as stream:
        stream.write(json.dumps(sys.argv[1:]) + "\\n")
''')
        python.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        FAKE_CALL_LOG=str(self.calls), FAKE_CLI="new", USE_LOCAL_OPENROAD="0",
                        FAKE_DOCKER_LOG=str(self.root / "docker-calls.jsonl"))
        self.env.pop("MANUAL_FLOW_CPUS", None)
        self.env.pop("MANUAL_FLOW_MEMORY", None)
        self.env.pop("AGENTICBENCH_ADMITTED_GPU", None)

    def execute(self, stage, output=None, cli="new"):
        args = ["bash", str(SCRIPT), stage, str(self.work)]
        if output is not None:
            args.append(str(output))
        return subprocess.run(args, env=dict(self.env, FAKE_CLI=cli),
                              capture_output=True, text=True)

    def last_call(self):
        return json.loads(self.calls.read_text().splitlines()[-1])

    def test_new_grt_uses_standalone_command(self):
        result = self.execute("grt")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = self.last_call()
        self.assertIn("global-route", argv)
        self.assertNotIn("run", argv)
        self.assertNotIn("--resume", argv)

    def test_new_grt_forwards_independent_output(self):
        output = self.work / "grt-inspection"
        result = self.execute("global_routing", output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = self.last_call()
        self.assertEqual(argv[argv.index("--output-dir") + 1], str(output))

    def test_old_split_stage_cli_is_preserved(self):
        result = self.execute("global_routing", cli="old")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = self.last_call()
        self.assertIn("run", argv)
        self.assertEqual(argv[argv.index("--from") + 1], "global_routing")
        self.assertEqual(argv[argv.index("--to") + 1], "global_routing")
        self.assertIn("--resume", argv)

    def test_old_cli_rejects_separate_output_instead_of_ignoring_it(self):
        result = self.execute("grt", self.work / "new", cli="old")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot run standalone GRT", result.stdout + result.stderr)
        self.assertFalse(self.calls.exists())

    def test_full_routing_still_uses_normal_runner(self):
        result = self.execute("routing")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = self.last_call()
        self.assertIn("run", argv)
        self.assertEqual(argv[argv.index("--from") + 1], "routing")
        self.assertNotIn("global-route", argv)

    def test_resource_caps_are_opt_in(self):
        result = self.execute("routing")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = json.loads((self.root / "docker-calls.jsonl").read_text().splitlines()[-1])
        for flag in ("--cpus", "--memory", "--memory-swap"):
            self.assertNotIn(flag, argv)

    def test_admitted_gpu_must_match_saved_workspace_gpu_before_log_rotation(self):
        previous = self.work / "logs/routing.log"
        previous.write_text("existing stage evidence\n")
        self.env["AGENTICBENCH_ADMITTED_GPU"] = "1"
        result = self.execute("routing")
        self.assertEqual(result.returncode, 2)
        self.assertIn("GPU admission mismatch", result.stderr)
        self.assertEqual(previous.read_text(), "existing stage evidence\n")
        self.assertEqual(list(previous.parent.glob("routing.previous.*")), [])
        self.assertFalse((self.root / "docker-calls.jsonl").exists())
        self.assertFalse(self.calls.exists())

    def test_admitted_gpu_accepts_matching_cuda_and_cpu_workspaces(self):
        for gpu in ("2", "none"):
            with self.subTest(gpu=gpu):
                (self.work / "gpu.txt").write_text(gpu + "\n")
                self.env["AGENTICBENCH_ADMITTED_GPU"] = gpu
                result = self.execute("grt")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                argv = json.loads((self.root / "docker-calls.jsonl").read_text().splitlines()[-1])
                if gpu == "none":
                    self.assertNotIn("--gpus", argv)
                else:
                    self.assertEqual(argv[argv.index("--gpus") + 1], "device=" + gpu)

    def test_init_rejects_admitted_gpu_mismatch_without_creating_workspace(self):
        work = self.root / "new-work"
        result = subprocess.run(
            ["bash", str(SCRIPT), "init", "bp_fe", str(work)], capture_output=True, text=True,
            env=dict(self.env, OPEN3DBENCH_ROOT=str(SCRIPT.parents[2]),
                     AGENTICBENCH_GPU="0", AGENTICBENCH_ADMITTED_GPU="1"),
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("GPU admission mismatch", result.stderr)
        self.assertFalse(work.exists())

    def test_invalid_admitted_gpu_is_rejected(self):
        for value in ("", "invalid", "-1", "0,1"):
            with self.subTest(value=value):
                self.env["AGENTICBENCH_ADMITTED_GPU"] = value
                result = self.execute("routing")
                self.assertEqual(result.returncode, 2)
                self.assertIn("AGENTICBENCH_ADMITTED_GPU", result.stderr)
                self.assertFalse((self.root / "docker-calls.jsonl").exists())

    def test_resource_caps_forward_without_changing_the_physical_config(self):
        before = (self.work / "config.yaml").read_bytes()
        for stage in ("grt", "routing", "placement"):
            with self.subTest(stage=stage):
                self.env.update(MANUAL_FLOW_CPUS="2.5", MANUAL_FLOW_MEMORY="16g")
                result = self.execute(stage)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                argv = json.loads((self.root / "docker-calls.jsonl").read_text().splitlines()[-1])
                for flag, value in (("--cpus", "2.5"), ("--memory", "16g"), ("--memory-swap", "16g")):
                    self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertEqual((self.work / "config.yaml").read_bytes(), before)

    def test_positive_resource_size_formats(self):
        for cpus, memory in (("4", "512m"), (".5", "1.5GiB"), ("0.25", "16777216")):
            with self.subTest(cpus=cpus, memory=memory):
                self.env.update(MANUAL_FLOW_CPUS=cpus, MANUAL_FLOW_MEMORY=memory)
                result = self.execute("grt")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_invalid_resource_caps_fail_before_starting_a_container(self):
        for key, values in {
            "MANUAL_FLOW_CPUS": ("", "0", "0.0", "-1", "NaN", "4 --privileged", "1e2"),
            "MANUAL_FLOW_MEMORY": ("", "0", "0.0g", "-1g", "unlimited", "16g --privileged"),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    self.env.pop("MANUAL_FLOW_CPUS", None)
                    self.env.pop("MANUAL_FLOW_MEMORY", None)
                    self.env[key] = value
                    result = self.execute("routing")
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn(key, result.stderr)
                    self.assertFalse((self.root / "docker-calls.jsonl").exists())
                    self.assertFalse(self.calls.exists())

    def test_output_outside_mount_or_for_other_stage_is_rejected(self):
        self.assertNotEqual(self.execute("grt", self.root / "outside").returncode, 0)
        self.assertNotEqual(self.execute("cts", self.work / "new").returncode, 0)
        self.assertFalse(self.calls.exists())

    def test_manual_flow_rejects_host_build_injection_before_docker(self):
        self.env["USE_LOCAL_OPENROAD"] = "1"
        result = self.execute("routing")
        self.assertEqual(result.returncode, 2)
        self.assertIn("pinned Docker image", result.stderr)
        self.assertFalse(self.calls.exists())

    def test_benchmark_launcher_rejects_host_build_but_keeps_candidate_builder(self):
        script = SCRIPT.with_name("benchmark_docker.sh")
        data = self.root / "unused-data"
        result = subprocess.run(
            ["bash", str(script), "list"], capture_output=True, text=True,
            env=dict(self.env, USE_LOCAL_OPENROAD="1", AGENTICBENCH_DATA_ROOT=str(data)),
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("pinned Docker image", result.stderr)
        self.assertFalse(data.exists())
        self.assertFalse(self.calls.exists())
        # An isolated candidate compiled inside the evaluation container is a
        # separate supported path, not an untracked host build injection.
        self.assertIn('OPEN3D_LOCAL_OPENROAD_BUILDER="${framework}/scripts/build_openroad_candidate.sh"', script.read_text())


if __name__ == "__main__":
    unittest.main()
