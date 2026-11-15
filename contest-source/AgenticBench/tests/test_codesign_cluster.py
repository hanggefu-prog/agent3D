from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from scripts import schedule_codesign_cluster as scheduler


def configuration(root="/tmp/cluster-test"):
    return {"cluster_id": "test-batch", "hosts": [{
        "id": "host-a", "ssh": "test-host", "root": root,
        "python": "/usr/bin/python3", "script": "/source/schedule_codesign_cluster.py",
        "slots": 2, "cpus": 16, "memory_mb": 32000, "disk_mb": 50000,
        "gpus": [{"index": 0, "memory_mb": 20000}]}], "jobs": [job("one")]}


def job(name, seconds=10, deps=()):
    return {"id": name, "deps": list(deps), "estimated_seconds": seconds,
            "resources": {"cpus": 4, "memory_mb": 4000, "disk_mb": 1000, "gpu_memory_mb": 4000},
            "command": ["{python}", "{root}/wrapper.py", "--slot", "{slot}", "--gpu", "{gpu}", "--out", "{job_dir}"]}


def live():
    return {"sampled_at": time.time(), "cpu_idle_cores": 32, "memory_available_mb": 64000,
            "disk_free_mb": 100000, "gpu_free_mb": {"0": 20000},
            "scheduler_sha256": scheduler.code_hash(), "gpu_utilization": {"0": 100}}


class ClusterSchedulerTest(unittest.TestCase):
    def test_critical_path_not_just_longest_single_job(self):
        jobs = {j["id"]: j for j in [job("short-parent", 2), job("long-tail", 100, ["short-parent"]), job("isolated", 20)]}
        state = {name: {"status": "pending"} for name in jobs}
        self.assertEqual(scheduler.ready(jobs, state), ["short-parent", "isolated"])
        self.assertEqual(scheduler.ranks(jobs)["short-parent"], 102)

    def test_static_plan_uses_fastest_admissible_host(self):
        config = configuration()
        other = {**config["hosts"][0], "id": "host-b", "speed": 2}
        config["hosts"].append(other)
        result = scheduler.plan(config)
        self.assertEqual(result["jobs"][0]["host"], "host-b")
        self.assertEqual(result["estimated_makespan_seconds"], 5)
        self.assertTrue(result["live_admission_not_checked"])

    def test_affinity_and_distinct_slots(self):
        config = configuration()
        config["hosts"].append({**config["hosts"][0], "id": "host-b"})
        config["jobs"] = [job("prepare"), job("timing", deps=["prepare"]), job("thermal", deps=["prepare"])]
        for item in config["jobs"]:
            item["affinity_group"] = "bp"
        plan = scheduler.plan(config)["jobs"]
        self.assertEqual({row["host"] for row in plan}, {"host-a"})
        self.assertEqual({row["slot"] for row in plan[1:]}, {"01", "02"})
        self.assertEqual({row["estimated_start"] for row in plan[1:]}, {10})

    def test_explicit_host_subset_and_cpu_only(self):
        config = configuration()
        config["hosts"].append({**config["hosts"][0], "id": "host-b"})
        config["jobs"][0].update(hosts=["host-b"])
        config["jobs"][0]["resources"]["gpu_memory_mb"] = 0
        config["hosts"][1]["gpus"] = []
        allocation = scheduler.plan(config)["jobs"][0]
        self.assertEqual((allocation["host"], allocation["gpu"]), ("host-b", "none"))

    def test_cycle_and_duplicate_rejected(self):
        config = configuration()
        config["jobs"] = [job("one", deps=["two"]), job("two", deps=["one"])]
        with self.assertRaisesRegex(ValueError, "cycle"):
            scheduler.validate(config)
        config["jobs"] = [job("one"), job("one")]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            scheduler.validate(config)

    def test_unknown_dependency_and_impossible_budget_rejected(self):
        config = configuration()
        config["jobs"][0]["deps"] = ["absent"]
        with self.assertRaisesRegex(ValueError, "dependency"):
            scheduler.plan(config)
        config["jobs"][0]["deps"] = []
        config["jobs"][0]["resources"]["memory_mb"] = 50000
        with self.assertRaisesRegex(ValueError, "resource-feasible"):
            scheduler.plan(config)

    def test_string_command_and_unknown_placeholder_fail_before_dispatch(self):
        config = configuration()
        for command in ("python wrapper.py", ["{missing}"]):
            config["jobs"][0]["command"] = command
            with mock.patch.object(scheduler, "rpc") as rpc:
                with self.assertRaises(ValueError):
                    scheduler.plan(config)
                rpc.assert_not_called()

    def test_busy_gpu_is_allowed_when_memory_fits(self):
        config = configuration()
        allocations = scheduler.allocations(config["jobs"][0], config["hosts"][0], [], live())
        self.assertEqual(allocations[0]["gpu"], "0")

    def test_parallel_objectives_prefer_distinct_gpus(self):
        config = configuration()
        config["hosts"][0]["gpus"].append({"index": 1, "memory_mb": 20000})
        config["jobs"] = [job("timing"), job("thermal")]
        rows = scheduler.plan(config)["jobs"]
        self.assertEqual([row["gpu"] for row in rows], ["0", "1"])
        self.assertEqual({row["estimated_start"] for row in rows}, {0})

    def test_parallel_objectives_may_share_only_admissible_gpu(self):
        config = configuration()
        config["hosts"][0]["gpus"].append({"index": 1, "memory_mb": 20000})
        hosts = {host["id"]: host for host in config["hosts"]}
        sample = {**live(), "gpu_free_mb": {"0": 20000, "1": 3000}}
        first = scheduler.choose(job("timing"), hosts, [], {"host-a": sample})
        second = scheduler.choose(job("thermal"), hosts, [first], {"host-a": sample})
        self.assertEqual((first["gpu"], second["gpu"]), ("0", "0"))
        self.assertEqual((first["slot"], second["slot"]), ("01", "02"))

    def test_gpu_tie_break_is_numeric_and_independent_of_config_order(self):
        config = configuration()
        host = config["hosts"][0]
        for indices in ((10, 2), (2, 10)):
            with self.subTest(indices=indices):
                host["gpus"] = [{"index": i, "memory_mb": 20000} for i in indices]
                self.assertEqual(scheduler.choose(job("one"), {host["id"]: host}, [])["gpu"], "2")

    def test_gpu_reserved_memory_precedes_job_count(self):
        config = configuration()
        host = config["hosts"][0]
        host.update(slots=4, gpus=[{"index": 0, "memory_mb": 20000}, {"index": 1, "memory_mb": 20000}])
        reservations = [
            {"host": host["id"], "slot": f"{i + 1:02d}", "gpu": gpu,
             "resources": {**job("reserved")["resources"], "gpu_memory_mb": memory}}
            for i, (gpu, memory) in enumerate((("0", 6000), ("1", 2000), ("1", 2000)))
        ]
        self.assertEqual(scheduler.choose(job("one"), {host["id"]: host}, reservations)["gpu"], "1")

    def test_equal_gpu_memory_prefers_fewer_reserved_jobs(self):
        config = configuration()
        host = config["hosts"][0]
        host.update(slots=4, gpus=[{"index": 0, "memory_mb": 20000}, {"index": 1, "memory_mb": 20000}])
        reservations = [
            {"host": host["id"], "slot": f"{i + 1:02d}", "gpu": gpu,
             "resources": {**job("reserved")["resources"], "gpu_memory_mb": memory}}
            for i, (gpu, memory) in enumerate((("0", 2000), ("0", 2000), ("1", 4000)))
        ]
        self.assertEqual(scheduler.choose(job("one"), {host["id"]: host}, reservations)["gpu"], "1")

    def test_live_cpu_memory_disk_gpu_limits_and_stale_probe(self):
        config = configuration()
        for key, value in (("cpu_idle_cores", 3), ("memory_available_mb", 3999),
                           ("disk_free_mb", 999), ("gpu_free_mb", {"0": 3999}),
                           ("sampled_at", time.time() - 61)):
            with self.subTest(key=key):
                sample = {**live(), key: value}
                self.assertEqual(scheduler.allocations(config["jobs"][0], config["hosts"][0], [], sample), [])

    def test_reservations_protect_not_yet_allocated_memory_and_slots(self):
        config = configuration()
        host, item = config["hosts"][0], config["jobs"][0]
        first = scheduler.allocations(item, host, [], live())[0]
        first["status"] = "unknown"
        sample = {**live(), "memory_available_mb": 7000}
        self.assertEqual(scheduler.allocations(item, host, [first], sample), [])
        second = scheduler.allocations(item, host, [first], live())[0]
        self.assertEqual(second["slot"], "02")
        self.assertEqual(scheduler.allocations(item, host, [first, second], live()), [])

    def test_failure_transitively_blocks_dependencies_not_independent_work(self):
        jobs = {j["id"]: j for j in [job("a"), job("b", deps=["a"]), job("c", deps=["b"]), job("d")]}
        state = {name: {"status": "pending"} for name in jobs}
        state["a"] = {"status": "failed"}
        scheduler.block_dependencies(jobs, state)
        self.assertEqual([state[k]["status"] for k in "abcd"], ["failed", "blocked", "blocked", "pending"])

    def test_unknown_dependency_waits_not_retries_or_fails(self):
        jobs = {j["id"]: j for j in [job("a"), job("b", deps=["a"])]}
        state = {"a": {"status": "unknown"}, "b": {"status": "pending"}}
        scheduler.block_dependencies(jobs, state)
        self.assertEqual(scheduler.ready(jobs, state), [])
        self.assertEqual(state["b"]["status"], "pending")

    def test_rendered_command_is_argv_with_explicit_slot(self):
        config = configuration("/tmp/root with space")
        host, item = config["hosts"][0], config["jobs"][0]
        host["slot_ids"] = ["east", "west"]
        a = scheduler.allocations(item, host, [])[0]
        spec = scheduler.make_spec(config, item, host, a)
        self.assertEqual(spec["slot"], "east")
        self.assertEqual(spec["command"][1], "/tmp/root with space/wrapper.py")
        self.assertEqual(spec["command"][-1], "/tmp/root with space/test-batch/jobs/one")

    def test_local_rpc_does_not_need_ssh(self):
        host = {**configuration()["hosts"][0], "ssh": "local"}
        with mock.patch.object(scheduler.subprocess, "run", return_value=mock.Mock(stdout='{"status":"queued"}')) as process:
            self.assertEqual(scheduler.rpc(host, "--dispatch", {"test": 1})["status"], "queued")
        self.assertEqual(process.call_args.args[0], [host["python"], host["script"], "--dispatch"])
        self.assertNotIn("shell", process.call_args.kwargs)

    def test_remote_rpc_quotes_argv_and_never_forwards_auth(self):
        host = {**configuration()["hosts"][0], "script": "/source with space/scheduler.py"}
        with mock.patch.object(scheduler.subprocess, "run", return_value=mock.Mock(stdout='{"status":"queued"}')) as process:
            scheduler.rpc(host, "--dispatch", {"test": 1})
        argv = process.call_args.args[0]
        self.assertEqual(argv[0], "ssh")
        self.assertIn("'/source with space/scheduler.py'", argv[-1])
        self.assertNotIn("-A", argv)

    def specification(self, path):
        config = configuration(str(path))
        host, item = config["hosts"][0], config["jobs"][0]
        return scheduler.make_spec(config, item, host, scheduler.allocations(item, host, [])[0])

    def test_dispatch_duplicate_does_not_start_second_tmux(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = self.specification(Path(temp))
            with mock.patch.object(scheduler.subprocess, "run", return_value=mock.Mock(returncode=0)) as process:
                self.assertEqual(scheduler.dispatch(spec)["status"], "queued")
                self.assertEqual(scheduler.dispatch(spec)["status"], "queued")
                calls = [call.args[0] for call in process.call_args_list]
                self.assertEqual(sum("new-session" in call for call in calls), 1)

    def test_dispatch_failure_keeps_intent_and_never_relaunches(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = self.specification(Path(temp))
            with mock.patch.object(scheduler.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["tmux"])):
                with self.assertRaises(subprocess.CalledProcessError):
                    scheduler.dispatch(spec)
            with mock.patch.object(scheduler.subprocess, "run", return_value=mock.Mock(returncode=1)) as process:
                self.assertEqual(scheduler.dispatch(spec)["status"], "unknown")
                self.assertNotIn("new-session", process.call_args.args[0])

    def test_changed_spec_and_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = self.specification(Path(temp))
            altered = {**spec, "scheduler_sha256": "wrong"}
            with self.assertRaisesRegex(ValueError, "source differs"):
                scheduler.dispatch(altered)
            scheduler.write(Path(spec["job_dir"]) / "spec.json", altered)
            with self.assertRaisesRegex(ValueError, "specification differs"):
                scheduler.inspect(spec)

    def test_worker_result_and_at_most_once(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = self.specification(Path(temp))
            path = Path(spec["job_dir"]) / "spec.json"
            scheduler.write(path, spec)
            with mock.patch.object(scheduler, "probe", return_value=live()), mock.patch.object(
                    scheduler.subprocess, "run", return_value=mock.Mock(returncode=0)) as process:
                self.assertEqual(scheduler.worker(path), 0)
                self.assertEqual(scheduler.worker(path), 0)
                self.assertEqual(process.call_count, 1)
            result = scheduler.inspect(spec)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["spec_sha256"], scheduler.digest(spec))

    def test_incomplete_started_worker_is_not_rerun(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = self.specification(Path(temp))
            path = Path(spec["job_dir"]) / "spec.json"
            scheduler.write(path, spec)
            scheduler.write(path.parent / "started.json", {"pid": 1})
            with mock.patch.object(scheduler.subprocess, "run") as process:
                self.assertEqual(scheduler.worker(path), 0)
                process.assert_not_called()
            self.assertFalse((path.parent / "result.json").exists())

    def test_worker_resource_race_is_failed_without_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = self.specification(Path(temp))
            path = Path(spec["job_dir"]) / "spec.json"
            scheduler.write(path, spec)
            with mock.patch.object(scheduler, "probe", return_value={**live(), "memory_available_mb": 0}), mock.patch.object(scheduler.subprocess, "run") as process:
                self.assertEqual(scheduler.worker(path), 1)
                process.assert_not_called()
            self.assertEqual(scheduler.read(path.parent / "result.json")["status"], "failed")

    def test_central_lock_prevents_second_controller(self):
        with tempfile.TemporaryDirectory() as temp:
            with scheduler.lock(Path(temp) / "lock"):
                with self.assertRaises(BlockingIOError):
                    scheduler.lock(Path(temp) / "lock")

    def test_ssh_disconnect_is_reconciled_without_duplicate_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            config = configuration(temp)
            state_dir = Path(temp) / "state"
            calls = []
            def rpc(host, action, payload):
                calls.append(action)
                if action == "--probe":
                    return live()
                if action == "--dispatch":
                    raise TimeoutError("Lost SSH acknowledgement")
                return {"status": "unknown", "reason": "Remote state not proven"}
            with mock.patch.object(scheduler, "rpc", side_effect=rpc):
                scheduler.run(config, state_dir, once=True)
                scheduler.run(config, state_dir, once=True)
            self.assertEqual(calls.count("--dispatch"), 1)
            self.assertEqual(scheduler.read(state_dir / "state.json")["one"]["status"], "unknown")
            record = scheduler.read(state_dir / "state.json")["one"]
            completed = scheduler.reconcile(record, {"status": "success", "spec_sha256": scheduler.digest(record["spec"])})
            self.assertEqual(completed["status"], "success")
            with self.assertRaisesRegex(ValueError, "match"):
                scheduler.reconcile(record, {"status": "success", "spec_sha256": "wrong"})
            with self.assertRaisesRegex(ValueError, "identity"):
                scheduler.reconcile(record, {"status": "success"})

    def test_config_change_cannot_resume_same_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            config = configuration(temp)
            state_dir = Path(temp) / "state"
            with mock.patch.object(scheduler, "rpc", side_effect=TimeoutError("offline")):
                scheduler.run(config, state_dir, once=True)
            config["jobs"][0]["estimated_seconds"] = 20
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                scheduler.run(config, state_dir, once=True)


if __name__ == "__main__":
    unittest.main()
