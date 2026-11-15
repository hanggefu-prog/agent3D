"""Offline control-plane tests; these are not native EDA qualification."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grt_bench import ENDPOINT
from grt_bench.common import digest, records, verify_records
from grt_bench.contract import DEFAULT_SEEDS, DEFAULT_SCORING, METRICS, OBJECTIVES, matrix, validate
from grt_bench.pareto import aggregate, frontier
from grt_bench.queue import Queue
from grt_bench.candidates import freeze, inventory, materialize, verify_scope, verify_snapshot
from grt_bench.runtime import docker_command
from grt_bench.flow import resolved_config, bind_method_seeds
from grt_bench.spef import add_name_map, validate_connectivity
from grt_bench.prefix import prepare as prepare_prefix


def task_fixture():
    return {"schema_version": 2, "id": "fixture", "endpoint": ENDPOINT,"evaluator_sha256":"a"*64,
        "objectives": list(OBJECTIVES), "scoring": copy.deepcopy(DEFAULT_SCORING),
        "metrics": list(METRICS), "opt_passes": 1, "activity": 0.1, "ambient_k": 318.15, "grid_size": 10,
        "seeds": copy.deepcopy(DEFAULT_SEEDS), "epsilon": 0,
        "normalization_floors": {m: 1 for m in METRICS},
        "cases": [{"id": c, "design": c, "pdk": "ng45", "weight": 1, "split": "train",
                   "config_sha256": "a" * 64, "input_identity": "b" * 64} for c in ("bp_fe", "bp")]}


class ContractTests(unittest.TestCase):
    def test_parallel_seeds_build_one_prefix_and_reject_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            controls = [root / ('control' + str(i)) for i in range(3)]
            for control in controls: control.mkdir()
            payload = {'candidate':'c'*64, 'task_digest':'t'*64, 'case':'bp', 'seed':DEFAULT_SEEDS[0]}
            resources = {'cpu':4,'memory_gb':8,'storage_gb':10,'gpu_count':0,'build_slots':0}
            def execute(command, output, name, **kwargs):
                import time
                time.sleep(.05)
                for stage in ('00_input','10_floorplan'):
                    stage_dir = output.parent / 'output/run' / stage
                    stage_dir.mkdir(parents=True)
                    (stage_dir / 'stage_manifest.json').write_text('{}')
                return {'exit_code':0}
            def one(i):
                return prepare_prefix(root,dict(payload,seed=DEFAULT_SEEDS[i]),'sha256:'+'a'*64,
                    resources,[],[],controls[i],{},lambda:None)
            with patch('grt_bench.prefix.run_container',side_effect=execute) as run:
                with ThreadPoolExecutor(max_workers=3) as pool: results = list(pool.map(one,range(3)))
                self.assertEqual(run.call_count,1)
            self.assertEqual(sum(reused for _,reused in results),2)
            self.assertEqual(len({str(path) for path,_ in results}),1)
            (results[0][0] / '00_input/stage_manifest.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError): one(0)

    def test_two_cases_three_paired_seeds(self):
        task = task_fixture()
        rows = matrix(task, "c" * 64)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({digest(r) for r in rows}), 6)
        self.assertTrue(all(r["seed"]["io"] == 1 for r in rows))

    def test_submission_interleaves_cases_without_changing_cells(self):
        from grt_bench.__main__ import submission_order
        task = task_fixture()
        rows = submission_order(task, "c" * 64)
        self.assertEqual({digest(r) for r in rows}, {digest(r) for r in matrix(task, "c" * 64)})
        for offset in range(0, len(rows), len(task["cases"])):
            batch = rows[offset:offset + len(task["cases"])]
            self.assertEqual(len({r["case"] for r in batch}), len(task["cases"]))
            self.assertEqual(len({r["seed"]["id"] for r in batch}), 1)

    def test_reject_mislabeled_endpoint_and_duplicate_seed(self):
        for field, value in (("endpoint", "post-cts"), ("activity", 0.2), ("grid_size", 20), ("opt_passes", 2)):
            t = task_fixture(); t[field] = value
            with self.assertRaises(ValueError): validate(t)
        t = task_fixture(); t["seeds"][1] = dict(t["seeds"][0], id="s2")
        with self.assertRaises(ValueError): validate(t)

    def test_no_fictional_opt_seed(self):
        t = task_fixture(); t["seeds"][0]["opt"] = 3
        with self.assertRaises(ValueError): validate(t)

    def test_case_identity_and_seed_injection(self):
        t = task_fixture()
        c = resolved_config({"design": "bp_fe", "pdk": "ng45", "type": "lol"}, t["cases"][0], t["seeds"][1])
        self.assertEqual(c["partition"]["seed"], 1002)
        self.assertEqual(c["placement"]["seed"], 1002)
        with self.assertRaises(ValueError): resolved_config({"design": "wrong", "pdk": "ng45", "type": "lol"}, t["cases"][0], t["seeds"][0])

    def test_seeds_reach_native_method_configs_without_changing_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"partition.json").write_text('{"partition":{"seed":42,"keep":1}}')
            (root/"placement.json").write_text('{"concurrent_dreamplace":{"initialization_seed":42}}')
            for seed in DEFAULT_SEEDS:
                out=root/seed["id"];out.mkdir()
                config={"partition":{"algorithm":"tp-gnn-flat","config":str(root/"partition.json")},
                        "placement":{"algorithm":"concurrent-dreamplace","config":str(root/"placement.json")}}
                bind_method_seeds(config,seed,out)
                self.assertEqual(json.loads((out/"partition.json").read_text())["partition"],{"seed":seed["partition"],"keep":1})
                self.assertEqual(json.loads((out/"placement.json").read_text())["concurrent_dreamplace"]["initialization_seed"],seed["placement"])
            self.assertEqual(json.loads((root/"partition.json").read_text())["partition"]["seed"],42)


class ParetoTests(unittest.TestCase):
    def setUp(self):
        self.task = task_fixture()
        self.rows = [{**r, "status": "verified", "metrics": {m: 2.0 for m in METRICS}}
                     for r in matrix(self.task, "c" * 64)]
        self.ref = [{**r, "candidate": "d" * 64} for r in self.rows]

    def test_reference_normalization_and_pairing(self):
        result = aggregate(self.task, "c" * 64, self.rows, self.ref)
        self.assertEqual(set(result["vector"].values()), {1.0})
        self.rows[0]["metrics"]["overflow"] = 0
        self.ref[0] = {**self.ref[0], "metrics": {m: 0 for m in METRICS}}
        self.assertTrue(all(v >= 0 for v in aggregate(self.task, "c" * 64, self.rows, self.ref)["vector"].values()))

    def test_failed_missing_extra_nonfinite_and_wrong_seed_are_fatal(self):
        mutations = [lambda r: r.pop(), lambda r: r.append(r[0]),
            lambda r: r[0].update(status="failed"), lambda r: r[0]["metrics"].update(power_w=float("nan")),
            lambda r: r[0]["seed"].update(partition=999)]
        for mutation in mutations:
            rows = copy.deepcopy(self.rows); mutation(rows)
            with self.assertRaises(ValueError): aggregate(self.task, "c" * 64, rows, self.ref)

    def row(self, name, values):
        return {"candidate": name, "task_digest": "t", "reference_digest": "r", "split": "train",
                "feasible": True, "vector": dict(zip(OBJECTIVES, values))}

    def test_tradeoffs_retained_and_dominated_removed(self):
        a, b, c = self.row("a", [1,1,1]), self.row("b", [.8,1,1.2]), self.row("c", [2,2,2])
        self.assertEqual({r["candidate"] for r in frontier([a,b,c])}, {"a", "b"})
        self.assertEqual(frontier([a,b,c]), frontier([c,b,a]))

    def test_epsilon_boxes_and_test_set_isolation(self):
        a, b = self.row("a", [1.001]*3), self.row("b", [1.002]*3)
        self.assertEqual(len(frontier([b,a], .01)), 1)
        b["split"] = "test"
        with self.assertRaises(ValueError): frontier([a,b])
        b["split"] = "train"; b["reference_digest"] = "other"
        with self.assertRaises(ValueError): frontier([a,b])


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.queue = Queue(Path(self.temp.name) / "queue.sqlite")
        self.capacity = {"cpu":32,"memory_gb":96,"storage_gb":200,"gpus":["0","1"],"build_slots":1}
        self.request = {"cpu":16,"memory_gb":48,"storage_gb":40,"gpu_count":1,"build_slots":0}
        self.queue.register_host("rafa1", self.capacity)

    def test_parallel_claim_has_no_duplicate_gpu_or_job(self):
        for i in range(8): self.queue.submit({"cell": i}, self.request)
        with ThreadPoolExecutor(max_workers=8) as pool:
            leases = [x for x in pool.map(lambda _: self.queue.claim("rafa1"), range(8)) if x]
        self.assertEqual(len(leases),2)
        self.assertEqual({x["gpus"][0] for x in leases},{"0","1"})
        self.assertEqual(len({x["id"] for x in leases}),2)
        first=leases[0]; self.queue.finish(first["id"],first["token"],{"ok":True},True)
        self.assertIsNotNone(self.queue.claim("rafa1"))

    def test_submit_deduplicates_and_restart_preserves_lease(self):
        job=self.queue.submit({"cell":1},self.request)
        self.assertEqual(job,self.queue.submit({"cell":1},self.request))
        lease=self.queue.claim("rafa1")
        reopened=Queue(self.queue.path)
        self.assertEqual(reopened.rows()[0]["state"],"running")
        with self.assertRaises(ValueError): reopened.finish(job,"wrong",{},True)
        reopened.finish(job,lease["token"],{},False)
        self.assertIsNone(reopened.claim("rafa1"))

    def test_expiry_does_not_release_gpu_or_replay(self):
        job=self.queue.submit({"cell":1},self.request); lease=self.queue.claim("rafa1")
        with patch("grt_bench.queue.time.time",return_value=10**12):
            self.assertEqual(self.queue.rows()[0]["state"],"uncertain")
        with self.assertRaises(ValueError): self.queue.finish(job,lease["token"],{},True)
        with self.assertRaises(ValueError): self.queue.reconcile(job,{})
        self.queue.reconcile(job,{"job":job,"terminated":True,"evidence_sha256":"a"*64})
        self.assertEqual(self.queue.rows()[0]["state"],"interrupted")
        self.assertIsNone(self.queue.claim("rafa1"))

    def test_cancel_running_reserves_capacity_until_stopped(self):
        job=self.queue.submit({"cell":1},self.request); lease=self.queue.claim("rafa1")
        self.queue.cancel(job)
        self.assertEqual(self.queue.rows()[0]["state"],"cancelling")
        with self.assertRaises(ValueError): self.queue.heartbeat(job,lease["token"])

    def test_build_slot_and_cross_host_admission(self):
        build=dict(self.request,gpu_count=0,build_slots=1)
        self.queue.register_host("other",self.capacity)
        self.queue.submit({"host":"rafa1","build":1},build)
        self.queue.submit({"host":"rafa1","build":2},build)
        self.assertIsNone(self.queue.claim("other"))
        self.assertIsNotNone(self.queue.claim("rafa1"))
        self.assertIsNone(self.queue.claim("rafa1"))


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.source=self.root/"source"; self.source.mkdir()
        (self.source/"algo.py").write_text("x=1\n")
        self.store=self.root/"store"
        self.policy={"allowed_paths":["algo.py"],"native":False,"allow_create":False,"allow_delete":False}

    def test_parent_recoverability_and_mutation_detection(self):
        parent=freeze(self.source,self.store)
        (self.source/"algo.py").write_text("x=2\n")
        child=freeze(self.source,self.store,parent=parent,policy=self.policy)
        self.assertNotEqual(parent,child)
        verify_snapshot(self.store/parent)
        materialize(self.store/child,self.root/"trial")
        self.assertEqual((self.root/"trial/algo.py").read_text(),"x=2\n")
        target=self.store/child/"tree/algo.py"; target.chmod(0o644); target.write_text("tampered")
        with self.assertRaises(ValueError): verify_snapshot(self.store/child)

    def test_symlink_and_outside_scope_rejected(self):
        before=inventory(self.source)
        (self.source/"other.py").write_text("x=3")
        with self.assertRaises(ValueError): verify_scope(before,inventory(self.source),self.policy)
        (self.source/"link").symlink_to("/etc/passwd")
        with self.assertRaises(ValueError): inventory(self.source)

    def test_trusted_internal_symlink_preserved_but_cannot_be_changed(self):
        (self.source/"link").symlink_to("algo.py")
        parent=freeze(self.source,self.store)
        materialize(self.store/parent,self.root/"trial")
        self.assertTrue((self.root/"trial/link").is_symlink())
        before=inventory(self.source)
        (self.source/"link").unlink()
        (self.source/"link").write_text("replacement")
        with self.assertRaises(ValueError): verify_scope(before,inventory(self.source),dict(self.policy,allowed_paths=["link"]))

    def test_artifact_path_traversal_and_corruption(self):
        evidence=records(self.source)
        verify_records(self.source,evidence)
        (self.source/"algo.py").write_text("new")
        with self.assertRaises(ValueError): verify_records(self.source,evidence)
        with self.assertRaises(ValueError): verify_records(self.source,{"../outside":next(iter(evidence.values()))})

    def test_embedded_tcl_requires_native_image_and_unsupported_build_rejected(self):
        before={"OpenROAD-3D/openroad/src/rsz/src/Resizer.tcl":{"sha256":"a","executable":False}}
        after=copy.deepcopy(before);next(iter(after.values()))["sha256"]="b"
        policy={"allowed_paths":["OpenROAD-3D/openroad/src/rsz"],"native":True}
        self.assertTrue(verify_scope(before,after,policy)["native_build_required"])
        before={"Place-3D/thirdparty/DREAMPlace/op.cpp":{"sha256":"a","executable":False}}
        after=copy.deepcopy(before);next(iter(after.values()))["sha256"]="b"
        with self.assertRaisesRegex(ValueError,"No installed native runtime"):
            verify_scope(before,after,{"allowed_paths":["Place-3D/thirdparty/DREAMPlace"],"native":True})


class RuntimeTests(unittest.TestCase):
    def test_transport_failure_never_releases_an_unconfirmed_container(self):
        import subprocess
        from grt_bench.runtime import run_container, UncertainExecution
        with tempfile.TemporaryDirectory() as d:
            process = Mock()
            process.wait.return_value = 125
            process.poll.return_value = 125
            unavailable = subprocess.CompletedProcess([], 1, '', 'Cannot connect to the Docker daemon')
            with patch('grt_bench.runtime.subprocess.Popen', return_value=process), patch('grt_bench.runtime.subprocess.run', return_value=unavailable):
                with self.assertRaises(UncertainExecution): run_container(['docker'], Path(d)/'run', 'grt-test')
            self.assertFalse(json.loads((Path(d)/'run/stop.json').read_text())['terminated'])
        from grt_bench.worker import loop
        client = Mock()
        client.call.return_value = {'id':'job', 'token':'token'}
        with patch('grt_bench.worker.evaluate', side_effect=UncertainExecution('unconfirmed')):
            with self.assertRaises(UncertainExecution): loop(client, {}, once=True)
        self.assertEqual([call.args[0] for call in client.call.call_args_list], ['claim'])

    def test_estimated_spef_names_reach_shared_passive_reader(self):
        from open3dbench.backend import post_route
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/"raw.spef";mapped=root/"mapped.spef";canonical=root/"canonical.spef"
            raw.write_text('*SPEF "ieee 1481-1999"\n*C_UNIT 1 FF\n*R_UNIT 1 KOHM\n*PORTS\nHBT_B2T_0_SRC O\n*D_NET a/b 2\n*CONN\n*I cell/name:Z O *D BUF\n*P HBT_B2T_0_SRC O\n*CAP\n1 a/b:0 1\n2 cell/name:Z 1\n*RES\n1 cell/name:Z a/b:0 0.1\n2 a/b:0 HBT_B2T_0_SRC 0.2\n*END\n')
            add_name_map(raw,mapped)
            self.assertEqual(post_route._spef_hbt_nets(mapped),{"a/b":["HBT_B2T_0_SRC"]})
            post_route._normalize_flat_spef(mapped,canonical,{"a/b":"HBT_B2T_0_SRC"})
            text=canonical.read_text()
            self.assertIn('cell__OPEN3D_SLASH__name',text)
            self.assertIn('*1 HBT_B2T_0_SRC',text)
            self.assertEqual(text.count('0.1'),1)
            self.assertEqual(text.count('0.2'),1)
            raw.write_text(raw.read_text().replace('2 a/b:0 HBT_B2T_0_SRC 0.2\n',''))
            with self.assertRaisesRegex(ValueError,'Disconnected'):
                validate_connectivity(raw)

    def test_pin_resource_and_isolation_contract(self):
        r={"cpu":16,"memory_gb":48,"gpu_count":1}
        cmd=docker_command("sha256:"+"a"*64,"grt-test",r,["0"],[],["-m","grt_bench.flow"])
        for flag in ("--network=none","--read-only","--pull=never","--cap-drop=ALL"):
            self.assertIn(flag,cmd)
        self.assertNotIn("/var/run/docker.sock", " ".join(cmd))
        with self.assertRaises(ValueError): docker_command("image:latest","grt-test",r,["0"],[],[])
        with self.assertRaises(ValueError): docker_command("sha256:"+"a"*64,"grt-test",r,[],[],[])


if __name__ == "__main__": unittest.main()
