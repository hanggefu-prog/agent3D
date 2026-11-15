import gzip
import json
from pathlib import Path

import pytest

from contest2.evaluation_files import publish
from contest2.agent_container import command


def test_full_logs_are_readable_without_exposing_private_or_binary_files(tmp_path):
    root = tmp_path.resolve()
    output = root / "native"
    (output / "60_opt").mkdir(parents=True)
    body = b"HEADER\n" + b"timing detail\n" * 100000 + b"FOOTER\n"
    (output / "60_opt/tool.log").write_bytes(body)
    (output / "60_opt/metrics.json").write_text('{"wns": -0.5}')
    (output / "60_opt/top.odb").write_bytes(b"database")
    (output / "control").mkdir()
    (output / "control/private.json").write_text('{"token": "private"}')
    secret = root / "secret.txt"
    secret.write_text("private")
    (output / "alias.log").symlink_to(secret)
    (output / "aliased").symlink_to(root, target_is_directory=True)
    feedback = {"status": "succeeded", "feedback": {"score": 3}}
    reply = publish(root / "visible", "one", feedback, [("cells/case/seed", output)])
    assert reply["artifacts_dir"] == "/workspace/evaluations/one"
    view = root / "visible/one"
    files = json.loads((view / "files.json").read_text())
    assert set(files) == {"cells/case/seed/60_opt/tool.log", "cells/case/seed/60_opt/metrics.json"}
    log = files["cells/case/seed/60_opt/tool.log"]
    assert gzip.decompress((view / log["path"]).read_bytes()) == body
    assert json.loads((view / "feedback.json").read_text()) == reply
    assert publish(root / "visible", "one", feedback, []) == reply
    with pytest.raises(ValueError):
        publish(root / "visible", "../escape", feedback, [])


def test_project_and_evaluation_mounts_are_readonly_and_separate(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    paths = {name: root / name for name in
             ("agent", "workspaces", "scratch", "bridge", "framework", "source", "evaluations")}
    for path in paths.values():
        path.mkdir()
    (root / "README.md").write_text("Contest guide")
    (paths["bridge"] / "bridge.sock").touch()
    monkeypatch.setattr(Path, "is_socket", lambda path: path.name == "bridge.sock")
    kwargs = dict(name="o3db-contest-agent-123456789abc", image="sha256:" + "a" * 64,
        agent_dir=paths["agent"], workspaces=paths["workspaces"], scratch=paths["scratch"],
        bridge_dir=paths["bridge"], framework=paths["framework"], source=paths["source"],
        evaluations=paths["evaluations"], training_inputs={})
    argv = command(**kwargs)
    mounts = [argv[i + 1] for i, item in enumerate(argv[:-1]) if item == "--mount"]
    for target in ("/opt/contest/source", "/workspace/evaluations", "/workspace/README.md"):
        assert any("target=" + target + ",readonly" in item for item in mounts)
    with pytest.raises(ValueError, match="overlap"):
        command(**{**kwargs, "evaluations": paths["scratch"]})



def test_concurrent_status_queries_publish_one_complete_directory(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    barrier = threading.Barrier(2)
    rename = Path.rename
    def together(path, target):
        barrier.wait(timeout=5)
        return rename(path, target)
    monkeypatch.setattr(Path, "rename", together)
    root = tmp_path.resolve() / "visible"
    feedback = {"status": "rejected", "reason": "invalid interface"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: publish(root, "one", feedback, []), range(2)))
    assert results[0] == results[1]
    assert json.loads((root / "one/feedback.json").read_text()) == results[0]
    assert list(root.iterdir()) == [root / "one"]
