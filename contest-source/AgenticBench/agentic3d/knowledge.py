"""Explicit, content-addressed knowledge lifecycle for STA protocol v2.

No discovery of previous runs or user-home skills. Transfer archives contain
reusable skills/task definitions only, never sessions, candidates or raw logs.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\\" in value
            or str(path) != value or any(p.startswith(".") for p in path.parts)):
        raise ValueError("unsafe knowledge path: " + value)
    return value


def identifier(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", value):
        raise ValueError("expected lowercase hyphenated identifier")


def validate_skill(skill: dict) -> None:
    if set(skill) != {"id", "description", "files"}:
        raise ValueError("skill requires id, description, files")
    identifier(skill["id"])
    if not isinstance(skill["description"], str) or not skill["description"].strip():
        raise ValueError("skill description is required")
    files = skill["files"]
    if not isinstance(files, dict) or "SKILL.md" not in files or len(files) > 12:
        raise ValueError("skill requires SKILL.md and at most 12 files")
    if len(canonical(skill)) > 60000:
        raise ValueError("skill exceeds 60KB")
    for name, content in files.items():
        safe_path(name)
        if not isinstance(content, str):
            raise ValueError("skill resources must be text")
        if name != "SKILL.md" and not (
            name.startswith("references/") and name.endswith((".md", ".json"))
            or name.startswith("scripts/") and name.endswith(".py")
        ):
            raise ValueError("unsupported skill resource")
        if name.endswith(".py"):
            ast.parse(content)  # syntax only; never execute generated helpers here
    text = files["SKILL.md"]
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError("SKILL.md requires YAML frontmatter")
    metadata = yaml.safe_load(text.split("---", 2)[1])
    if not isinstance(metadata, dict) or metadata.get("name") != skill["id"] or not metadata.get("description"):
        raise ValueError("skill frontmatter name/description mismatch")


def read_package(path: Path) -> dict:
    value = json.loads(path.read_text())
    payload = value["payload"]
    if value.get("sha256") != digest(payload):
        raise ValueError("knowledge package checksum mismatch")
    if payload.get("version") != 2 or payload.get("phase") != "train":
        raise ValueError("only v2 training knowledge packages can be imported")
    cases = payload.get("training_cases")
    if not isinstance(cases, list) or not cases or any(not isinstance(c, str) for c in cases):
        raise ValueError("training lineage is required")
    for record in payload["records"]:
        validate_record(record)
        if record["provenance"].get("phase") != "train" or record["provenance"].get("case") not in cases:
            raise ValueError("record is not from declared training lineage")
    return value


def validate_record(record: dict) -> None:
    if set(record) != {"skills", "tasks", "provenance"}:
        raise ValueError("invalid knowledge record")
    for skill in record["skills"]:
        validate_skill(skill)
    if not isinstance(record["tasks"], list) or not isinstance(record["provenance"], dict):
        raise ValueError("invalid task/provenance record")


class KnowledgeStore:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.verify()

    @classmethod
    def initialize(cls, root: Path, *, seed_files: dict[str, str], case: str,
                   phase: str = "independent", split: dict | None = None,
                   imported: Path | None = None, adapt_test: bool = False) -> "KnowledgeStore":
        if phase not in {"independent", "train", "test"}:
            raise ValueError("invalid knowledge phase")
        if phase == "independent" and imported:
            raise ValueError("independent runs cannot import evolved knowledge")
        split = split or {"train": [], "test": []}
        if set(split) != {"train", "test"} or any(
            not isinstance(v, list) or any(not isinstance(x, str) or not x for x in v)
            or len(v) != len(set(v)) for v in split.values()
        ):
            raise ValueError("split must contain unique train/test case lists")
        if set(split["train"]) & set(split["test"]):
            raise ValueError("train/test overlap")
        if phase != "independent" and case not in split[phase]:
            raise ValueError("case not in declared split phase")
        if phase == "test" and not imported:
            raise ValueError("test requires an explicit training package")
        package = read_package(imported) if imported else None
        if package:
            lineage = package["payload"]["training_cases"]
            if not set(lineage) <= set(split["train"]) or set(lineage) & set(split["test"]):
                raise ValueError("training lineage conflicts with split")
            if package["payload"]["seed_sha256"] != digest(seed_files):
                raise ValueError("training package uses different seed knowledge")
        for name in seed_files:
            safe_path(name)
        manifest = dict(version=2, case=case, phase=phase, split=split,
                        adapt_test=adapt_test, seed_sha256=digest(seed_files),
                        imported_sha256=package["sha256"] if package else None)
        if (root / "manifest.json").exists():
            store = cls(root)
            if store.manifest != manifest:
                raise ValueError("knowledge resume configuration differs")
            return store
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "seed/snapshot.json", seed_files)
        if package:
            write_json(root / "imported/package.json", package)
        write_json(root / "manifest.json", manifest)
        return cls(root)

    def verify(self) -> None:
        seed = json.loads((self.root / "seed/snapshot.json").read_text())
        if digest(seed) != self.manifest["seed_sha256"]:
            raise ValueError("seed snapshot changed")
        if self.manifest["imported_sha256"]:
            package = read_package(self.root / "imported/package.json")
            if package["sha256"] != self.manifest["imported_sha256"]:
                raise ValueError("imported snapshot changed")

    def records(self) -> list[dict]:
        self.verify()
        imported = self.root / "imported/package.json"
        records = read_package(imported)["payload"]["records"] if imported.exists() else []
        # Frozen tests retain proposals for audit but never expose them in context.
        if self.manifest["phase"] != "test" or self.manifest["adapt_test"]:
            for path in sorted((self.root / "evolved").glob("*/record.json")):
                envelope = json.loads(path.read_text())
                if envelope["sha256"] != digest(envelope["record"]):
                    raise ValueError("evolved knowledge changed")
                validate_record(envelope["record"])
                records.append(envelope["record"])
        return records

    def save(self, event: str, skills: list, tasks: list, provenance: dict) -> None:
        identifier(event)
        record = dict(skills=skills, tasks=tasks, provenance={
            **provenance, "case": self.manifest["case"], "phase": self.manifest["phase"],
            "seed_sha256": self.manifest["seed_sha256"],
        })
        validate_record(record)
        area = "test-proposals" if self.manifest["phase"] == "test" and not self.manifest["adapt_test"] else "evolved"
        folder = self.root / area / event
        envelope = {"record": record, "sha256": digest(record)}
        path = folder / "record.json"
        if path.exists():
            if json.loads(path.read_text()) != envelope:
                raise ValueError("knowledge event changed on resume")
            return
        write_json(path, envelope)
        for skill in skills:
            for name, text in skill["files"].items():
                dest = folder / "skills" / skill["id"] / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(text, encoding="utf-8")
        for task in tasks:
            identifier(task["id"])
            write_json(folder / "tasks" / (task["id"] + ".json"), task)

    def export(self, destination: Path) -> None:
        if self.manifest["phase"] != "train":
            raise ValueError("only training runs may export transferable knowledge")
        records = self.records()
        cases = {self.manifest["case"]}
        imported = self.root / "imported/package.json"
        if imported.exists():
            cases.update(read_package(imported)["payload"]["training_cases"])
        payload = dict(version=2, phase="train", training_cases=sorted(cases),
                       seed_sha256=self.manifest["seed_sha256"], records=records)
        package = dict(payload=payload, sha256=digest(payload))
        if destination.exists():
            if json.loads(destination.read_text()) != package:
                raise ValueError("refusing to overwrite a different knowledge package")
            return
        write_json(destination, package)
