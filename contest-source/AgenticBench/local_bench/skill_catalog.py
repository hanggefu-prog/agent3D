"""Validated registry for benchmark-owned analyzer, task, and infrastructure Skills."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILL_ROOT = PROJECT_ROOT / "skills"
CATEGORIES = frozenset({"analyzer", "task_agent", "infrastructure"})


@dataclass(frozen=True)
class SkillRecord:
    name: str
    directory: str
    category: str
    capabilities: Tuple[str, ...]
    requires: Tuple[str, ...]
    root: Path

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "directory": self.directory,
            "category": self.category,
            "capabilities": list(self.capabilities),
            "requires": list(self.requires),
            "skill": str(self.root / "SKILL.md"),
        }


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("Skill is missing YAML frontmatter: {}".format(path))
    try:
        closing = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError("Skill frontmatter is not closed: {}".format(path)) from exc
    value = yaml.safe_load("\n".join(lines[1:closing]))
    if not isinstance(value, dict):
        raise ValueError("Skill frontmatter must be an object: {}".format(path))
    if set(value) != {"name", "description"}:
        raise ValueError("Skill frontmatter must contain only name and description: {}".format(path))
    if not isinstance(value["description"], str) or not value["description"].strip():
        raise ValueError("Skill description is empty: {}".format(path))
    return value


def load_skill_catalog(skill_root: Path = DEFAULT_SKILL_ROOT) -> Tuple[SkillRecord, ...]:
    root = skill_root.expanduser().resolve()
    catalog_path = root / "catalog.yaml"
    data = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"schema_version", "skills"}:
        raise ValueError("skill catalog must contain schema_version and skills")
    if int(data["schema_version"]) != 1 or not isinstance(data["skills"], list):
        raise ValueError("unsupported or malformed skill catalog")

    records = []
    names = set()
    directories = set()
    required_keys = {"name", "directory", "category", "capabilities", "requires"}
    for row in data["skills"]:
        if not isinstance(row, dict) or set(row) != required_keys:
            raise ValueError("each skill catalog row must contain {}".format(sorted(required_keys)))
        name = str(row["name"])
        directory = str(row["directory"])
        category = str(row["category"])
        capabilities = tuple(str(item) for item in row["capabilities"])
        requires = tuple(str(item) for item in row["requires"])
        if not name or name in names:
            raise ValueError("skill names must be unique and nonempty: {!r}".format(name))
        if (
            not directory
            or directory in directories
            or Path(directory).name != directory
            or directory in {".", ".."}
        ):
            raise ValueError("skill directories must be unique single names: {!r}".format(directory))
        if category not in CATEGORIES:
            raise ValueError("unsupported skill category {!r}".format(category))
        if not capabilities or len(set(capabilities)) != len(capabilities):
            raise ValueError("skill capabilities must be nonempty and unique: {}".format(name))
        if len(set(requires)) != len(requires) or name in requires:
            raise ValueError("skill requirements must be unique and non-recursive: {}".format(name))
        skill_dir = root / directory
        metadata = _frontmatter(skill_dir / "SKILL.md")
        if metadata.get("name") != name:
            raise ValueError("catalog/frontmatter name mismatch for {}".format(directory))
        records.append(
            SkillRecord(
                name=name,
                directory=directory,
                category=category,
                capabilities=capabilities,
                requires=requires,
                root=skill_dir,
            )
        )
        names.add(name)
        directories.add(directory)

    disk_directories = {
        path.name for path in root.iterdir() if path.is_dir() and (path / "SKILL.md").is_file()
    }
    if disk_directories != directories:
        raise ValueError(
            "skill catalog and filesystem differ: missing={} extra={}".format(
                sorted(disk_directories - directories), sorted(directories - disk_directories)
            )
        )
    for record in records:
        unknown = set(record.requires) - names
        if unknown:
            raise ValueError("{} requires unknown Skills: {}".format(record.name, sorted(unknown)))
    return tuple(records)


def skill_by_name(records: Iterable[SkillRecord], name: str) -> SkillRecord:
    matches = [record for record in records if record.name == name]
    if len(matches) != 1:
        raise KeyError("unknown or ambiguous Skill {!r}".format(name))
    return matches[0]


def skill_by_directory(records: Iterable[SkillRecord], directory: str) -> SkillRecord:
    matches = [record for record in records if record.directory == directory]
    if len(matches) != 1:
        raise KeyError("unknown or ambiguous Skill directory {!r}".format(directory))
    return matches[0]


def required_skills(records: Iterable[SkillRecord], name: str) -> Tuple[SkillRecord, ...]:
    """Return transitive requirements once, in dependency-first order."""
    rows = tuple(records)
    ordered = []
    visiting = set()
    visited = set()

    def visit(current: str) -> None:
        if current in visited:
            return
        if current in visiting:
            raise ValueError("cyclic Skill requirements include {}".format(current))
        visiting.add(current)
        record = skill_by_name(rows, current)
        for dependency in record.requires:
            visit(dependency)
            if dependency not in visited:
                raise AssertionError("dependency traversal failed")
        visiting.remove(current)
        visited.add(current)
        if current != name:
            ordered.append(record)

    visit(name)
    return tuple(ordered)
