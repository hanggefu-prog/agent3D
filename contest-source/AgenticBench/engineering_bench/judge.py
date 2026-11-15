"""Independent task-specific review. The judge cannot alter physical results.

Review version 1 (kept to revalidate old replies) cites {"path", "line"}.
Review version 2 additionally requires the quoted text of the cited line, so a
citation is checked against the file contents, not only the line count. Version
2 also adds a judge-family independence check, deterministic redaction of model
names/ids/authorship, and a >=3 sample median panel (see model.judge_panel).
The judge never contributes to the objective headline (weight 0).
"""
from pathlib import Path
import json
import os
import re
import shutil
import statistics

from .tasks import relative

REVIEW_VERSIONS = (1, 2)
CURRENT_REVIEW_VERSION = 2
MIN_JUDGE_SAMPLES = 3
MIN_QUOTE_CHARACTERS = 8

# Model family -> lowercase name prefixes. A configurable JSON file with the
# same shape replaces this map (judge_task --model-family-map).
DEFAULT_FAMILIES = {
    "anthropic": ["claude", "opus", "sonnet", "haiku", "fable"],
    "deepseek": ["deepseek"],
    "google": ["gemini", "gemma"],
    "meta": ["llama"],
    "mistral": ["mistral", "mixtral", "codestral"],
    "openai": ["gpt", "codex", "o1", "o3", "o4", "chatgpt"],
    "qwen": ["qwen", "qwq"],
    "xai": ["grok"],
}


def validate_families(families):
    if not isinstance(families, dict) or not families:
        raise ValueError("Model family map must be a non-empty object")
    owners = {}
    for family, prefixes in families.items():
        if not isinstance(family, str) or not family.strip():
            raise ValueError("Invalid model family name")
        if not isinstance(prefixes, list) or not prefixes:
            raise ValueError("Model family needs name prefixes: " + family)
        for prefix in prefixes:
            if not isinstance(prefix, str) or not prefix.strip() or prefix != prefix.strip().lower():
                raise ValueError("Model family prefixes must be lowercase strings: " + family)
            if owners.setdefault(prefix, family) != family:
                raise ValueError("Model prefix assigned to two families: " + prefix)
    return families


def load_families(path=None):
    if path is None:
        return DEFAULT_FAMILIES
    return validate_families(json.loads(Path(path).read_text()))


def model_family(model, families=None):
    """Family of a model id by its longest matching prefix, or None.

    A provider path (``openai/gpt-5``) or date/version suffix does not matter;
    the final path component is matched at a name boundary.
    """
    families = validate_families(DEFAULT_FAMILIES if families is None else families)
    if not isinstance(model, str) or not model.strip():
        return None
    name = model.strip().lower().rsplit("/", 1)[-1]
    best = None
    for family, prefixes in families.items():
        for prefix in prefixes:
            if name == prefix or (name.startswith(prefix) and not name[len(prefix)].isalpha()):
                if best is None or len(prefix) > len(best[1]):
                    best = (family, prefix)
    return best[0] if best else None


def check_judge_independence(judge_model, developer_model, families=None, allow_same_family_reason=None):
    """Fail closed unless judge and developer are known, different families.

    Same or unknown families need an explicit, non-empty recorded reason.
    """
    families = DEFAULT_FAMILIES if families is None else families
    judge_family = model_family(judge_model, families)
    developer_family = model_family(developer_model, families)
    independent = judge_family is not None and developer_family is not None and judge_family != developer_family
    reason = allow_same_family_reason
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("A same-family override needs a non-empty recorded reason")
    if not independent and reason is None:
        if judge_family is None or developer_family is None:
            raise ValueError("Unknown model family for judge/developer; independence cannot be shown")
        raise ValueError("Judge and developer belong to the same model family: " + judge_family)
    return {"judge_model": judge_model, "developer_model": developer_model,
            "judge_family": judge_family, "developer_family": developer_family,
            "independent": independent,
            "override_reason": None if independent else reason.strip()}


_AUTHORSHIP = re.compile(r"(?i)\b(co-authored-by|signed-off-by|authored-by|generated[ -](?:with|by)|written[ -]by|reviewed-by)\b\s*:?.*$")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_UUID = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
_SESSION = re.compile(r"(?i)\b((?:thread|session|conversation|request|run|chat)[_-]?id)\b(\s*[:=]\s*)\S+")
# Model-like tokens need a family word plus a version suffix (gpt-5, claude-3.5,
# llama3) so that ordinary identifiers are not touched.
_MODEL = re.compile(r"(?i)(?<![A-Za-z0-9])(?:(?:gpt|claude|gemini|gemma|llama|codex|deepseek|qwen|mistral|mixtral|grok)(?:[-_. ]?(?:\d[\w.]*|mini|pro|turbo|sonnet|opus|haiku|flash|instruct|coder|preview|max|nano|sol))+|o[134]-(?:mini|pro|preview|high|low))(?![A-Za-z0-9])")
_PRODUCT = re.compile(r"(?i)(?<![A-Za-z0-9])(?:claude code|chatgpt|github copilot|copilot|codex cli|gemini cli)(?![A-Za-z0-9])")


def redact_text(text, names=()):
    """Deterministic, line-preserving removal of model names/ids/authorship.

    ``names`` are additional literal strings (the actual developer and judge
    model ids). Each line keeps its position so citations remain valid.
    """
    literals = sorted({n.strip() for n in names if isinstance(n, str) and n.strip()}, key=lambda n: (-len(n), n))
    literal = re.compile("|".join(re.escape(n) for n in literals), re.IGNORECASE) if literals else None
    lines = text.split("\n")
    result = []
    for line in lines:
        line = _AUTHORSHIP.sub(lambda m: m.group(1) + ": [REDACTED-AUTHOR]", line)
        line = _EMAIL.sub("[REDACTED-EMAIL]", line)
        line = _SESSION.sub(lambda m: m.group(1) + m.group(2) + "[REDACTED-ID]", line)
        line = _UUID.sub("[REDACTED-ID]", line)
        if literal is not None:
            line = literal.sub("[REDACTED-MODEL]", line)
        line = _MODEL.sub("[REDACTED-MODEL]", line)
        line = _PRODUCT.sub("[REDACTED-MODEL]", line)
        result.append(line)
    return "\n".join(result)


def redact_value(value, names=()):
    """Redact every string in a JSON-like value (keys are left unchanged)."""
    if isinstance(value, str):
        return redact_text(value, names)
    if isinstance(value, list):
        return [redact_value(v, names) for v in value]
    if isinstance(value, dict):
        return {k: redact_value(v, names) for k, v in value.items()}
    return value


def prepare_review_view(subject, changed_paths, destination, names=()):
    """Read-only judge view of a sealed submission with redacted changed files.

    Unchanged files are hard links (or copies across devices) of the sealed
    tree; Git metadata is excluded because it records authorship and is
    replaced by an empty repository (the container runtime requires a
    standalone .git; the reviewed diff is provided separately in /task). Each
    changed text file is replaced by its line-preserving redaction. The sealed
    subject itself is never modified. Returns a host-side receipt (not for the
    judge).
    """
    subject, destination = Path(subject).resolve(strict=True), Path(destination)
    def link(source, target):
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    shutil.copytree(subject, destination, symlinks=True, copy_function=link,
                    ignore=shutil.ignore_patterns(".git"))
    redacted = []
    for name in sorted(changed_paths):
        relative(name)
        target = destination / name
        if not target.is_file() or target.is_symlink():
            continue
        try:
            original = target.read_text()
        except UnicodeDecodeError:
            continue
        clean = redact_text(original, names)
        if clean != original:
            target.unlink()
            target.write_text(clean)
            redacted.append(name)
    git = destination / ".git"
    for directory in ("objects/info", "objects/pack", "refs/heads", "refs/tags"):
        (git / directory).mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "config").write_text("[core]\n\trepositoryformatversion = 0\n\tfilemode = true\n\tbare = false\n")
    return {"view": str(destination), "redacted_paths": redacted, "git_metadata_excluded": True,
            "line_preserving": True}


def legality_checks(checks):
    """Status map for reviewer evidence; an unverified check is never passed."""
    status = {}
    for name, value in sorted(checks.items()):
        if isinstance(value, dict) and value.get("status") in ("passed", "failed", "unverified"):
            state = value["status"]
            if state == "passed" and value.get("passed") is not True:
                state = "unverified"
        else:
            state = "passed" if isinstance(value, dict) and value.get("passed") is True else "failed"
        status[name] = state
    return {"status_map": status, "unverified": [k for k, v in status.items() if v == "unverified"],
            "all_passed": bool(status) and all(v == "passed" for v in status.values())}


def aggregate_samples(reviews, min_samples=MIN_JUDGE_SAMPLES):
    """Median panel score with dispersion; too few graded samples is no score."""
    if type(min_samples) is not int or min_samples < MIN_JUDGE_SAMPLES:
        raise ValueError("The judge panel needs at least three samples")
    scores = [r["score"] for r in reviews if r.get("status") == "graded" and type(r.get("score")) is int]
    record = {"aggregation": "median", "samples": len(reviews), "graded_samples": len(scores),
              "scores": scores, "weight_in_headline": 0}
    if len(scores) < min_samples:
        record.update(status="insufficient_samples", score=None, dispersion=None)
        return record
    median = statistics.median(scores)
    record.update(status="graded", score=median, dispersion={
        "min": min(scores), "max": max(scores), "range": max(scores) - min(scores),
        "pstdev": statistics.pstdev(scores),
        "median_absolute_deviation": statistics.median([abs(s - median) for s in scores])})
    return record


def review_prompt(task, version=1):
    if version not in REVIEW_VERSIONS:
        raise ValueError("Unknown review version")
    if version == 2:
        return task.judge_instruction + """

## Review workspace

The anonymous submitted code is mounted read-only at /workspace. Model names,
session ids and authorship lines have been redacted. The trusted task definition
and diff are in /task. Functional and legality evidence is in /results; it does
not contain the developer model identity or QoR rankings. Use only the container
terminal to inspect those files. Do not run builds, candidate scripts or physical
evaluations. There is no developer conversation to continue. Review this task on
its own rubric and return the requested JSON.

Use this exact result shape (no other fields):
{"status":"graded", "score":3, "summary":"...",
 "strengths":[{"explanation":"...", "evidence":[{"path":"relative/file.py", "line":1, "quote":"exact text from that line"}]}],
 "issues":[], "unverified":["..."]}
Every evidence quote must be copied verbatim from the cited line (at least eight
non-space characters, or the whole line if it is shorter); quotes are checked
against the file. For unavailable code use status "ungradable" and score null.
Empty strengths or issues are fine, but a graded review must cite at least one
inspected location.
"""
    return task.judge_instruction + """

## Review workspace

The anonymous submitted code is mounted read-only at /workspace. The trusted
task definition and diff are in /task. Functional and legality evidence is in
/results; it does not contain the developer model identity or QoR rankings.
Use only the container terminal to inspect those files. Do not run builds,
candidate scripts or physical evaluations. There is no developer conversation
to continue. Review this task on its own rubric and return the requested JSON.

Use this exact result shape (no other fields):
{"status":"graded", "score":3, "summary":"...",
 "strengths":[{"explanation":"...", "evidence":[{"path":"relative/file.py", "line":1}]}],
 "issues":[], "unverified":["..."]}
For unavailable code use status "ungradable" and score null. Empty strengths or
issues are fine, but a graded review must cite at least one inspected location.
"""


def _squash(text):
    return " ".join(text.split())


def validate_review(text, evidence_roots, *, version=1):
    if version not in REVIEW_VERSIONS:
        raise ValueError("Unknown review version")
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    def reject(value):
        raise ValueError("Nonfinite review JSON")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate review field: " + key)
            result[key] = value
        return result
    data = json.loads(text, parse_constant=reject, object_pairs_hook=unique)
    expected = {"status", "score", "summary", "strengths", "issues", "unverified"}
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError("Invalid review fields")
    if data["status"] == "graded":
        if type(data["score"]) is not int or not 1 <= data["score"] <= 5:
            raise ValueError("Judge score must be an integer from 1 to 5")
    elif data["status"] == "ungradable":
        if data["score"] is not None:
            raise ValueError("Ungradable evidence must not receive a score")
    else:
        raise ValueError("Invalid review status")
    if not isinstance(data["summary"], str) or not data["summary"].strip():
        raise ValueError("Missing review summary")
    if not isinstance(data["unverified"], list) or any(not isinstance(x, str) for x in data["unverified"]):
        raise ValueError("Invalid unverified list")
    evidence_count = 0
    for field in ("strengths", "issues"):
        if not isinstance(data[field], list):
            raise ValueError("Invalid review evidence list")
        for item in data[field]:
            if not isinstance(item, dict) or set(item) != {"explanation", "evidence"} or not isinstance(item.get("explanation"), str) or not item["explanation"].strip():
                raise ValueError("Evidence needs an explanation")
            if not isinstance(item.get("evidence"), list) or not item["evidence"]:
                raise ValueError("A finding requires inspected evidence")
            for evidence in item["evidence"]:
                keys = {"path", "line", "quote"} if version == 2 else {"path", "line"}
                if not isinstance(evidence, dict) or set(evidence) != keys:
                    raise ValueError("Evidence must contain a relative path and line"
                                     + (" and quote" if version == 2 else ""))
                path = relative(evidence["path"])
                line = evidence["line"]
                if type(line) is not int or line < 1:
                    raise ValueError("Evidence line must be positive")
                quote = None
                if version == 2:
                    if not isinstance(evidence["quote"], str) or not _squash(evidence["quote"]):
                        raise ValueError("Evidence quote must be non-empty text")
                    quote = _squash(evidence["quote"])
                found = False
                for root in evidence_roots:
                    root = Path(root).resolve(strict=True)
                    candidate = (root / path).resolve()
                    try:
                        candidate.relative_to(root)
                    except ValueError:
                        continue
                    if candidate.is_file():
                        lines = candidate.read_text().splitlines()
                        found = line <= len(lines)
                        if found and quote is not None:
                            cited = _squash(lines[line - 1])
                            long_enough = len(quote.replace(" ", "")) >= MIN_QUOTE_CHARACTERS or quote == cited
                            found = long_enough and quote in cited
                        if found:
                            break
                if not found:
                    raise ValueError(("Quoted text does not match cited line: " if quote is not None
                                      else "Unverifiable evidence location: ") + path)
                evidence_count += 1
    if data["status"] == "graded" and not evidence_count:
        raise ValueError("A graded review needs evidence, not only an assertion")
    return data
