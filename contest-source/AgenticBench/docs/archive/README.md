# Archived AgenticBench reports

These files are kept byte-identical as design history. They are not current
task definitions and are not maintained.

| Archived file | Original path | Replaced by |
| --- | --- | --- |
| [TASK_SETTINGS_REPORT_zh_20261004.md](TASK_SETTINGS_REPORT_zh_20261004.md) | `AgenticBench/docs/TASK_SETTINGS_REPORT.md` | [TASK_SETTINGS_REPORT.md](../TASK_SETTINGS_REPORT.md) |
| [HARNESS_DESIGN_REPORT_zh_20261004.md](HARNESS_DESIGN_REPORT_zh_20261004.md) | `AgenticBench/docs/HARNESS_DESIGN_REPORT.md` | [HARNESS_DESIGN_REPORT.md](../HARNESS_DESIGN_REPORT.md) |

The archived reports are mostly in Chinese and carry a verification date of
2026-10-04, with dated amendments up to 2026-10-16. They describe task
directories that have since been retired, such as the combined
initialization/preconditioner task, net ordering and several TP-GNN variants.
Their relative links were written for `AgenticBench/docs/`, so many of them no
longer resolve from this directory, and some never resolved because their
targets were deleted.

The maintained reports were rewritten in English on 2026-10-19 against the
code at that time. English was chosen because the task prompts, `task.yaml`
files, the other `AgenticBench/docs/` guides and the root README are all in
English. Keeping one language means the documentation consistency checker can
compare the reports with the code.
