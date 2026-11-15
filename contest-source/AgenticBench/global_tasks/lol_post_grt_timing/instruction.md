---
components: [partition, placement, hbt, timing_opt, cts, grt]
---

Improve the source code and parameters of the two-die LoL backend through
multi-stage co-optimization, with particular attention to partition and placement.
Use the supplied optimization cases and paired seeds. Minimize normalized timing
(0.7 WNS + 0.3 TNS) plus 0.1 times the per-cell overflow penalty, min-max normalized
from 300 to 600 and clipped to 0–1 before seed/case averaging. Do not evaluate
power or temperature. Keep the fixed flow, physical constraints and one OPT pass.
Explore freely in the selected parent development environment using the supplied
optimization cases, within resource limits. Final submissions follow the task-bound
source permissions and pass the independent evaluator. Return one candidate per round.
