# Hierarchical optimization loop

Agentic3D uses one ask/tell vocabulary without treating the Global Analyzer as
a code-search backend. The Global Analyzer is a read-only task-modeling step
that runs beside the outer ask operation:

```text
user command + accepted whole-flow PPA
  -> Global Analyzer models bounded StageTasks
  -> each StageTask runs an inner ask/evaluate/tell search
  -> locally accepted LocalCorrectionResults
  -> outer ask deterministically aggregates compatible corrections
  -> one integrated whole-flow PPA evaluation
  -> outer tell records normalized global feedback
  -> next Global Analyzer modeling round
```

The code-defined contracts live in `agentic3d/hierarchical.py`.

## Global Analyzer output

`global_plan.local_tasks` contains one or more instances of maintained task
templates. A task model contains no source code or parameter values. It states
the hypothesis, objective, budget, evidence, and a scope that may only narrow
the selected template.

```json
{
  "task_id": "layout-preconditioner-round-3",
  "template": "placement_init_preconditioner",
  "stage": "layout",
  "hypothesis": "late density gradients are underweighted",
  "objective": {
    "goal": "reduce placement HPWL and overflow",
    "metric_weights": {"hpwl": 0.9, "overflow": 0.1}
  },
  "acceptance_gates": ["combined local score improves"],
  "correctness_gates": ["both dies remain legal"],
  "evaluation_budget": 20,
  "source_scope": [
    "Place-3D/thirdparty/DREAMPlace/dreamplace/PlaceObj.py"
  ],
  "config_scope": [],
  "context_artifacts": ["30_placement/placement.log"]
}
```

The maintained templates are currently:

| Template | Modeled work |
| --- | --- |
| `netlist_partition` | partition algorithm and Mt-KaHyPar controls |
| `tp_gnn_learning_objective` | differentiable TP-GNN embedding objective |
| `placement_init_preconditioner` | placement initialization and preconditioner source |
| `placement_parameters` | dynamic placement control |
| `global_routing_parameters` | die-by-die global-routing controls |
| `global_routing_net_ordering` | shared-or-die-specific FastRoute reroute ordering |
| `global_routing_ripup_policy` | edge-level FastRoute rip-up eligibility |
| `global_routing_maze_cost` | bounded Dijkstra maze-expansion cost |
| `global_routing_iteration_controller` | dynamic cost/search/rip-up schedule |

`StageTaskMaterializer` converts this overlay into a normal task directory
accepted by `local_bench.load_task`. It copies the small trusted task assets,
filters prompt source ranges, and rejects any requested source or configuration
path outside the template policy. Open3DBench source and benchmark results are
not copied into the Agentic3D repository.

## Inner loop result

Every local optimizer returns one `LocalCorrectionResult` containing:

- the exact `StageTask` contract;
- the accepted candidate payload;
- normalized local `CandidateFeedback`;
- verifier evidence and artifact references;
- the immutable Open3DBench base revision.

The inner optimizer may be EvoStage, OpenEvolve, a custom multi-agent system,
or any `SearchBackend`. The hierarchical coordinator does not inspect its
population or prompting implementation.

`FirstImprovementRunner` is the maintained inner-loop policy for experiments
that separate cheap stage feedback from expensive whole-flow evaluation. It
evaluates local proposals serially and stops at the first strict improvement
over the accepted local incumbent. Its local evaluation count is recorded in
the correction artifacts but does not consume the outer PPA/STA budget.

## Outer ask and tell

`CorrectionAggregationBackend.ask` combines locally accepted corrections. The
deterministic aggregation gate requires:

- one shared base revision;
- unique correction IDs;
- source paths inside each modeled scope;
- configuration paths inside each modeled scope;
- no two patches modifying the same file;
- no contradictory values at the same configuration or candidate-file path.

It returns one ordinary `CandidateProposal`. Its `program` field is a JSON
`IntegratedProposal`, so the existing `AskTellRunner` remains unchanged.

`IntegratedPPAEvaluator` decodes that proposal and invokes a supplied
whole-flow runner. The runner is responsible for creating an isolated checkout,
applying the aggregate, compiling as required, executing the configured
Open3DBench full flow, and enforcing PPA/correctness comparability. Its result
is normalized into `CandidateFeedback` and delivered to the aggregator through
`tell`. Exceptions and failed flows produce `score=None` rather than a valid
zero score.

```python
from agentic3d import IntegratedPPAEvaluator


def run_full_flow(integrated):
    # Apply integrated.candidate in an isolated checkout, run Open3DBench,
    # and return normalized whole-flow evidence.
    return {
        "score": 1.037,
        "metrics": {
            "combined_score": 1.037,
            "runs_successfully": 1.0,
            "hpwl": 2.31e6,
            "overflow": 0.018,
        },
        "artifacts": {"run_dir": "/data/.../global-round-003"},
    }


global_evaluator = IntegratedPPAEvaluator(run_full_flow)
```

## Fixed collaboration order

`HierarchicalRoundCoordinator.run_round` fixes the order in code:

1. call the Global Analyzer task modeler;
2. reject duplicate tasks and require one remaining outer global evaluation;
3. run one inner optimizer per modeled task;
4. reject corrections returned for the wrong task;
5. aggregate locally accepted corrections through outer `ask`;
6. run exactly one whole-flow evaluator;
7. deliver exactly one global feedback record through outer `tell`.

The resulting `GlobalRoundResult` contains the modeling context, modeled tasks,
local corrections, and complete outer ask/tell history. It is suitable as the
read-only input to the next Global Analyzer round.
