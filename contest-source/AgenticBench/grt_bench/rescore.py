"""Host-authorized analysis-only replay of identical, immutable physical runs."""
import argparse
import ast
import copy
from pathlib import Path
import time
from .common import read, sha, write
from .contract import validate, timing_only


def validate_transition(task, old_task, plan, old_plan):
    """Allow a named host-selected case subset and new scorer, never changed cells."""
    if validate(old_task) != old_plan["task_digest"] or validate(task) != plan["task_digest"]:
        raise ValueError("Rescore task identity differs")
    old = copy.deepcopy(old_task)
    objective_migration = timing_only(task) and not timing_only(old_task)
    if objective_migration:
        baseline = old_task.get('optimization', {}).get('baseline')
        if baseline is None or plan['candidate'] != baseline:
            raise ValueError('Objective migration requires original baseline evidence')
        if task['scoring']['timing_weights'] != old_task['scoring']['timing_weights']:
            raise ValueError('Objective migration changed timing weights')
        if task['normalization_floors'] != {k: old_task['normalization_floors'][k] for k in task['metrics']}:
            raise ValueError('Objective migration changed normalization floors')
        from .optimization import prompt_scope
        if prompt_scope(task['optimization']['prompt']) != prompt_scope(old_task['optimization']['prompt']):
            raise ValueError('Objective migration changed source scope')
        for key in ('schema_version', 'endpoint', 'metrics', 'objectives', 'scoring', 'normalization_floors'):
            old[key] = task[key]
        for key in ('activity', 'ambient_k', 'grid_size'):
            old.pop(key, None)
        for key in ('prompt', 'objective'):
            old['optimization'][key] = task['optimization'][key]
    old["id"] = task["id"]
    old["evaluator_sha256"] = task["evaluator_sha256"]
    selected = {case["id"] for case in task["cases"]}
    old["cases"] = [case for case in old_task["cases"] if case["id"] in selected]
    # A host may reserve formerly observed optimization cases for a new final
    # test split. Only original-reference evidence can cross this boundary;
    # previously optimized candidates must not be relabelled as held-out runs.
    splits = {case['id']: case['split'] for case in task['cases']}
    split_changed = any(case['split'] != splits[case['id']] for case in old['cases'])
    if split_changed:
        if plan['candidate'] != old_task.get('optimization', {}).get('baseline'):
            raise ValueError('Split migration requires original baseline evidence')
        if any(case['split'] != splits[case['id']] and
               (case['split'], splits[case['id']]) != ('train', 'test') for case in old['cases']):
            raise ValueError('Split migration may only reserve training cases for final tests')
        old['cases'] = [dict(case, split=splits[case['id']]) for case in old['cases']]
    if 'frontend_inputs' in task and 'frontend_inputs' not in old_task:
        if plan['candidate'] != old_task.get('optimization', {}).get('baseline'):
            raise ValueError('Only the original baseline can qualify a new fixed-input contract')
        # verify_physical independently checks BOTH original manifest hashes,
        # their complete artifact records, image, upstream chain and I/O seed.
        old['frontend_inputs'] = task['frontend_inputs']
    if old != task:
        raise ValueError("Rescore changed the physical task contract")
    if plan['endpoint'] != task['endpoint'] or old_plan['endpoint'] != old_task['endpoint']:
        raise ValueError('Rescore endpoint differs from its task')
    for key in ("candidate", "case", "seed") if objective_migration else ("candidate", "case", "seed", "endpoint"):
        if plan[key] != old_plan[key]:
            raise ValueError("Rescore changed " + key)
    if plan['split'] != old_plan['split'] and not split_changed:
        raise ValueError('Rescore changed split')
    case = next((c for c in task["cases"] if c["id"] == plan["case"]), None)
    if case is None or case["split"] != plan["split"] or plan["seed"] not in task["seeds"]:
        raise ValueError("Rescore cell is not in the selected task")


def validate_drivers(previous_framework, framework, old_task, task=None):
    from .common import evaluator_identity
    previous, current = Path(previous_framework) / "grt_bench", Path(framework) / "grt_bench"
    if evaluator_identity(previous) != old_task["evaluator_sha256"]:
        raise ValueError("Previous evaluator identity changed")
    objective_migration = bool(task and timing_only(task) and not timing_only(old_task))
    migration = bool(task and 'frontend_inputs' in task and 'frontend_inputs' not in old_task)
    qualified_previous = {
        'flow.py': '1ee228cef3974ee01bb75ac56ab52e29559dcd9e65a293b95be35088b99dd7f8',
        'contract.py': 'd542ffcf330d496346eb7fcdbf9054dbe0e6c34869c315f94ef92a461524f97c',
    }
    for name in ("flow.py", "runtime.py", "prefix.py", "contract.py"):
        if sha(previous / name) != sha(current / name):
            if objective_migration and name == 'contract.py':
                # validate_transition permits only the explicitly enumerated
                # analysis fields; every physical task setting remains identical.
                continue
            if objective_migration and name == 'flow.py':
                def physical_ast(path):
                    tree = ast.parse(path.read_text())
                    for node in ast.walk(tree):
                        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                                and node.func.id == 'run' and len(node.args) == 4
                                and ast.dump(node.args[3]) == ast.dump(ast.parse('output / "analysis"', mode='eval').body)):
                            node.keywords = [k for k in node.keywords if k.arg != 'endpoint']
                    return ast.dump(tree, include_attributes=False)
                if physical_ast(previous / name) == physical_ast(current / name):
                    continue
            if not migration or qualified_previous.get(name) != sha(previous / name):
                raise ValueError("Physical execution driver changed: " + name)
    if migration:
        # This migration fixes the input boundary, not scoring or extraction.
        for name in ('endpoint.py', 'latch_audit.py'):
            if sha(previous / name) != sha(current / name):
                raise ValueError('Fixed-input qualification changed measurement code: ' + name)
        helper = 'scripts/workspace_cts_thermal.py'
        if sha(previous.parent / helper) != sha(current.parent / helper):
            raise ValueError('Fixed-input qualification changed power/thermal extraction')


def validate_plan(plan, task):
    proof = plan["rescore"]
    if sha(proof["previous_plan"]) != proof["previous_plan_sha256"]:
        raise ValueError("Previous physical plan changed")
    previous = read(proof["previous_plan"])
    validate_transition(task, read(proof["previous_task"]), plan, previous)


def execute(plan_path):
    from .common import evaluator_identity
    from .endpoint import run
    from .verify import verify_physical
    from open3dbench.pdks import get_pdk
    plan = read(plan_path)
    task = read(plan["task"])
    if evaluator_identity() != task["evaluator_sha256"]:
        raise ValueError("Evaluator differs from frozen task")
    validate_plan(plan, task)
    case = next(c for c in task["cases"] if c["id"] == plan["case"])
    verify_physical(plan, task, case)
    output = Path(plan["output"])
    platform = Path(plan["source"]) / "OpenROAD-3D/flow/platforms" / (get_pdk(case["pdk"]).openroad_platform + "_3D")
    started = time.monotonic()
    run(Path(plan["physical_output"]) / "grt/global_route_checkpoint.json", platform,
        Path(plan["hotspot"]), output / "analysis", endpoint=task["endpoint"])
    write(output / "execution.json", {"kind":"immutable-GRT-analysis-replay", "physical_execution_reused":True,
        "source_job":plan["rescore"]["source_job"], "source_plan_sha256":plan["rescore"]["previous_plan_sha256"],
        "analysis_seconds":time.monotonic()-started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    execute(parser.parse_args().plan)
