from local_bench.openevolve_adapter import evaluate_from_task_file


def evaluate(program_path):
    return evaluate_from_task_file(__file__, program_path)
