"""Small dispatcher preserving native subprocess/container isolation."""
import os
from pathlib import Path
import subprocess
import sys

COMMANDS = {
    'qualify': 'qualify_tasks.py',
    'reference': 'cts_reference_queue.py',
    'clock-context': 'prepare_local_contexts.py',
    'multi-context': 'prepare_multi_contexts.py',
    'multi-qualify': 'prepare_multi_qualification.py',
    'clock-final': 'clock_final_queue.py',
    'multi-final': 'multi_final_queue.py',
    'cts-final': 'cts_final_queue.py',
    'measure': 'measure_host.py',
    'develop': 'develop.py',
    'judge': 'judge_task.py',
}


def launch(command, arguments):
    script = Path(__file__).with_name('physical') / COMMANDS[command]
    environment = os.environ.copy()
    framework = str(Path(__file__).resolve().parents[1])
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, (framework, environment.get('PYTHONPATH'))))
    return subprocess.call([sys.executable, str(script), *arguments], env=environment)
