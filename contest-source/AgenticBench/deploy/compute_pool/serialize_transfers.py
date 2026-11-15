"""Drain the already-started bounded transfers one host at a time.

The campus VPN becomes unresponsive with all six archive streams active. Pause
only our explicitly verified detached process groups; never touch experiment
containers. Resume each next node once the previous data transfer has finished.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {'ta3': (33762, 36753), 'ta2': (34062, 36707), 'amax3': (34090, 36746)}


def ours(pid, node):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='],
                            stdout=subprocess.PIPE, text=True)
    if not result.stdout.strip():
        return False
    if 'deploy/compute_pool/stream_install.py ' + node not in result.stdout:
        raise RuntimeError('refusing to signal reused or unrelated PID ' + str(pid))
    if os.getpgid(pid) != pid:
        raise RuntimeError('expected an independent session: ' + str(pid))
    return True


def send(node, sig):
    for pid in GROUPS[node]:
        if ours(pid, node):
            os.killpg(pid, sig)


def main():
    if '--detach' in sys.argv:
        log = ROOT / 'analysis/compute_pool_transfer_queue.log'
        with log.open('x') as stream:
            proc = subprocess.Popen([sys.executable, '-u', __file__],
                                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        print('queue_pid={} log={}'.format(proc.pid, log))
        return
    for node, pids in GROUPS.items():
        for pid in pids:
            ours(pid, node)
    paused = set()
    try:
        for node in ('ta2', 'amax3'):
            paused.add(node)
            send(node, signal.SIGSTOP)
            print(json.dumps({'node': node, 'transfer': 'queued'}), flush=True)
        for node in ('ta3', 'ta2', 'amax3'):
            send(node, signal.SIGCONT)
            paused.discard(node)
            print(json.dumps({'node': node, 'transfer': 'active'}), flush=True)
            started = time.monotonic()
            log = ROOT / 'analysis' / ('compute_pool_' + node + '_stream.log')
            while ours(GROUPS[node][0], node):
                if 'starting node installation and qualification' in log.read_text():
                    break
                if time.monotonic() - started > 6 * 3600:
                    raise TimeoutError('transfer exceeded six hours: ' + node)
                time.sleep(15)
            print(json.dumps({'node': node, 'transfer': 'left_data_phase'}), flush=True)
    finally:
        for node in paused:
            send(node, signal.SIGCONT)


if __name__ == '__main__':
    main()
