"""Launch bounded server-side imports using existing SSH authentication."""
import argparse
from pathlib import Path
import subprocess
from stream_install import NODES, ssh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('node', choices=NODES)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--component', choices=('runtime', 'sta-dependencies', 'sta-framework'), default='runtime')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    ip, port, node_root = NODES[args.node]
    script_name = {'runtime': 'install_shared_runtime.sh', 'sta-dependencies': 'install_sta_dependencies.sh',
                   'sta-framework': 'install_sta_framework.sh'}[args.component]
    script = Path(__file__).with_name(script_name).read_bytes()
    if args.attempt < 1:
        raise ValueError('attempt must be positive')
    suffix = '' if args.attempt == 1 else '_attempt' + str(args.attempt)
    stem = {'runtime': 'shared_runtime', 'sta-dependencies': 'sta_dependencies', 'sta-framework': 'sta_framework'}[args.component]
    log = root / 'analysis' / ('compute_pool_' + args.node + '_' + stem + '_20260908' + suffix + '.log')
    with log.open('x') as output:
        process = subprocess.Popen(ssh(ip, port, 'bash -s -- ' + node_root),
                                   stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        process.stdin.write(script)
        process.stdin.close()
    print('pid={} node={} log={}'.format(process.pid, args.node, log))


if __name__ == '__main__':
    main()
