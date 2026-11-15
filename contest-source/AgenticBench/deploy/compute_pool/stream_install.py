"""Stream native ta1 dependencies through authenticated SSH, bypassing slow NAS.

No private key is copied and no listening service is exposed on either server.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

POOL = '/nas1/gcr/agentic3d-dev/compute-pool-20260908'
EXPECTED = 'sha256:fe4689919f8f474c77d09bca8210b1755ae4fd862b8d3200d43d690986cd3262'
NODES = {'ta2': ('180.209.6.83', 22, '/data/gcr/agentic3d-compute-20260908'),
         'ta3': ('180.209.6.62', 22, '/data/gcr/agentic3d-compute-20260908'),
         'amax3': ('10.26.17.10', 223, '/data/agentic3d/agentic3d-compute-20260908')}


def ssh(ip, port, command):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4',
            '-o', 'ProxyCommand=docker exec -i nju-easyconnect busybox nc %h %p',
            '-p', str(port), 'gcr@' + ip, 'bash -o pipefail -c ' + shlex.quote(command)]


def pipe(source, destination):
    producer = subprocess.Popen(source, stdout=subprocess.PIPE)
    consumer = subprocess.Popen(destination, stdin=producer.stdout)
    producer.stdout.close()
    consumer_code = consumer.wait()
    if consumer_code:
        producer.terminate()
    producer_code = producer.wait()
    if producer_code or consumer_code:
        raise RuntimeError('transfer failed: producer={}, consumer={}'.format(producer_code, consumer_code))


def extract_and_mark(root, marker):
    """Commit completion on the transfer connection, not a fragile second SSH hop."""
    if marker not in {'git-objects', 'source'}:
        raise ValueError('unsupported completion marker')
    return ('set -e; zstd -dc | tar --no-same-owner -xf - -C '
            + shlex.quote(root + '/data') + '; date -Is > '
            + shlex.quote(root + '/' + marker + '.ready'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('node', choices=NODES)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--git-only', action='store_true')
    args = parser.parse_args()
    if args.detach:
        suffix = '_git_stream.log' if args.git_only else '_stream.log'
        log = Path(__file__).resolve().parents[2] / 'analysis' / ('compute_pool_' + args.node + suffix)
        with log.open('x') as output:
            command = [sys.executable, '-u', __file__, args.node]
            if args.git_only:
                command.append('--git-only')
            proc = subprocess.Popen(command,
                                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        print('pid={} log={}'.format(proc.pid, log))
        return
    ip, port, root = NODES[args.node]
    def target(command):
        return ssh(ip, port, command)
    subprocess.run(target('mkdir -p {0}/data; test ! -f {0}/installed.ready'.format(root)), check=True)
    if args.git_only:
        print('streaming pinned alternate Git objects to ' + args.node, flush=True)
        source = ('tar -C /data/gcr/agentic3d-dev -cf - '
                  'runs/evolution-20260830/open3dbench-source/.git/objects | zstd -q -T2 -3')
        pipe(ssh('180.209.0.53', 22, source), target(extract_and_mark(root, 'git-objects')))
        print('git objects installed', flush=True)
        return
    ready = subprocess.run(target('test -f ' + root + '/runtime.ready')).returncode == 0
    if not ready:
        print('streaming runtime to ' + args.node, flush=True)
        container = 'agentic3d-stream-export-' + args.node + '-20260908'
        source = ('set -e; test "$(docker image inspect agentic3d/open3dbench-v2-runtime:offline '
                  '--format "{{.Id}}")" = ' + shlex.quote(EXPECTED) + '; '
                  'docker create --name ' + container + ' agentic3d/open3dbench-v2-runtime:offline /bin/true >/dev/null; '
                  'trap "docker rm ' + container + ' >/dev/null" EXIT; '
                  'docker export ' + container + ' | zstd -q -T2 -3')
        destination = ('zstd -dc | docker import '
                       '--change ' + shlex.quote('LABEL org.agentic3d.source-image=' + EXPECTED)
                       + ' --change ' + shlex.quote('ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')
                       + ' --change ' + shlex.quote('ENV LD_LIBRARY_PATH=/opt/legacy-libs')
                       + ' --change ' + shlex.quote('ENV LANG=C.UTF-8')
                       + ' - agentic3d/compute-pool-runtime:20260908')
        pipe(ssh('180.209.0.53', 22, source), target(destination))
        subprocess.run(target('printf "%s\\n" ' + shlex.quote(EXPECTED) + ' > ' + root + '/runtime.ready'), check=True)
    ready = subprocess.run(target('test -f ' + root + '/source.ready')).returncode == 0
    if not ready:
        print('streaming source and native frozen inputs to ' + args.node, flush=True)
        source = ('tar -C /data/gcr/agentic3d-dev -cf - '
                  'open3dbench-source-grt-order-20260831-5round EvoStage-main '
                  'runs/grt_net_order_50round_20260902/inputs/bp '
                  'runs/grt_net_order_ariane133_20260901_r5/benchmark-input | zstd -q -T2 -3')
        pipe(ssh('180.209.0.53', 22, source), target(extract_and_mark(root, 'source')))
    print('starting node installation and qualification', flush=True)
    subprocess.run(target('bash ' + POOL + '/tools/install_node.sh ' + POOL + ' ' + root + ' ' + args.node), check=True)


if __name__ == '__main__':
    main()
