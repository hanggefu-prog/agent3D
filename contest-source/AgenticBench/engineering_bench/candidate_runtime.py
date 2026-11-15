"""Build CTS-native contributions once; independent measurement uses the base image."""
from pathlib import Path
import shutil
import subprocess
import time

from .protocol import digest, sha, write_new, read_json
from .submission import verify_seal


def verify(subject, sealed, receipt, base_image):
    from grt_bench.builder import verify_runtime
    from local_bench.workspace import openroad_source_fingerprint
    verify_seal(subject, sealed)
    if receipt['source_sha256'] != sealed['source_sha256'] or receipt['base_image'] != base_image:
        raise ValueError('Candidate runtime bound to another source or base image')
    native = Path(subject) / 'OpenROAD-3D/openroad'
    if receipt['manifest']['source_fingerprint'] != openroad_source_fingerprint(native):
        raise ValueError('Candidate native runtime source differs')
    verify_runtime(receipt)
    return receipt['image']


def build(subject, sealed, base_image, output, cache, threads=8):
    from grt_bench.builder import local_base_tag, verify_runtime
    from scripts.openroad_compiler_cache import compile_cached
    from local_bench.workspace import openroad_source_fingerprint
    verify_seal(subject, sealed)
    if sealed['task'] != 'cross_die_cts':
        raise ValueError('Only CTS has native engineering scope')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    native = Path(subject) / 'OpenROAD-3D/openroad'
    framework = Path(__file__).resolve().parents[1]
    recipe = framework / 'scripts/build_openroad_candidate.sh'
    started = time.monotonic()
    compiled = compile_cached(base_image=base_image, source_tree=native, builder=recipe,
        output=output / 'compile', cache_root=Path(cache), threads=threads)
    if compiled['source_fingerprint'] != openroad_source_fingerprint(native):
        raise ValueError('Source changed during native build')
    package = output / 'package'
    package.mkdir()
    shutil.copy2(compiled['binary_path'], package / 'openroad')
    (package / 'openroad').chmod(0o755)
    manifest = {'kind': 'engineering-installed-cts-v1', 'candidate': sealed['submission_sha256'],
        'binary_sha256': compiled['binary_sha256'], 'source_fingerprint': compiled['source_fingerprint'],
        'base_image': base_image, 'cache_key': compiled['cache_key'],
        'recipe_sha256': sha(recipe), 'consumer': 'candidate-cts/fixed-base-independent-scoring'}
    write_new(package / 'runtime.json', manifest)
    (package / 'Dockerfile').write_text('FROM ' + local_base_tag(base_image) +
        '\nUSER root\nCOPY openroad /usr/local/bin/openroad\nCOPY runtime.json /opt/agentic3d-grt-runtime.json\n')
    with (output / 'package.log').open('x') as log:
        subprocess.run(['docker', 'build', '--pull=false', '--network=none', '--iidfile',
            str(output / 'image.id'), str(package)], stdout=log, stderr=subprocess.STDOUT,
            check=True, timeout=600)
    receipt = {'submission_sha256': sealed['submission_sha256'], 'source_sha256': sealed['source_sha256'], 'image': (output / 'image.id').read_text().strip(),
        'base_image': base_image, 'binary_sha256': compiled['binary_sha256'], 'manifest': manifest,
        'wall_seconds': time.monotonic() - started}
    receipt['verification'] = verify_runtime(receipt)
    verify(subject, sealed, receipt, base_image)
    write_new(output / 'receipt.json', receipt)
    return receipt
