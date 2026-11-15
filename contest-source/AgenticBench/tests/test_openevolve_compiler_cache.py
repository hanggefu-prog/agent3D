"""Persistent cache wiring and failure evidence for candidate builds."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import yaml
from unittest import mock

from scripts import openevolve_docker as launcher


class CompilerCacheWiringTest(unittest.TestCase):
    def test_plan_cache_is_writable_mounted_and_outside_candidate_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / 'reference'
            reference.mkdir()
            output = root / 'experiment'
            def export(_):
                bundle = output / 'bundle'
                bundle.mkdir()
                (bundle / 'config.yaml').write_text('llm: {}\n')
            with mock.patch.object(launcher, 'validate_benchmark_input'), \
                 mock.patch.object(launcher, 'task_main', side_effect=export), \
                 mock.patch.object(launcher, 'configure_deepseek'):
                status = launcher.main([
                    'global_routing_ripup_policy', '--profile', 'test',
                    '--benchmark-input', str(root / 'input'),
                    '--reference-run', str(reference), '--output-dir', str(output),
                    '--data-root', str(root), '--max-tokens', '32768',
                    '--llm-timeout', '600', '--plan-only'])
            self.assertEqual(status, 0)
            config = yaml.safe_load((output / "bundle/config.yaml").read_text())
            self.assertEqual(config["llm"]["max_tokens"], 32768)
            self.assertEqual(config["llm"]["timeout"], 600)
            self.assertIs(type(config["llm"]["timeout"]), int)
            env = json.loads((output / 'launch_plan.json').read_text())['container_environment']
            cache = Path(env['CCACHE_DIR'])
            self.assertEqual(cache, (output / 'openroad-ccache').resolve())
            self.assertTrue(cache.is_dir())
            self.assertEqual(cache.stat().st_mode & 0o777, 0o700)
            self.assertEqual(env['CCACHE_MAXSIZE'], '8G')
            self.assertEqual(env['CCACHE_COMPILERCHECK'], 'content')
            self.assertNotIn('evaluations', cache.parts)

    def test_failed_build_retains_after_statistics_and_exit_status(self):
        recipe = Path(launcher.__file__).with_name('build_openroad_candidate.sh')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            (source / 'CMakeLists.txt').touch()
            build = root / 'result/runtime/build'
            build.mkdir(parents=True)
            commands = root / 'commands'
            commands.mkdir()
            for name, text in [('ccache', '#!/bin/sh\necho cache-statistics\n'),
                               ('cmake', '#!/bin/sh\nexit 7\n')]:
                tool = commands / name
                tool.write_text(text)
                tool.chmod(0o755)
            result = subprocess.run(['bash', str(recipe), str(source), str(build), '1'],
                env={**os.environ, 'PATH': str(commands) + ':' + os.environ['PATH'],
                     'CCACHE_DIR': str(root / 'cache')}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7, result.stderr)
            for phase in ('before', 'after'):
                self.assertIn('cache-statistics',
                    (root / ('result/openroad-ccache-' + phase + '.log')).read_text())
