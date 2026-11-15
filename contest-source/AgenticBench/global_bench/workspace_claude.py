"""Claude Code turns with isolated tools and bounded transport recovery.

Retries continue the same session, source workspace and parent receipt. A pending
tool operation is never replayed. Candidate evaluation remains outside this code.
"""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid

from grt_bench.common import read, write

TOOLS = ['mcp__terminal__exec', 'mcp__terminal__checkout_parent', 'mcp__terminal__gpu_check']


def client_transport_error(event):
    """Recognize the observed Claude client marker, never a provider model."""
    message = event.get('message')
    if not isinstance(message, dict):
        return None
    text = 'API Error: Connection lost mid-response. The response above may be incomplete.'

    def zero_usage(value):
        if value is None:
            return True
        if type(value) in (int, float):
            return value == 0
        return isinstance(value, dict) and all(zero_usage(v) for v in value.values())

    usage = message.get('usage')
    if (event.get('type') == 'assistant' and event.get('is_api_error_message') is True
            and event.get('error') == 'server_error' and message.get('model') == '<synthetic>'
            and message.get('role') == 'assistant'
            and message.get('content') == [{'type': 'text', 'text': text}]
            and isinstance(usage, dict) and usage.get('input_tokens') == 0
            and usage.get('output_tokens') == 0 and zero_usage(usage)):
        return text
    return None


def inspect_events(path, session):
    result, observed, pending, tool_calls, models = None, None, set(), 0, set()
    errors = []; client_transport_errors = 0
    for line in Path(path).read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get('session_id'):
            if observed and event['session_id'] != observed:
                raise ValueError('Claude session changed within one attempt')
            observed = event['session_id']
        message = event.get('message') or {}
        if isinstance(message, dict):
            transport_error = client_transport_error(event)
            if transport_error:
                errors.append(transport_error); client_transport_errors += 1
            elif message.get('model'):
                models.add(message['model'])
            content = message.get('content', [])
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict): continue
                if block.get('type') == 'tool_use':
                    pending.add(block['id']); tool_calls += 1
                elif block.get('type') == 'tool_result':
                    pending.discard(block.get('tool_use_id'))
        if event.get('type') == 'result':
            result = event
            models.update(event.get('modelUsage', {}))
            errors.extend(event.get('errors', []))
            if event.get('is_error') and event.get('result'):
                errors.append(event['result'])
        if event.get('type') == 'error' or event.get('subtype') == 'api_error':
            errors.append(str(event.get('error', event)))
    valid = observed == session
    return {'result': result, 'session_valid': valid, 'pending_tools': sorted(pending),
            'session_mismatch': observed is not None and not valid,
            'tool_calls': tool_calls, 'models': sorted(models), 'errors': errors,
            'client_transport_errors': client_transport_errors}


def retryable(summary, stderr, timed_out=False):
    if timed_out or summary['pending_tools'] or summary.get('session_mismatch'):
        return False
    text = '\n'.join(map(str, summary['errors'])) + '\n' + stderr
    # Auth, invalid models, malformed arguments and tools are not transport failures.
    if re.search(r'\b(400|401|403|404)\b|invalid.api.key|authentication|unknown model|invalid.model', text, re.I):
        return False
    return bool(re.search(r'\b(408|429|500|502|503|504|529)\b|overload|rate.limit|connection.error|connection (reset|lost)|ECONNRESET|ETIMEDOUT|fetch failed|network.error|socket hang up|stream.*(disconnect|interrupt)|request timed out', text, re.I))


def build_command(binary, model, session, resume, mcp, schema):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,199}', model):
        raise ValueError('Invalid explicit model identifier')
    uuid.UUID(session)
    return [binary, '--print', '--bare', '--restricted', '--verbose',
            '--output-format', 'stream-json', '--model', model, '--effort', 'high',
            '--tools', '', '--strict-mcp-config', '--mcp-config', str(mcp),
            '--permission-mode', 'dontAsk', '--allowedTools', ','.join(TOOLS),
            '--json-schema', json.dumps(schema, separators=(',', ':')),
            '--resume' if resume else '--session-id', session]


def run_claude_turn(*, binary, model, control, output, framework, tool_config,
                    output_schema, provider_config, config_directory, python,
                    prompt, thread_id=None, timeout_seconds=3600,
                    environment=None, max_attempts=5, backoff_seconds=10):
    if not 1 <= max_attempts <= 8 or not 0 <= backoff_seconds <= 60:
        raise ValueError('Invalid bounded transport retry policy')
    output, control = Path(output), Path(control)
    output.mkdir(parents=True, exist_ok=False)
    credentials = read(provider_config)
    if set(credentials) != {'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL'} or not all(credentials.values()):
        raise ValueError('Missing private provider configuration')
    if Path(provider_config).stat().st_mode & 0o077:
        raise ValueError('Private provider configuration must be owner-only')
    env = dict(environment or os.environ)
    env.update(credentials)
    env.update(CLAUDE_CONFIG_DIR=str(config_directory), ENABLE_TOOL_SEARCH='false',
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1', ANTHROPIC_MODEL=model)
    Path(config_directory).mkdir(parents=True, exist_ok=True, mode=0o700)
    mcp = control/'claude-mcp.json'
    write(mcp, {'mcpServers': {'terminal': {'command': python,
        'args': ['-B', '-m', 'global_bench.grt_round_tools', '--config', str(tool_config)],
        'env': {'PYTHONPATH': str(framework), 'ANTHROPIC_AUTH_TOKEN': '', 'ANTHROPIC_API_KEY': ''}}}})
    session = thread_id or str(uuid.uuid4())
    write(output/'session.json', {'session_id':session,'resumed':bool(thread_id),
                                'model':model,'max_client_attempts':max_attempts})
    (output/'prompt.md').write_text(prompt)
    started = time.monotonic(); attempts = []; resume = bool(thread_id)
    current_prompt = prompt; final_message = ''; success = False; failure_reason = None
    for number in range(1, max_attempts+1):
        remaining = timeout_seconds-(time.monotonic()-started)
        if remaining <= 0: break
        folder = output/('attempt-%02d' % number); folder.mkdir()
        command = build_command(binary, model, session, resume, mcp, read(output_schema))
        write(folder/'command.json', command)
        (folder/'prompt.md').write_text(current_prompt)
        timed_out = False
        with (folder/'events.jsonl').open('w') as events, (folder/'stderr.log').open('w') as errors:
            process = subprocess.Popen(command, cwd=control, env=env, stdin=subprocess.PIPE,
                stdout=events, stderr=errors, text=True, start_new_session=True)
            try:
                process.communicate(current_prompt, timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
        summary = inspect_events(folder/'events.jsonl', session)
        result = summary['result'] or {}
        model_mismatch = (bool(set(summary['models']) - {model})
                          or (result.get('subtype') == 'success' and model not in summary['models']))
        if model_mismatch:
            failure_reason = 'Provider returned unexpected model identity: ' + (', '.join(summary['models']) or '<missing>')
        structured = result.get('structured_output')
        success = (process.returncode == 0 and not timed_out and summary['session_valid']
                   and result.get('subtype') == 'success' and not result.get('is_error')
                   and isinstance(structured, dict) and not summary['pending_tools'] and not model_mismatch)
        can_retry = not success and not model_mismatch and retryable(summary, (folder/'stderr.log').read_text(), timed_out)
        attempts.append({'attempt':number,'exit_code':process.returncode,'timed_out':timed_out,
            'session_valid':summary['session_valid'],'pending_tools':summary['pending_tools'],
            'tool_calls':summary['tool_calls'],'models':summary['models'],'retryable':can_retry,
            'model_identity_valid':not model_mismatch,
            'client_transport_errors':summary['client_transport_errors']})
        write(folder/'result.json', attempts[-1])
        if success:
            final_message = json.dumps(structured); break
        if not can_retry or number == max_attempts: break
        resume = summary['session_valid']
        current_prompt = ('Continue the interrupted round using the saved conversation and current workspace. '
            'Use completed tool results; do not repeat completed operations or checkout_parent. '
            'Finish the original task and return its required structured response.' if resume else prompt)
        delay = min(backoff_seconds * 2**(number-1), 60,
                    max(0, timeout_seconds-(time.monotonic()-started)))
        time.sleep(delay)
    response = {'model':model,'thread_id':session,'success':success,'final_message':final_message,
                'failure_reason':failure_reason,
                'duration_seconds':time.monotonic()-started,'client_attempts':attempts,
                'api_latency_seconds':None,'session_identity_valid':bool(attempts and attempts[-1]['session_valid'])}
    write(output/'result.json', response)
    if success: (output/'final.txt').write_text(final_message+'\n')
    return response
