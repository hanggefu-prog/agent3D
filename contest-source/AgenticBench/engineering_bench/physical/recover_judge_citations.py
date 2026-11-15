"""Validate an already completed review with explicit mounted-path namespaces.

No model call, text rewrite, relaxed score/schema/line check or candidate edit.
This revalidates version-1 single reviews only; a version-2 panel stores each
sample's reply under review/sample-N and is validated with quoted citations.
"""
import argparse
from pathlib import Path
import shutil

from engineering_bench.judge import validate_review
from engineering_bench.protocol import read_json, sha, write_new, now
from engineering_bench.submission import verify_seal


def main():
    p = argparse.ArgumentParser()
    for k in ('judge_root', 'subject', 'sealed'):
        p.add_argument('--' + k.replace('_', '-'), type=Path, required=True)
    a = p.parse_args()
    verify_seal(a.subject, read_json(a.sealed))
    root = a.judge_root
    turn = read_json(root / 'review/turn/result.json')
    if not turn['success']:
        raise ValueError('Only a successfully completed judge reply can be revalidated')
    aliases = root / 'evidence-namespaces'
    shutil.copytree(root / 'results', aliases / 'results')
    shutil.copytree(root / 'control', aliases / 'task')
    raw = root / 'review/turn/final.txt'
    review = validate_review(raw.read_text(), [a.subject, root / 'control', root / 'results', aliases])
    write_new(root / 'review/review.json', review)
    write_new(root / 'citation-recovery.json', {'reason': 'Reviewer used results/ and task/ prefixes from actual container mounts',
        'original_reply_sha256': sha(raw), 'text_unchanged': True, 'model_calls': 0, 'completed_at': now()})
    write_new(root / 'result.json', {'status': review['status'], 'score': review['score'], 'completed_at': now()})
    print(review['status'], review['score'])


if __name__ == '__main__':
    main()
