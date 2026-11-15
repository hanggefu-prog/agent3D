"""Host-only recovery of retained model reservations and trusted usage receipts."""
import argparse
import json
from pathlib import Path
import sqlite3

from .budget import BudgetLedger
from .runtime_recovery import recover_provider, settle_receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, help='Trusted provider usage receipt, never a participant cost claim')
    args = parser.parse_args(argv)
    path = args.ledger.resolve(strict=True)
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        settings = dict(db.execute('SELECT key,value FROM settings'))
    ledger = BudgetLedger(path, evaluation_limit=int(settings['evaluation_limit']),
        model_cost_limit_rmb=settings['model_cost_limit_rmb'], task_id=settings['task_id'],
        allowed_model=settings['allowed_model'] or None, evaluation_mode=settings['evaluation_mode'])
    recover_provider(ledger)
    if args.receipt:
        value = json.loads(args.receipt.read_text())
        request_id = value['request_id']
        row = next((r for r in ledger.snapshot()['model_calls'] if r['request_id'] == request_id), None)
        if not row or row['status'] == 'reserved':
            raise ValueError('Receipt requires an ended, retained model request')
        settle_receipt(ledger, request_id, value)
    state = ledger.snapshot()
    print(json.dumps({key: state[key] for key in ('model_cost_spent_rmb',
        'model_cost_reserved_rmb', 'model_cost_available_rmb', 'unresolved_model_calls')}, indent=2))

if __name__ == '__main__':
    main()
