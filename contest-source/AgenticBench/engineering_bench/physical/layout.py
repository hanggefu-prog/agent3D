"""Explicit host-owned deployment paths; no machine-specific defaults."""
import json
from pathlib import Path


def layout_path(root, key):
    root = Path(root).resolve(strict=True)
    path = root / 'engineering-layout.json'
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError('Duplicate deployment key: ' + k)
            result[k] = v
        return result
    data = json.loads(path.read_text(), object_pairs_hook=unique)
    keys = {'schema_version', 'release', 'platform', 'reference_root'}
    if (not isinstance(data, dict) or set(data) != keys or
            type(data['schema_version']) is not int or data['schema_version'] != 1):
        raise ValueError('Invalid engineering deployment layout')
    if key not in keys - {'schema_version'}:
        raise ValueError('Unknown deployment path: ' + key)
    for field in keys - {'schema_version'}:
        value = data[field]
        if (not isinstance(value, str) or not value.strip() or
                any(x in value for x in ('\x00', '\n', '\r', ','))):
            raise ValueError('Unsafe deployment path: ' + field)
    result = Path(data[key])
    if not result.is_absolute():
        result = root / result
    result = result.resolve(strict=True)
    if result == Path('/') or (key == 'release' and not result.is_file()) or (key != 'release' and not result.is_dir()):
        raise ValueError('Invalid deployment path type: ' + key)
    return result
