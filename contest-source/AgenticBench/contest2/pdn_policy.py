"""Check the native contest PDN policy without replacing physical functions."""
from __future__ import annotations

import hashlib
from pathlib import Path

POLICY = "ideal_supply"
REVIEWED_PDN_SHA256 = frozenset({"c48597ace42c475c7a64229c2ee7863df8d75e5d72404a6d8363b2a85e25526b"})


def bind_config(config):
    pdn = config.setdefault("pdn", {})
    if not isinstance(pdn, dict) or pdn.get("algorithm", "openroad") != "openroad":
        raise ValueError("Contest timing requires the fixed OpenROAD PDN generation")
    pdn.update(algorithm="openroad", connectivity_policy=POLICY)
    return config


def install():
    """Compatibility entry: validate the policy already built into native PDN."""
    from open3dbench.backend import pdn
    if (getattr(pdn, "CONNECTIVITY_POLICY", None) != POLICY or
            hashlib.sha256(Path(pdn.__file__).read_bytes()).hexdigest() not in REVIEWED_PDN_SHA256):
        raise ValueError("PDN policy requires the reviewed native ideal-supply implementation")
