from __future__ import annotations

from .profile_cli import run_cli


STA_PROFILES = frozenset(
    {
        "placement_two_pass",
        "tp_gnn_placement_two_pass",
        "tp_gnn_ta3d_two_pass",
    }
)


def main(argv=None) -> int:
    return run_cli(
        argv,
        allowed_profiles=STA_PROFILES,
        prog="openagentic3dbench-sta",
        description=(
            "Run evaluator-owned two-pass STA promotion and validation profiles. "
            "These profiles are not optimization tasks."
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
