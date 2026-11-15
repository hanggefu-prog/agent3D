"""Trusted matrix and physical-flow entry with fixed contest policies."""
import sys


def main():
    if sys.argv[1:2] == ["--flow"]:
        del sys.argv[1]
        from .pdn_policy import install
        install()
        from open3dbench.cli import main as flow_main
        return flow_main()
    from .execution_policy import install
    install()
    from local_bench.matrix_docker import main as matrix_main
    return matrix_main()


if __name__ == "__main__":
    raise SystemExit(main())
