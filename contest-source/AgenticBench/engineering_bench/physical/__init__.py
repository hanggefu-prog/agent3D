"""Protected native adapters, executed as scripts via engineering_bench native.

Sibling imports intentionally also work when the directory is mounted at
/adapter inside the isolated physical runtime. Do not import physical flow
modules into the host control plane.
"""
