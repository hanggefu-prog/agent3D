# Opt-in OpenROAD compiler-cache reuse

This is a build optimization, not a new algorithm, evaluator, objective or
physical checkpoint-reuse policy. It currently connects to the maintained
**local routing-source candidate-image builder**. The general workspace
evaluator still rejects native C++/header changes until its own trusted
candidate-image integration is qualified; this option does not bypass that gate.

## Selection and ownership

Both `scripts/build_routing_candidate_image.py` and
`scripts/run_local_task_smoke.py` accept:

```text
--compiler-cache-dir /absolute/owned/batch/compiler-cache
--compiler-cache-gb 8
```

Without `--compiler-cache-dir`, the existing private cold-build path remains
selected. The smoke launcher requires the cache inside its batch root and
separate from experiment outputs, source, frozen inputs and references. The
standalone builder also excludes task and candidate archives. Existing,
nonempty directories without the cache ownership marker are rejected.

This is independent of `--retention full|best` (physical experiment archive)
and `--build-cache discard|retain` (successful private build copies). The
ccache size setting accepts integer values 1–64 GiB. It is ccache's approximate
bound **per compatibility namespace**, not a global disk ceiling. Changing
image/toolchain/configuration may create another namespace; no old namespaces,
Docker images, historical failures or other experiments are pruned automatically.

## Build and provenance contract

1. Verify the candidate scope and clean pinned checkout, clone into a new
   attempt, apply the trusted materialized patch and fingerprint the exact tree.
2. Probe the already-local pinned image for compiler, CMake, ccache, build tool,
   architecture and dependency identity. No image pull or package download.
3. Hash that identity with the trusted recipe/driver/configuration to select a
   cache namespace. Candidate source hashes identify outputs, **not** cache
   compatibility; unchanged translation units can share cache entries.
4. Compile with a private CMake/object tree and shared ccache in a bounded
   container: 16 CPUs, 48 GiB memory, 48 GiB memory-plus-swap, network disabled,
   no host credentials or Docker socket. The controller's slot stays held
   across compilation, packaging and evaluation. Build threads stay at eight
   in the smoke launcher. Compiler flags and the original shell recipe are
   unchanged.
5. Keep the verified input immutable. Native CMake uses a separate private
   writable `build-source`: OpenROAD generates its version header and message
   catalogs in-source. Compare complete before/after inventories, allow only
   the explicit audited generated-file list, retain those bytes/hashes/modes,
   and reject any other mutation. The generated-source policy participates in
   the cache identity; generated inputs are not silently excluded from provenance.
6. Stage and hash the executable without executing it in the compiler container.
   Check version/dependencies in a separate read-only container with no shared
   cache or writable source mount, then copy it into a new runtime image. Do not
   execute candidate code during image-layer construction. Re-read the image-owned
   receipt and independently check the installed binary hash, selected command
   path, version and dependencies in another read-only, unprivileged container.
   Evaluation uses only that image-installed runtime.

There is no shared mutable CMake object tree: fresh configure/dependency
traversal/link work can remain even on cache hits. Report this as **compiler-cache
reuse**, not a zero-work build. Raw before/after ccache counters are namespace
statistics; concurrent writers mean their delta is not a per-attempt hit count.

## Evidence and failure boundaries

`image-build-events.jsonl` separates source cloning, cached compilation, image
packaging and installed-runtime verification. `compiler-cache-attempt/` contains
probe/build commands and logs, cache identity, original/build-source inventories,
generated-source evidence (in the host-only `generated-source/` archive),
configure/compile-link events and raw cache statistics. Generated-file archiving
uses new-only, no-follow copies and rejects symlinked ancestors/destinations.
The sealed runtime receipt binds original source, generated build input, cache
compatibility key, exact base image and installed binary hash.

A failed or uncertain attempt is retained and never automatically retried.
After a verified receipt only, `discard` removes this attempt's `workspace`,
`context/openroad`, `compiler-cache-attempt/build` and `build-source`. It retains
recipes, patches, generated files, logs, binary/identity receipts and the shared
ccache. A retention error is a storage warning, not a rewritten evaluation score.

Offline tests cover identity, configuration, safety boundaries and packaging
handshakes. Native cold/warm builds, actual cache-hit/speed measurements and an
independent physical evaluation still require separate qualification. No speedup,
binary equivalence or physical success is established by mocked tests.

## Execution accounting

Local transport records `execution_attempts`, separating evaluator dispatch from
evidenced native-flow launches. `candidate_physical_attempts` counts candidates
with at least one launched flow command (not individual OpenROAD processes).
`*_known` gives the evidenced lower bound and `*_unknown` counts unresolved
attempts; the total is `null` while any attempt is unresolved. A build failure
before evaluator dispatch has zero native attempts but still consumes its
already-generated candidate/final slot. Reference qualification stays separate.

`Runtime.execute` retains queue/admission timestamps and splits admission,
image/command preparation, evaluator-process and total wall time. Native local
results additionally retain `artifacts.native_flow_execution` for the main stage
and optional CTS-thermal command, including failed/timeout/uncertain starts.
Workspace feedback separately records preparation, admission, native-flow
commands, runtime overlay, verifier, timeout cleanup and total duration. Dispatch
of a Docker client is not proof of native OpenROAD execution.

Workspace Agentic3D summaries separate role-turn attempts/completions/failures,
generated Task candidates, finals, measured improvements and role-accepted
improvements. Model/role duration is **agent-turn wall time**, including client,
network, terminal tools and waits. Pure API inference latency and underlying
API request count are unavailable without provider-level evidence and are not
fabricated from turn counts. Historical records are not backfilled with zeros.

## Legacy OpenEvolve Docker launcher

`scripts/openevolve_docker.py` now creates an experiment-owned
`OUTPUT/openroad-ccache` (8 GB approximate ccache limit, compiler identity checked
by content). The existing writable data-root mount exposes the same directory
to successive candidate containers. The local GPT launcher can use the same
three container environment values: `CCACHE_DIR`, `CCACHE_MAXSIZE=8G`, and
`CCACHE_COMPILERCHECK=content`. Do not put the cache under an individual
candidate's `result/runtime` or `result/workspace`, which compact retention removes.

This fixes compiler-cache persistence in that legacy launcher; it does not
replace its runtime admission contract with the qualified candidate-image path
described above. No shared CMake tree, skipped compilation, or physical result
reuse is introduced. Each experiment starts cold. Fresh configure, dependency
traversal, generated sources and final linking still run on warm candidates.
The shell recipe retains `openroad-ccache-before.log` and
`openroad-ccache-after.log` in each workspace evaluator result, including build
failures. For other build layouts these snapshots stay inside the build tree;
the qualified image driver continues retaining its own statistics in logs.
Counters are shared-cache totals, so compare snapshots and account for any
concurrent writers. A `Building CXX object` line alone does not establish a cache
miss. Existing runs do not acquire the new environment automatically.
