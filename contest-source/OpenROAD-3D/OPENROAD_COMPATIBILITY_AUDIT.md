# OpenROAD Official-Core Rebase and Flow Validation

## Result

The vendored OpenROAD tree has been rebased onto the normal upstream OpenROAD
commit `790a4a35df35f7f95a65335ca571cf8bf2981dd2` (2026-04-07). All other
Research-fork source changes were removed except the explicitly retained
Open3D placement, resizer, and routing extensions described below.

The rebased tree compiled successfully and completed two isolated `bp_fe`
flows from the placement checkpoint through placement optimization, detailed
placement, CTS, GRT, DRT, final extraction/STA/power/area reporting, and
HotSpot.

## Source provenance

| Component | Revision |
|---|---|
| OpenROAD upstream baseline | `790a4a35df35f7f95a65335ca571cf8bf2981dd2` |
| OpenSTA | `43177bba8f5f88dfb7dc35795242080a4fe2e986` |
| ABC | `17cadca080b33edc5a13c8e6e8e8092f8c350b25` |

The OpenSTA and ABC files in the vendored tree were compared file-by-file with
these upstream revisions and had no differences.

An itemized comparison against a pristine upstream `790a4a35` source archive
found exactly 26 intentionally modified source files:

- 9 GRT/FastRoute files for die-by-die global routing;
- 8 DRT/FlexDR files for die-specific layer enforcement and guide handling;
- 7 Resizer files for the retained 3D repair logic and dont-touch guards;
- `src/mpl/src/clusterEngine.cpp` for 3D macro-placement compatibility;
- `src/ifp/src/InitFloorplan.cc` for the associated 3D floorplan compatibility.

Research-only ChipletPar, GPU DPO, GKlib, METIS, and other source additions
are absent. CTS is the unmodified upstream `790a4a35` implementation.

## Selectable resizer behavior

`OPEN3D_RESIZER` selects the retained tier-aware restriction:

- `OPEN3D_RESIZER=0` (default): normal upstream repair decisions; retained
  dont-touch safety guards still apply when a design actually marks objects
  dont-touch.
- `OPEN3D_RESIZER=1`: additionally reject a newly buffered branch if its moved
  or remaining loads still span top and bottom tiers.

The flow Makefile exports the variable and preserves `OPEN3D_RESIZER=1` across
recursive stage invocations. Runtime logs explicitly print whether the mode is
enabled or disabled.

On the tested `bp_fe` placement input:

| Resize result | 3D mode | Upstream mode |
|---|---:|---:|
| Resized instances | 168 | 168 |
| Inserted repair buffers | 902 | 1,025 |
| Repaired nets | 466 | 466 |
| Mixed-tier candidates rejected | 1,533 | 0 |

These counts demonstrate that both modes execute and that the mode switch
changes only the intended tier-aware candidate filtering.

## Build validation

The source was rebuilt from a new build directory in
`openroad-dev-codex:latest` with 24 threads. The clean build reached
`[100%] Built target openroad` in 5 minutes 57 seconds. The resulting binary
also ran in the actual `openroad-dev-codex-flow:latest` flow image and exposed
the required `repair_design`, `set_net_routing_layers`, and `detailed_route`
commands.

The executable reports `GITDIR-NOTFOUND` because this is a vendored source
tree without nested Git metadata. The exact source and submodule revisions are
recorded in `openroad/OPEN3D_SOURCE.md` and `openroad/OPEN3D_BASE_COMMIT`.

## End-to-end validation

### 3D resizer plus HBT die-by-die routing

Variant: `bp_fe_official790a_3dresizer_die_by_die_20260721`

- `OPEN3D_RESIZER=1`
- `OPEN3D_DIE_BY_DIE=1`
- `OPEN3D_HBT=1`
- 1,149 HBT terminals inserted
- bottom GRT window: metal2-metal10; top GRT window: metal11-metal20
- bottom DRT window: metal1-metal10; top DRT window: metal11-metal20
- status: `COMPLETE`

| Metric | Result |
|---|---:|
| WNS | -1.3503 ns |
| TNS | -2,689.79 ns |
| Total power | 0.196 W |
| Instance area | 211,139 um^2 |
| Bottom/top GRT wirelength | 1,563,196 / 104,552 um |
| Bottom/top GRT overflow | 195 / 0 |
| Final DRT wirelength | 1,390,821 um |
| Final DRT vias | 282,115 |
| Final top-pass DRT violations | 10,308 |
| HotSpot maxima | 54.60 C / 53.04 C |
| OpenROAD stage time | 481 s |

Both die-specific DRT checkpoints and the merged final checkpoint were
produced. The official CTS created the two design clock trees `clk_i` and
`clk_i_regs`, confirming that this is upstream CTS behavior rather than a
custom CTS patch.

### Upstream resizer plus HBT die-by-die routing

Variant: `bp_fe_official790a_stockresizer_die_by_die_20260721`

- `OPEN3D_RESIZER=0`
- `OPEN3D_DIE_BY_DIE=1`
- `OPEN3D_HBT=1`
- 1,162 HBT terminals inserted
- status: `COMPLETE`

| Metric | Upstream resizer | 3D resizer | Upstream minus 3D |
|---|---:|---:|---:|
| Repair buffers | 1,025 | 902 | +123 |
| WNS | -1.1087 ns | -1.3503 ns | +0.2416 ns |
| TNS | -472.39 ns | -2,689.79 ns | +2,217.40 ns |
| Total power | 0.194562 W | 0.195534 W | -0.000972 W |
| Instance area | 211,368 um^2 | 211,139 um^2 | +229 um^2 |
| Bottom/top GRT nets | 31,892 / 1,162 | 31,769 / 1,149 | +123 / +13 |
| Bottom/top GRT wirelength | 1,558,203 / 103,862 um | 1,563,196 / 104,552 um | -4,993 / -690 um |
| Bottom/top GRT overflow | 238 / 0 | 195 / 0 | +43 / 0 |
| Final DRT wirelength | 1,385,197 um | 1,390,821 um | -5,624 um |
| Final DRT vias | 282,886 | 282,115 | +771 |
| Final top-pass DRT violations | 9,974 | 10,308 | -334 |
| HotSpot maxima | 54.69 C / 53.09 C | 54.60 C / 53.04 C | +0.09 C / +0.05 C |
| OpenROAD stage time | 468 s | 481 s | -13 s |

This is the controlled resizer comparison: both runs use the same placement
input, official `790a4a35` CTS, HBT split, and die-by-die routing sequence.
The upstream resizer inserts 123 additional repair buffers and yields much
better final timing, slightly lower power, and slightly shorter routed
wirelength, at the cost of 229 um^2 additional area and 43 more units of
bottom-die GRT overflow. Both runs complete final STA and HotSpot, but neither
is DRC-clean; the reported DRT violations remain a flow-quality limitation.

### Upstream resizer plus standard routing

Variant: `bp_fe_official790a_stockresizer_standard_20260721`

- `OPEN3D_RESIZER=0`
- `OPEN3D_DIE_BY_DIE=0`
- `OPEN3D_HBT=0`
- original `scripts/global_route.tcl` and `scripts/detail_route.tcl`
- status: `COMPLETE`

| Metric | Result |
|---|---:|
| WNS | -1.1081 ns |
| TNS | -470.79 ns |
| Total power | 0.194 W |
| Instance area | 211,368 um^2 |
| GRT wirelength | 1,511,107 um |
| GRT overflow | 0 |
| DRT wirelength | 1,285,258 um |
| DRT vias | 265,317 |
| DRT violations | 45,637 |
| HotSpot maxima | 54.67 C / 53.08 C |
| OpenROAD stage time | 353 s |

This second run verifies that the normal OpenROAD resizer path remains usable
through sign-off, not just that the environment variable is accepted.

The two tables are validation outcomes, not a controlled routing-only PPA
comparison: the first flow also enables HBT splitting, the Open3D technology
LEF, and die-by-die routing. A strict router comparison should freeze one
identical `4_cts.odb` and vary only the routing entrypoint.

## Main-tree replacement validation (2026-08-04)

The audited source was migrated into the current `main` working tree from
`origin/codex/open3d-die-routing` (`0692f8909cba05f30884a205813462a299c7395a`)
without checking out that branch's older repository-wide state. The current
Place-3D work and root documentation were preserved. Five vendored build assets
from `main` were retained: the ABC Makefile, PPL `Core.cpp`/`Core.h`, and the two
FLUTE lookup tables. Without the ABC Makefile, the source-copy Docker build
fails during CMake configuration at ABC's `cmake_info` target.

Validation results for image `open3dbench:2.0` (image ID
`sha256:e91e2cc3e983bcfbb77d075285ae667a7c824cc40c459fb2e7a4b298ea8bcf2a`):

- OpenROAD compiled and linked at 100% as `OpenROAD Open3DBench-v2.0`.
- `make check-image` passed.
- `make verify` passed all 75 Place-3D tests.
- `repair_design`, `set_global_routing_layer_adjustment`,
  `set_routing_layers`, `global_route`, `detailed_route`, and
  `clock_tree_synthesis` are present in the installed Tcl command surface.
- The `src/drt/test/top_level_term2.tcl` regression completed detailed routing
  and reported `No differences found`. Its stale conflicted branch golden was
  replaced by the deterministic official-baseline golden already on `main`.
- A fresh `bp_fe_top` standard-flow variant reached `6_final.odb`,
  `6_final.def`, final STA, power, and area reporting. It produced WNS
  `-1.1683 ns`, total power about `0.133 W`, area `211875 um^2`, and 10,986
  detailed-routing violations. The flow is not DRC-clean.

The final `make finish` exit code for that smoke run was nonzero only after the
OpenROAD backend completed, when GDS export requested KLayout. The image also
lacks Yosys; the validation mounted host Yosys 0.47 read-only for synthesis.
Pinned Yosys and KLayout packages must be added to the final image before this
counts as a self-contained RTL-to-GDS run. The repository's MoL evaluation pack
was absent, so the 3D/HBT matrix was not rerun during this replacement.

## Known flow warnings

The completed flows retain existing Open3DBench warnings, including the
missing `icache_id_i[0]` constraint port, unresolved SPEF instance names during
final STA, and high DRT violation counts. These warnings were also present in
the pre-rebase methodology and did not prevent final reports or HotSpot
outputs. They are benchmark/flow-quality issues rather than compilation or
API compatibility failures introduced by this rebase.
