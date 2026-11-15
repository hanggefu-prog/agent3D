# ASAP7 + ASAP7 3-D platform

Both independent die databases use the native ASAP7 M1-M9 stack. Signals route on M2-M9, HBT boundary terminals land on M9, and the unmodified upstream PDN uses M1/M2/M5/M6. Standard cells use RVT/TT NLDM views. Registered benchmark memories are generated with the upstream ASAP7 FakeRAM parameters; their derived physical views retain M1-M3 OBS and expose M4 pin access. HBT electrical timing remains the shared passive 3 ohm/0.6 fF model. See provenance.json for the pinned ORFS commit.
