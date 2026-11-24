# Compatibility investigation: Grand Theft Auto VI

**Status:** Experimental; not confirmed playable
**Last updated:** 2026-11-24

This document records an independent compatibility investigation using
VIctory. It is a development report, not an official compatibility claim.

## Test environment

| Component | Configuration |
|---|---|
| CPU | Intel Core i7-8700K |
| GPU | AMD Radeon RX 5600 XT, 6 GB |
| RAM | 16 GB DDR4 |
| Operating system | Windows |
| Graphics API | Direct3D 12 |
| GPU architecture | AMD RDNA 1 (Navi 10) |

## Test target

- Title executable: `game_scarlett_master.exe`
- Game files are supplied locally by the tester.
- No proprietary game assets or executable contents are included in this repository.

## Shader translation results

A translation scan reported:

| Metric | Result |
|---|---:|
| Containers scanned | 26,599 |
| Programs translated | 26,148 / 26,148 |
| Vertex/pixel DXIL emitted | 12,250 / 14,039 |
| Compute DXIL emitted | 1,441 / 2,582 |
| Mesh DXIL emitted | 8,278 / 8,713 |
| DXIL validated | 21,783 / 21,969 |

Approximately 99.15% of emitted DXIL programs passed validation.

These figures describe translation and validation only. They do not
demonstrate that the title can render or run correctly.

Pipeline investigation also encountered failures involving mesh pipelines.
Host mesh-shader support and pipeline creation remain areas for further
investigation.

## Runtime observations

The title progresses into GPU initialization and resource activity but
does not reach confirmed playable gameplay.

Observed issues include:

- Direct3D 12 `E_OUTOFMEMORY` errors (`0x8007000E`).
- Failed allocation of a 94,371,840-byte shared buffer (90 MiB).
- Command-list submission or closing failure following allocation errors.
- D3D12 resource-state validation warnings involving
  `UNORDERED_ACCESS` and `VERTEX_AND_CONSTANT_BUFFER`.

The shared buffer uses a CPU-visible, system-memory allocation path.
Therefore, the allocation failure alone does not establish that dedicated
GPU VRAM was exhausted.

The resource-state warnings occurred shortly before an allocation failure
in one captured run. Their causal relationship has not been established.

## Current assessment

**Confirmed:**
- The workspace builds successfully in release mode with GPU and core
  diagnostics enabled.
- Shader translation completes for the reported program counts.
- Most emitted DXIL passes validation.
- Runtime resource allocation failures occur.

**Not yet established:**
- The root cause of the shared-buffer allocation failure.
- Whether resource-state mismatches contribute to the failure.
- Whether the observed allocation behaviour is a memory leak,
  fragmentation, budget exhaustion, or another issue.
- Whether the title can reach stable gameplay on this configuration.

## Next investigation areas

1. Measure shared-buffer allocations and lifetimes.
2. Investigate the lifetime and growth of loose shadow-memory allocations.
3. Trace D3D12 resource-state transitions around the first failure.
4. Determine whether allocation failures are reproducible.
5. Re-test after any targeted fixes.

## Reporting and reproducibility

Results depend on the exact VIctory commit, game build, graphics driver,
configuration and test conditions.

Diagnostic logs and translation outputs are retained separately from
this repository. They may contain machine-specific information and
should be reviewed before sharing.
