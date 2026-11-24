# D3D12 debugging notes

This document records diagnostic procedures and observations from the
experimental VIctory compatibility investigation.

## Diagnostic build

Build the workspace with GPU and core diagnostics enabled:

```powershell
cargo build --release --workspace --features victory-gpu/diagnostics,victory-core/diagnostics
```

## Runtime diagnostic capture

Set the diagnostic environment variables:

```powershell
$env:VICTORY_D3D12_DEBUG = "1"
$env:VICTORY_D3D12_DRED = "1"
$env:VICTORY_GPU_TIMING = "1"
$env:VICTORY_LOG = "victory_gpu::shadow=debug,info"
```

Run the title using locally supplied game files:

```powershell
.\target\release\victory.exe run "PATH_TO_TITLE_EXECUTABLE" --execute *> .\victory-debug.log
```

Replace `PATH_TO_TITLE_EXECUTABLE` with the local executable path.

These settings enable the D3D12 debug layer, DRED diagnostics, GPU timing
and shadow-memory logging. Diagnostic availability depends on the build
features and host graphics driver.

## Observed allocation failure

One captured run reported:

```text
E_OUTOFMEMORY (0x8007000E)
no shared buffer
size=94371840
```

The failed allocation was 94,371,840 bytes (90 MiB).

The shared-buffer allocation path uses a CPU-visible D3D12 custom heap
with WRITE_BACK CPU page properties and the L0 memory pool.

This is a system-memory allocation path. The error does not, by itself,
prove that dedicated GPU VRAM was exhausted.

## Resource-state validation

The D3D12 debug layer also reported resource-state mismatches involving:

- `UNORDERED_ACCESS`
- `VERTEX_AND_CONSTANT_BUFFER`

These warnings appeared shortly before an allocation failure in one
captured run. Their relationship to the allocation failure is unknown.

Resource transitions and command-list state tracking require further
investigation.

## Shadow-memory observations

The shadow-memory implementation maintains host-visible copies of title
memory.

Relevant investigation areas include:

- Registered title buffers and their shadow allocations.
- Loose shadow-memory allocations and cache lifetime.
- Allocation sizes and alignment.
- Resource references retained by command lists.
- Allocation reuse and release behaviour.

A previous log contained eight distinct registered buffers of at least
16 MiB, totalling approximately 446.92 MiB.

A separate run recorded 55 unique loose-memory allocation keys totalling
approximately 252 MiB. These figures came from different runs and must
not be combined into one memory total.

The loose-memory cache currently warrants investigation because its
entries are retained in a map. This is a hypothesis for possible
allocation growth, not a confirmed cause of the runtime failure.

## Interpreting results

Do not assume that `E_OUTOFMEMORY` means physical VRAM exhaustion.

Investigate the following independently:

1. System-memory availability at the time of failure.
2. D3D12 resource allocation sizes and heap types.
3. Resource lifetime and reference retention.
4. GPU resource-state validation messages.
5. Command-list submission and synchronization.
6. Reproducibility across repeated runs.

A diagnostic warning occurring before an allocation failure does not
establish causation.

## Reporting a reproducible issue

When reporting a failure, include:

- VIctory commit hash.
- CPU, GPU, RAM and driver version.
- Operating-system version.
- Diagnostic build features.
- Exact command and environment variables.
- First allocation failure and surrounding log messages.
- Resource-state warnings preceding the failure.
- Whether the failure reproduces on another run.

Review logs before sharing them. Remove machine-specific paths and never
include proprietary game assets, executables, shader dumps or extracted
game data in the repository.
