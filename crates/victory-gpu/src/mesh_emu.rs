//! Shared GPU memory and preparation shaders for mesh emulation.
//!
//! An emulated mesh pipeline first runs the translated mesh shader as
//! compute, writing packed primitive indices and attributes into `output`.
//! Its generated vertex shader then reads that memory by `SV_VertexID` and
//! feeds the title's pixel shader. Indirect mesh dispatches first run a
//! tiny preparation pass that sums the bounded group counts, builds a slot
//! map across indirect records, and writes host dispatch/draw arguments.

use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{Host, HostList, host, name, transition};
use crate::indirect::uav_barrier;

const SLOT_MAP_BYTES: u64 = 65_536 * 4;
const INDIRECT_ARGS_BYTES: u64 = 256;
pub(crate) const MAX_INDIRECT_RECORDS: u32 = 65_535;
const OUTPUT_FALLBACK_SIZES: &[u64] = &[128 << 20, 64 << 20, 32 << 20, 16 << 20];

pub(crate) const PREP_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), UAV(u0), UAV(u1), UAV(u2), UAV(u3), UAV(u4)"
cbuffer Info : register(b0) {
    uint stride;
    uint max_records;
    uint max_primitives;
    uint group_cap;
    uint has_count;
    uint build_map;
};
// The source arguments and count use the UAV state also used by title
// compute producers; this pass does not store to them.
RWByteAddressBuffer args : register(u0);
RWByteAddressBuffer count_buf : register(u1);
RWByteAddressBuffer out_args : register(u2);
RWByteAddressBuffer slot_map : register(u3);
RWByteAddressBuffer budget : register(u4);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint n = has_count != 0 ? min(count_buf.Load(0), max_records) : max_records;
    uint cap = min(group_cap, 65535u);
    uint total = 0;
    for (uint r = 0; r < n; ++r) {
        if (total >= cap) {
            break;
        }
        uint3 g = args.Load3(r * stride);
        uint gx = (g.y != 0 && g.z != 0) ? g.x : 0;
        uint left = cap - total;
        uint take = min(gx, left);
        if (build_map != 0) {
            for (uint k = 0; k < take; ++k) {
                slot_map.Store((total + k) * 4, k);
            }
        }
        total += take;
    }
    bool over = false;
    uint cost = total * 128u;
    if (cost > 0) {
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < (1u << 27)) {
            uint next = ((1u << 27) - before <= cost) ? (1u << 27) : (before + cost);
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        over = before >= (1u << 27) || (1u << 27) - before < cost;
    }
    if (over) total = 0;
    out_args.Store4(0, uint4(total, 1, 1, 0));
    out_args.Store4(16, uint4(total * max_primitives * 3, 1, 0, 0));
}
"#;

pub(crate) struct Buffers {
    pub(crate) output: ID3D12Resource,
    pub(crate) output_gpu: u64,
    pub(crate) output_bytes: u64,
    pub(crate) slot_map: ID3D12Resource,
    pub(crate) slot_map_gpu: u64,
    pub(crate) indirect_args: ID3D12Resource,
    pub(crate) indirect_args_gpu: u64,
    pub(crate) dummy_srv: ID3D12Resource,
    pub(crate) dummy_srv_gpu: u64,
    prep_root: Option<ID3D12RootSignature>,
    prep_pso: Option<ID3D12PipelineState>,
}

// SAFETY: D3D12 resources, root signatures, and pipeline states are free-threaded.
unsafe impl Send for Buffers {}
// SAFETY: as above.
unsafe impl Sync for Buffers {}

pub(crate) fn buffers() -> Option<&'static Buffers> {
    static BUFFERS: OnceLock<Option<Buffers>> = OnceLock::new();
    BUFFERS
        .get_or_init(|| match make() {
            Some(b) => Some(b),
            None => {
                tracing::warn!("no mesh emulation buffers; emulated mesh dispatches dropped");
                None
            }
        })
        .as_ref()
}

fn configured_output_bytes() -> Option<u64> {
    let raw = std::env::var("VICTORY_MESH_BUFFER_MB").ok()?;
    let mb = raw.trim().parse::<u64>().ok()?;
    if (1..=1024).contains(&mb) {
        Some(mb << 20)
    } else {
        tracing::warn!(
            value = %raw,
            "VICTORY_MESH_BUFFER_MB out of range (1..=1024); using default"
        );
        None
    }
}

fn allocate_output(host: &Host) -> Option<(ID3D12Resource, u64)> {
    if let Some(bytes) = configured_output_bytes()
        && let Some(buf) = host.default_buffer(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
    {
        return Some((buf, bytes));
    }
    for &bytes in OUTPUT_FALLBACK_SIZES {
        if let Some(buf) = host.default_buffer(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
            return Some((buf, bytes));
        }
    }
    None
}

fn compile_prep(
    device: &ID3D12Device,
) -> windows::core::Result<(ID3D12RootSignature, ID3D12PipelineState)> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: static HLSL source and entry strings live through the call;
    // output pointers are ours.
    let compiled = unsafe {
        D3DCompile(
            PREP_SOURCE.as_ptr().cast(),
            PREP_SOURCE.len(),
            s!("mesh_indirect_prep"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    let fallback;
    let bytes: &[u8] = if compiled.is_ok() {
        let code = code.as_ref().ok_or_else(windows::core::Error::empty)?;
        // SAFETY: the blob owns its bytes.
        unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        }
    } else {
        if let Some(errors) = errors {
            // SAFETY: the blob owns its bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::info!(
                "mesh_indirect_prep: system compiler refused it ({}); trying DXC",
                String::from_utf8_lossy(text).trim()
            );
        }
        fallback = crate::indirect::dxc_compile(PREP_SOURCE).inspect_err(|e| {
            tracing::warn!(error = %e, "mesh_indirect_prep: DXC failed too");
        })?;
        &fallback
    };
    // SAFETY: `bytes` is valid compiled shader bytecode with an embedded
    // root signature, alive through pipeline creation.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok((root, pso))
    }
}

fn make() -> Option<Buffers> {
    let host = host()?;
    let (output, output_bytes) = allocate_output(host)?;
    let slot_map =
        host.default_buffer(SLOT_MAP_BYTES, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)?;
    let indirect_args = host.default_buffer(
        INDIRECT_ARGS_BYTES,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    )?;
    let (dummy_srv, dummy_ptr) = host.upload_buffer(256)?;
    // SAFETY: `dummy_ptr` points to 256 writable bytes of the live upload buffer.
    unsafe { std::ptr::write_bytes(dummy_ptr, 0, 256) };
    let prep = compile_prep(&host.device)
        .inspect_err(|e| tracing::warn!(error = %e, "mesh_indirect_prep pipeline failed"))
        .ok();
    let (prep_root, prep_pso) = (prep.as_ref().map(|p| p.0.clone()), prep.as_ref().map(|p| p.1.clone()));
    name(&output, || format!("mesh emu output {output_bytes:#x}"));
    name(&slot_map, || "mesh emu slot_map".into());
    name(&indirect_args, || "mesh emu indirect_args".into());
    name(&dummy_srv, || "mesh emu dummy_srv".into());
    // SAFETY: all are live buffer resources.
    let (output_gpu, slot_map_gpu, indirect_args_gpu, dummy_srv_gpu) = unsafe {
        (
            output.GetGPUVirtualAddress(),
            slot_map.GetGPUVirtualAddress(),
            indirect_args.GetGPUVirtualAddress(),
            dummy_srv.GetGPUVirtualAddress(),
        )
    };
    tracing::info!(
        output_mb = output_bytes >> 20,
        "mesh shader emulation buffers ready"
    );
    Some(Buffers {
        output,
        output_gpu,
        output_bytes,
        slot_map,
        slot_map_gpu,
        indirect_args,
        indirect_args_gpu,
        dummy_srv,
        dummy_srv_gpu,
        prep_root,
        prep_pso,
    })
}

impl Buffers {
    /// Maximum number of mesh thread groups that fit in `output` in one chunk.
    pub(crate) fn group_cap(&self, group_bytes: u32) -> u32 {
        (self.output_bytes / u64::from(group_bytes.max(4))).min(65_535) as u32
    }

    /// Compacts indirect mesh groups into host dispatch and draw arguments,
    /// charges only the groups that fit in the output-buffer chunk, and writes
    /// each compacted group's record-local id to `slot_map`.
    /// Leaves `indirect_args` in `INDIRECT_ARGUMENT` and `slot_map` readable.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_indirect(
        &self,
        list: &HostList,
        args_gpu: u64,
        count_gpu: Option<u64>,
        budget_gpu: u64,
        stride: u32,
        max_records: u32,
        max_primitives: u32,
        group_cap: u32,
    ) -> bool {
        let (Some(prep_root), Some(prep_pso)) = (&self.prep_root, &self.prep_pso) else {
            return false;
        };
        if max_records > MAX_INDIRECT_RECORDS
            || group_cap == 0
            || stride < 12
            || !stride.is_multiple_of(4)
            || u64::from(stride) * u64::from(max_records) > u64::from(u32::MAX)
            || !args_gpu.is_multiple_of(4)
            || count_gpu.is_some_and(|gpu| !gpu.is_multiple_of(4))
            || !budget_gpu.is_multiple_of(4)
        {
            return false;
        }
        let l = &list.list;
        let build_map = max_records > 1;
        let constants = [
            stride,
            max_records,
            max_primitives,
            group_cap,
            u32::from(count_gpu.is_some()),
            u32::from(build_map),
        ];
        // SAFETY: an open list; all GPU addresses name live buffers kept for
        // the submission.
        unsafe {
            // No SRV clamp precedes this pass: source buffers stay in UAV
            // state, and this barrier orders any title compute producer.
            l.ResourceBarrier(&[uav_barrier()]);
            l.SetComputeRootSignature(prep_root);
            l.SetPipelineState(prep_pso);
            l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            l.SetComputeRootUnorderedAccessView(1, args_gpu);
            l.SetComputeRootUnorderedAccessView(2, count_gpu.unwrap_or(args_gpu));
            l.SetComputeRootUnorderedAccessView(3, self.indirect_args_gpu);
            l.SetComputeRootUnorderedAccessView(4, self.slot_map_gpu);
            l.SetComputeRootUnorderedAccessView(5, budget_gpu);
            l.Dispatch(1, 1, 1);
            let args_barrier = transition(
                &self.indirect_args,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            );
            if build_map {
                let map_barrier = transition(
                    &self.slot_map,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                );
                l.ResourceBarrier(&[uav_barrier(), args_barrier, map_barrier]);
            } else {
                l.ResourceBarrier(&[uav_barrier(), args_barrier]);
            }
        }
        true
    }
}
