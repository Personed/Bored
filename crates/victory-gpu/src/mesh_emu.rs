//! Shared GPU memory for compute + vertex mesh emulation.
//!
//! An emulated mesh pipeline first runs the translated mesh shader as
//! compute, writing packed primitive indices and attributes into `output`.
//! Its generated vertex shader then reads that memory by `SV_VertexID` and
//! feeds the title's pixel shader. The buffer is shared by every emulated
//! mesh dispatch, with direct dispatches split into chunks that fit.

use std::sync::OnceLock;

use crate::host::{Host, host, name};
use windows::Win32::Graphics::Direct3D12::*;

const OUTPUT_FALLBACK_SIZES: &[u64] = &[128 << 20, 64 << 20, 32 << 20, 16 << 20];

pub(crate) struct Buffers {
    pub(crate) output: ID3D12Resource,
    pub(crate) output_gpu: u64,
    pub(crate) output_bytes: u64,
    pub(crate) dummy_srv: ID3D12Resource,
    pub(crate) dummy_srv_gpu: u64,
}

// SAFETY: D3D12 resources are free-threaded.
unsafe impl Send for Buffers {}
// SAFETY: as above.
unsafe impl Sync for Buffers {}

pub(crate) fn buffers() -> Option<&'static Buffers> {
    static BUFFERS: OnceLock<Option<Buffers>> = OnceLock::new();
    BUFFERS
        .get_or_init(|| match make() {
            Some(b) => Some(b),
            None => {
                tracing::warn!("no mesh emulation output buffer; emulated mesh dispatches dropped");
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

fn make() -> Option<Buffers> {
    let host = host()?;
    let (output, output_bytes) = allocate_output(host)?;
    let (dummy_srv, dummy_ptr) = host.upload_buffer(256)?;
    // SAFETY: `dummy_ptr` points to 256 writable bytes of the live upload buffer.
    unsafe { std::ptr::write_bytes(dummy_ptr, 0, 256) };
    name(&output, || format!("mesh emu output {output_bytes:#x}"));
    name(&dummy_srv, || "mesh emu dummy_srv".into());
    // SAFETY: both are live buffer resources.
    let (output_gpu, dummy_srv_gpu) = unsafe {
        (
            output.GetGPUVirtualAddress(),
            dummy_srv.GetGPUVirtualAddress(),
        )
    };
    tracing::info!(
        output_mb = output_bytes >> 20,
        "mesh shader emulation output buffer ready"
    );
    Some(Buffers {
        output,
        output_gpu,
        output_bytes,
        dummy_srv,
        dummy_srv_gpu,
    })
}

impl Buffers {
    /// Maximum number of mesh thread groups that fit in `output` in one chunk.
    pub(crate) fn group_cap(&self, group_bytes: u32) -> u32 {
        (self.output_bytes / u64::from(group_bytes.max(4))).min(65_535) as u32
    }
}
