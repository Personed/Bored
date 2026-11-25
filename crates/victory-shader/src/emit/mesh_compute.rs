//! Shared buffer layout for the additive mesh-as-compute fallback.
//!
//! Each dispatched workgroup writes one fixed-size slot. The slot starts
//! with a 16-byte header (vertex count, primitive count, two reserved words),
//! followed by vertex records and then primitive records. Every vertex
//! record stores position at byte offset 0, optional clip/cull distances,
//! then one `float4` per parameter location. A primitive record stores
//! three vertex indices, a cull flag, then one `float4` per per-primitive
//! parameter location.

/// Byte offsets and strides for one workgroup's mesh-compute output slot.
///
/// Offsets within a vertex record are relative to the start of that record;
/// `vertices_offset` and `primitives_offset` are relative to the slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshComputeLayout {
    /// Bytes in one slot, including the header and all records.
    pub slot_bytes: u64,
    /// Byte offset of the first vertex record in the slot.
    pub vertices_offset: u64,
    /// Bytes in each vertex record.
    pub vertex_stride: u64,
    /// Byte offset of the first primitive record in the slot.
    pub primitives_offset: u64,
    /// Bytes in each primitive record, including its index/cull header.
    pub primitive_stride: u64,
    /// Declared maximum vertices for a workgroup.
    pub max_vertices: u32,
    /// Declared maximum primitives for a workgroup.
    pub max_primitives: u32,
    /// Number of clip-distance values stored per vertex.
    pub clip_count: u32,
    /// Offset of clip distances within each vertex record.
    pub clip_offset: u64,
    /// Number of cull-distance values stored per vertex.
    pub cull_count: u32,
    /// Offset of cull distances within each vertex record.
    pub cull_offset: u64,
    /// Parameter locations and their byte offsets within each vertex record,
    /// in ascending location order.
    pub params: Vec<(u32, u64)>,
    /// Per-primitive parameter locations and their byte offsets within each
    /// primitive record, in ascending location order.
    pub primitive_params: Vec<(u32, u64)>,
}

/// Checked resource and draw sizes for one mesh-compute dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshComputeDispatch {
    /// Number of mesh workgroups represented by this dispatch.
    pub groups: u32,
    /// Bytes needed for all workgroup output slots.
    pub output_bytes: u64,
    /// Vertex count for the fallback's single non-indexed triangle draw.
    pub draw_vertices: u32,
}

/// Computes the 16-byte-aligned slot layout used by the mesh-as-compute
/// fallback. Parameter locations are sorted and duplicate rows are stored
/// only once.
///
/// Returns `None` if the resulting offsets or slot size overflow `u64`.
pub fn mesh_compute_layout(
    max_vertices: u32,
    max_primitives: u32,
    clip_count: u32,
    cull_count: u32,
    parameter_locations: &[u32],
    primitive_parameter_locations: &[u32],
) -> Option<MeshComputeLayout> {
    const HEADER_BYTES: u64 = 16;
    const RECORD_BYTES: u64 = 16;

    let vec4s = |count: u32| u64::from(count).div_ceil(4);
    let clip_vec4s = vec4s(clip_count);
    let cull_vec4s = vec4s(cull_count);
    let mut params = parameter_locations.to_vec();
    params.sort_unstable();
    params.dedup();
    let mut primitive_params = primitive_parameter_locations.to_vec();
    primitive_params.sort_unstable();
    primitive_params.dedup();

    let vertex_stride = 1u64
        .checked_add(clip_vec4s)?
        .checked_add(cull_vec4s)?
        .checked_add(u64::try_from(params.len()).ok()?)?
        .checked_mul(RECORD_BYTES)?;
    let vertices_offset = HEADER_BYTES;
    let clip_offset = RECORD_BYTES;
    let cull_offset = RECORD_BYTES.checked_add(clip_vec4s.checked_mul(RECORD_BYTES)?)?;
    let mut params_with_offsets = Vec::with_capacity(params.len());
    for (index, location) in params.into_iter().enumerate() {
        let vec4_index = 1u64
            .checked_add(clip_vec4s)?
            .checked_add(cull_vec4s)?
            .checked_add(u64::try_from(index).ok()?)?;
        params_with_offsets.push((location, vec4_index.checked_mul(RECORD_BYTES)?));
    }
    let primitive_stride = 1u64
        .checked_add(u64::try_from(primitive_params.len()).ok()?)?
        .checked_mul(RECORD_BYTES)?;
    let primitive_params_with_offsets = primitive_params
        .into_iter()
        .enumerate()
        .map(|(index, location)| {
            Some((
                location,
                RECORD_BYTES.checked_add(u64::try_from(index).ok()?.checked_mul(RECORD_BYTES)?)?,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    let primitives_offset =
        vertices_offset.checked_add(u64::from(max_vertices).checked_mul(vertex_stride)?)?;
    let slot_bytes =
        primitives_offset.checked_add(u64::from(max_primitives).checked_mul(primitive_stride)?)?;

    Some(MeshComputeLayout {
        slot_bytes,
        vertices_offset,
        vertex_stride,
        primitives_offset,
        primitive_stride,
        max_vertices,
        max_primitives,
        clip_count,
        clip_offset,
        cull_count,
        cull_offset,
        params: params_with_offsets,
        primitive_params: primitive_params_with_offsets,
    })
}

/// Computes the output-buffer allocation and non-indexed draw count for
/// `groups` workgroups. Returns `None` if the allocation overflows `u64` or
/// the draw count cannot be represented by D3D12's `DrawInstanced` arguments.
pub fn mesh_compute_dispatch(
    layout: &MeshComputeLayout,
    groups: u32,
) -> Option<MeshComputeDispatch> {
    let output_bytes = layout.slot_bytes.checked_mul(u64::from(groups))?;
    let draw_vertices = u64::from(groups)
        .checked_mul(u64::from(layout.max_primitives))?
        .checked_mul(3)?;
    Some(MeshComputeDispatch {
        groups,
        output_bytes,
        draw_vertices: u32::try_from(draw_vertices).ok()?,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        MeshComputeDispatch, MeshComputeLayout, mesh_compute_dispatch, mesh_compute_layout,
    };

    #[test]
    fn aligns_records_and_sorts_unique_parameter_locations() {
        assert_eq!(
            mesh_compute_layout(2, 3, 5, 1, &[8, 2, 8], &[6, 4, 6]),
            Some(MeshComputeLayout {
                slot_bytes: 352,
                vertices_offset: 16,
                vertex_stride: 96,
                primitives_offset: 208,
                primitive_stride: 48,
                max_vertices: 2,
                max_primitives: 3,
                clip_count: 5,
                clip_offset: 16,
                cull_count: 1,
                cull_offset: 48,
                params: vec![(2, 64), (8, 80)],
                primitive_params: vec![(4, 16), (6, 32)],
            })
        );
    }

    #[test]
    fn supports_slots_without_optional_outputs() {
        let layout = mesh_compute_layout(3, 4, 0, 0, &[], &[]).unwrap();
        assert_eq!(layout.vertex_stride, 16);
        assert_eq!(layout.vertices_offset, 16);
        assert_eq!(layout.primitives_offset, 64);
        assert_eq!(layout.slot_bytes, 128);
        assert!(layout.params.is_empty());
        assert!(layout.primitive_params.is_empty());
    }

    #[test]
    fn rejects_layouts_whose_size_overflows() {
        assert!(mesh_compute_layout(u32::MAX, u32::MAX, u32::MAX, 0, &[], &[]).is_none());
    }

    #[test]
    fn sizes_the_output_buffer_and_raster_draw_for_all_groups() {
        let layout = mesh_compute_layout(2, 3, 5, 1, &[8, 2, 8], &[6, 4, 6]).unwrap();
        assert_eq!(
            mesh_compute_dispatch(&layout, 4),
            Some(MeshComputeDispatch {
                groups: 4,
                output_bytes: 1408,
                draw_vertices: 36,
            })
        );
    }

    #[test]
    fn zero_groups_need_no_buffer_or_draw() {
        let layout = mesh_compute_layout(2, 3, 0, 0, &[], &[]).unwrap();
        assert_eq!(
            mesh_compute_dispatch(&layout, 0),
            Some(MeshComputeDispatch {
                groups: 0,
                output_bytes: 0,
                draw_vertices: 0,
            })
        );
    }

    #[test]
    fn rejects_overflow_and_unrepresentable_draw_counts() {
        let mut layout = mesh_compute_layout(1, 1, 0, 0, &[], &[]).unwrap();
        layout.slot_bytes = u64::MAX;
        assert!(mesh_compute_dispatch(&layout, 2).is_none());

        let layout = mesh_compute_layout(1, u32::MAX, 0, 0, &[], &[]).unwrap();
        assert!(mesh_compute_dispatch(&layout, 1).is_none());
    }
}
