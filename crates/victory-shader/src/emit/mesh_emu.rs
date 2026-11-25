//! Mesh shader emulation (`cs_6_6` + generated unpack `vs_6_6`) for SM 6.6
//! GPUs that lack hardware mesh shaders (`D3D12_MESH_SHADER_TIER_NOT_SUPPORTED`).
//!
//! An emulated mesh program (`MeshMode::Emulate`) compiles as a compute shader
//! (`cs_6_6`) with the console's `[numthreads(threads, 1, 1)]` and LDS, writing
//! packed primitive indices, per-vertex attributes (`OSG1`), and per-primitive
//! attributes (`PSG1`) into a fixed-stride per-group slice of a raw UAV buffer
//! at `u0, space 2` ([`MESH_EMU_SPACE`]):
//!
//! ```text
//! group g (byte offset = g * group_bytes):
//!   [0 .. index_bytes)                                  max_primitives x u32 (v0 | v1<<10 | v2<<20 | cull<<31)
//!   [index_bytes .. prim_offset_bytes)                  max_vertices x vertex_dwords x u32
//!   [prim_offset_bytes .. group_bytes)                  max_primitives x prim_dwords x u32
//! ```
//!
//! [`unpack_vertex_shader`] builds the matching `vs_6_6` that reads that raw
//! buffer at `t0, space 2` by `SV_VertexID`, culls bit-31 primitives via quiet
//! `NaN` on `SV_Position`, and emits every vertex and per-primitive attribute in
//! the exact signature layout the title's pixel shader (`ps_6_6`) expects.

use std::collections::BTreeMap;

use super::bitcode::{Attr, Bin, Cast, Term, pred};
use super::container::SigElement;
use super::shader::Shader;
use crate::ShaderError;
use crate::dxbc::{Container, FourCC, u32_at};

/// Sentinel `base_group` value in `b0, space 2` telling an emulated mesh
/// compute shader that reads `SV_GroupID` to look up its per-record group ID
/// in the `slot_map` raw SRV (`t0, space 2`).
pub const INDIRECT_SLOT_MAP_FLAG: u32 = u32::MAX;

/// Whether a console mesh shader (`HwStage::Ms`) translates to native `ms_6_6`
/// or to a compute + vertex emulation pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshMode {
    #[default]
    Native,
    Emulate,
}

/// One vertex or per-primitive attribute stored in the emulated mesh buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshOutput {
    pub name: String,
    pub index: u32,
    pub kind: u8,
    pub comp: u8,
    pub interp: u8,
    pub row: Option<u32>,
    pub col: u8,
    pub cols: u8,
    /// Dword offset of channel 0 from the start of a vertex or primitive record.
    pub dword_offset: u32,
}

/// Memory layout and metadata emitted by an emulated mesh compute shader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshEmu {
    pub threads: u32,
    pub max_vertices: u32,
    pub max_primitives: u32,
    pub vertex_dwords: u32,
    pub prim_dwords: u32,
    pub reads_group_id: bool,
    pub vertex_outputs: Vec<MeshOutput>,
    pub prim_outputs: Vec<MeshOutput>,
}

impl MeshEmu {
    /// Bytes reserved per group for packed primitive index words.
    pub fn index_bytes(&self) -> u32 {
        self.max_primitives * 4
    }

    /// Byte offset of vertex attribute records within one group slice.
    pub fn vertex_offset_bytes(&self) -> u32 {
        self.index_bytes()
    }

    /// Stride in bytes of one vertex record.
    pub fn vertex_stride_bytes(&self) -> u32 {
        self.vertex_dwords * 4
    }

    /// Byte offset of per-primitive attribute records within one group slice.
    pub fn prim_offset_bytes(&self) -> u32 {
        self.index_bytes() + self.max_vertices * self.vertex_stride_bytes()
    }

    /// Stride in bytes of one per-primitive record.
    pub fn prim_stride_bytes(&self) -> u32 {
        self.prim_dwords * 4
    }

    /// Total bytes occupied by one threadgroup in the intermediate buffer.
    pub fn group_bytes(&self) -> u32 {
        self.prim_offset_bytes() + self.max_primitives * self.prim_stride_bytes()
    }
}

#[derive(Clone, Copy)]
enum OutputSource {
    Vertex { dword_offset: u32, avail_cols: u8 },
    Prim { dword_offset: u32, avail_cols: u8 },
    Zero,
}

struct UnpackOutput {
    sig: SigElement,
    src: OutputSource,
}

/// Generates a `vs_6_6` DXIL container that unpacks `emu`'s raw buffer (`t0,
/// space 2`) by `SV_VertexID` into the input signature expected by `ps_dxil`
/// (or rasterizer system values only for depth-only pipelines).
pub fn unpack_vertex_shader(emu: &MeshEmu, ps_dxil: Option<&[u8]>) -> Result<Vec<u8>, ShaderError> {
    if emu.max_vertices == 0 || emu.max_primitives == 0 {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "an empty mesh shader output",
        });
    }

    let plan = build_output_plan(emu, ps_dxil)?;

    let mut s = Shader::new(false);
    s.inputs = vec![SigElement {
        name: "SV_VertexID".into(),
        index: 0,
        kind: 1,
        comp: 5,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 1,
        usage: 1,
        half: false,
    }];
    s.outputs = plan.iter().map(|o| o.sig.clone()).collect();

    let buf = s.emu_srv_handle();
    let e = s.entry;
    let i32t = s.t.i32;
    let undef_i32 = s.m.undef(i32t);
    let vid = {
        let zero = s.m.i32(0);
        let z8 = s.m.i8(0);
        s.op(
            e,
            "loadInput.i32",
            Attr::ReadNone,
            i32t,
            4,
            &[zero, zero, z8, undef_i32],
        )
    };

    let k3 = s.m.i32(3);
    let tri = s.m.bin(e, Bin::UDiv, vid, k3);
    let corner = s.m.bin(e, Bin::URem, vid, k3);
    let (group_idx, prim_idx) = if emu.max_primitives == 1 {
        (tri, s.m.i32(0))
    } else {
        let mp = s.m.i32(emu.max_primitives);
        (
            s.m.bin(e, Bin::UDiv, tri, mp),
            s.m.bin(e, Bin::URem, tri, mp),
        )
    };

    let group_stride = s.m.i32(emu.group_bytes());
    let group_base = s.m.bin(e, Bin::Mul, group_idx, group_stride);
    let k2 = s.m.i32(2);
    let prim_byte = s.m.bin(e, Bin::Shl, prim_idx, k2);
    let idx_off = s.m.bin(e, Bin::Add, group_base, prim_byte);

    let idx_load = {
        let mask1 = s.m.i8(1);
        let align4 = s.m.i32(4);
        s.op(
            e,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            s.t.res_ret,
            139,
            &[buf, idx_off, undef_i32, mask1, align4],
        )
    };
    let idx_f = s.m.extract(e, idx_load, 0);
    let packed = s.m.cast(e, Cast::Bitcast, idx_f, i32t);
    let zero_i = s.m.i32(0);
    let culled = s.m.cmp(e, pred::SLT, packed, zero_i);

    let cull_bb = s.m.new_block();
    let draw_bb = s.m.new_block();
    s.m.terminate(e, Term::CondBr(culled, cull_bb, draw_bb));

    // Culled primitive: write quiet NaN to SV_Position so the rasterizer drops
    // the triangle regardless of depth-clip settings, and zero other outputs.
    for (id, out) in plan.iter().enumerate() {
        for c in 0..out.sig.cols {
            let (suffix, val) = if out.sig.kind == 3 {
                (".f32", s.m.f32(0x7FC0_0000))
            } else if out.sig.comp == 9 {
                (".f32", s.m.f32(0))
            } else {
                (".i32", zero_i)
            };
            let id_k = s.m.i32(id as u32);
            let col_k = s.m.i8(c);
            s.op(
                cull_bb,
                &format!("storeOutput{suffix}"),
                Attr::NoUnwind,
                s.t.void,
                5,
                &[id_k, zero_i, col_k, val],
            );
        }
    }
    s.m.terminate(cull_bb, Term::Ret);

    // Active primitive: load vertex and/or per-primitive attributes and store them.
    let needs_vert = plan
        .iter()
        .any(|o| matches!(o.src, OutputSource::Vertex { .. }));
    let needs_prim = plan
        .iter()
        .any(|o| matches!(o.src, OutputSource::Prim { .. }));

    let v_base = if needs_vert {
        let k10 = s.m.i32(10);
        let shift = s.m.bin(draw_bb, Bin::Mul, corner, k10);
        let shifted = s.m.bin(draw_bb, Bin::LShr, packed, shift);
        let mask = s.m.i32(0x3FF);
        let v_raw = s.m.bin(draw_bb, Bin::And, shifted, mask);
        let mv = s.m.i32(emu.max_vertices);
        let in_bounds = s.m.cmp(draw_bb, pred::IULT, v_raw, mv);
        let last_v = s.m.i32(emu.max_vertices - 1);
        let v_idx = s.m.select(draw_bb, in_bounds, v_raw, last_v);
        let v_stride = s.m.i32(emu.vertex_stride_bytes());
        let v_rel = s.m.bin(draw_bb, Bin::Mul, v_idx, v_stride);
        let v_off_base = s.m.i32(emu.vertex_offset_bytes());
        let v_off = s.m.bin(draw_bb, Bin::Add, v_off_base, v_rel);
        Some(s.m.bin(draw_bb, Bin::Add, group_base, v_off))
    } else {
        None
    };

    let p_base = if needs_prim {
        let p_stride = s.m.i32(emu.prim_stride_bytes());
        let p_rel = s.m.bin(draw_bb, Bin::Mul, prim_idx, p_stride);
        let p_off_base = s.m.i32(emu.prim_offset_bytes());
        let p_off = s.m.bin(draw_bb, Bin::Add, p_off_base, p_rel);
        Some(s.m.bin(draw_bb, Bin::Add, group_base, p_off))
    } else {
        None
    };

    let zero_f = s.m.f32(0);
    let align4 = s.m.i32(4);
    for (id, out) in plan.iter().enumerate() {
        let (base, dword_offset, avail_cols) = match out.src {
            OutputSource::Vertex {
                dword_offset,
                avail_cols,
            } => (v_base.expect("vertex base"), dword_offset, avail_cols),
            OutputSource::Prim {
                dword_offset,
                avail_cols,
            } => (p_base.expect("prim base"), dword_offset, avail_cols),
            OutputSource::Zero => {
                for c in 0..out.sig.cols {
                    let (suffix, val) = if out.sig.comp == 9 {
                        (".f32", zero_f)
                    } else {
                        (".i32", zero_i)
                    };
                    let id_k = s.m.i32(id as u32);
                    let col_k = s.m.i8(c);
                    s.op(
                        draw_bb,
                        &format!("storeOutput{suffix}"),
                        Attr::NoUnwind,
                        s.t.void,
                        5,
                        &[id_k, zero_i, col_k, val],
                    );
                }
                continue;
            }
        };

        let load_cols = out.sig.cols.min(avail_cols).min(4);
        let loaded = if load_cols > 0 {
            let addr = if dword_offset == 0 {
                base
            } else {
                let disp = s.m.i32(dword_offset * 4);
                s.m.bin(draw_bb, Bin::Add, base, disp)
            };
            let mask = s.m.i8((1u8 << load_cols) - 1);
            Some(s.op(
                draw_bb,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                s.t.res_ret,
                139,
                &[buf, addr, undef_i32, mask, align4],
            ))
        } else {
            None
        };

        for c in 0..out.sig.cols {
            let (suffix, val) = match loaded {
                Some(agg) if c < load_cols => {
                    let f = s.m.extract(draw_bb, agg, u32::from(c));
                    if out.sig.comp == 9 {
                        (".f32", f)
                    } else {
                        (".i32", s.m.cast(draw_bb, Cast::Bitcast, f, i32t))
                    }
                }
                _ if out.sig.comp == 9 => (".f32", zero_f),
                _ => (".i32", zero_i),
            };
            let id_k = s.m.i32(id as u32);
            let col_k = s.m.i8(c);
            s.op(
                draw_bb,
                &format!("storeOutput{suffix}"),
                Attr::NoUnwind,
                s.t.void,
                5,
                &[id_k, zero_i, col_k, val],
            );
        }
    }
    s.m.terminate(draw_bb, Term::Ret);
    Ok(s.finish())
}

fn build_output_plan(
    emu: &MeshEmu,
    ps_dxil: Option<&[u8]>,
) -> Result<Vec<UnpackOutput>, ShaderError> {
    let mut plan = Vec::new();
    let mut occupied: BTreeMap<u32, u8> = BTreeMap::new();
    let mut next_row: u32 = 0;

    fn occupy(occupied: &mut BTreeMap<u32, u8>, next_row: &mut u32, row: u32, col: u8, cols: u8) {
        let mask = ((1u8 << cols.min(4)) - 1) << col.min(3);
        *occupied.entry(row).or_default() |= mask;
        *next_row = (*next_row).max(row + 1);
    }

    let mut emitted_sys: Vec<(u8, u32)> = Vec::new();

    if let Some(ps_bytes) = ps_dxil {
        let c = Container::parse(ps_bytes)?;
        let ps_inputs = match c.part(FourCC::ISG1) {
            Some(part) => super::signature(part.data)?,
            None => Vec::new(),
        };
        let ps_interps = c
            .part(FourCC::PSV0)
            .map(|p| psv_input_interps(p.data))
            .unwrap_or_default();

        for pi in &ps_inputs {
            if !matches!(pi.kind, 0 | 3 | 4 | 5 | 6 | 7) {
                continue;
            }
            let (found, from_prim) = match find_mesh_output(&emu.vertex_outputs, pi) {
                Some(o) => (Some(o), false),
                None => match find_mesh_output(&emu.prim_outputs, pi) {
                    Some(o) => (Some(o), true),
                    None => (None, false),
                },
            };
            let row = pi.row.unwrap_or(next_row);
            let (col, cols) = if pi.kind == 3 {
                (0, 4)
            } else {
                (pi.col, pi.cols)
            };
            let src = match found {
                Some(o) if pi.kind == 3 => OutputSource::Vertex {
                    dword_offset: o.dword_offset,
                    avail_cols: o.cols,
                },
                Some(o) if col >= o.col && (col - o.col) < o.cols => {
                    let skip = u32::from(col - o.col);
                    let avail_cols = o.cols - skip as u8;
                    let dword_offset = o.dword_offset + skip;
                    if from_prim {
                        OutputSource::Prim {
                            dword_offset,
                            avail_cols,
                        }
                    } else {
                        OutputSource::Vertex {
                            dword_offset,
                            avail_cols,
                        }
                    }
                }
                _ => OutputSource::Zero,
            };

            let ps_interp = ps_interps
                .iter()
                .find(|&&(r, c, k, _)| r == row && c == pi.col && k == pi.kind)
                .map(|&(_, _, _, m)| m)
                .unwrap_or(0);
            let interp = match pi.kind {
                3 => 4,
                4 | 5 => 1,
                6 | 7 => 2,
                _ => match ps_interp {
                    1 => 1,
                    2 | 3 | 6 => 2,
                    4 | 5 | 7 => 4,
                    _ if from_prim => 1,
                    _ if pi.comp == 9 => 2,
                    _ => 1,
                },
            };

            if pi.kind != 0 {
                emitted_sys.push((pi.kind, pi.index));
            }
            occupy(&mut occupied, &mut next_row, row, col, cols);
            plan.push(UnpackOutput {
                sig: SigElement {
                    name: pi.name.clone(),
                    index: pi.index,
                    kind: pi.kind,
                    comp: pi.comp,
                    interp,
                    row: Some(row),
                    col,
                    cols,
                    usage: (1u8 << cols) - 1,
                    half: false,
                },
                src,
            });
        }
    }

    for (outputs, from_prim) in [(&emu.vertex_outputs, false), (&emu.prim_outputs, true)] {
        for o in outputs {
            if !matches!(o.kind, 3..=7) {
                continue;
            }
            if emitted_sys.contains(&(o.kind, o.index)) {
                continue;
            }
            emitted_sys.push((o.kind, o.index));
            let mask = ((1u8 << o.cols.min(4)) - 1) << o.col.min(3);
            let (row, col) = match o.row {
                Some(r) if (occupied.get(&r).copied().unwrap_or(0) & mask) == 0 => (r, o.col),
                _ => (next_row, 0),
            };
            occupy(&mut occupied, &mut next_row, row, col, o.cols);
            let interp = match o.kind {
                3 => 4,
                4 | 5 => 1,
                _ if from_prim => 1,
                _ => 2,
            };
            let src = if from_prim {
                OutputSource::Prim {
                    dword_offset: o.dword_offset,
                    avail_cols: o.cols,
                }
            } else {
                OutputSource::Vertex {
                    dword_offset: o.dword_offset,
                    avail_cols: o.cols,
                }
            };
            plan.push(UnpackOutput {
                sig: SigElement {
                    name: o.name.clone(),
                    index: o.index,
                    kind: o.kind,
                    comp: o.comp,
                    interp,
                    row: Some(row),
                    col,
                    cols: o.cols,
                    usage: (1u8 << o.cols) - 1,
                    half: false,
                },
                src,
            });
        }
    }

    if !plan.iter().any(|o| o.sig.kind == 3) {
        let row = next_row;
        plan.push(UnpackOutput {
            sig: SigElement {
                name: "SV_Position".into(),
                index: 0,
                kind: 3,
                comp: 9,
                interp: 4,
                row: Some(row),
                col: 0,
                cols: 4,
                usage: 0xF,
                half: false,
            },
            src: OutputSource::Zero,
        });
    }

    Ok(plan)
}

fn find_mesh_output<'a>(outputs: &'a [MeshOutput], pi: &SigElement) -> Option<&'a MeshOutput> {
    let matches = |o: &&MeshOutput| {
        if pi.kind != 0 {
            o.kind == pi.kind && o.index == pi.index
        } else {
            o.kind == 0 && o.index == pi.index && o.name.eq_ignore_ascii_case(&pi.name)
        }
    };
    outputs
        .iter()
        .find(|o| matches(o) && o.col == pi.col)
        .or_else(|| outputs.iter().find(matches))
}

/// Reads `(row, col, kind, interp)` tuples for `PSV0` input signature elements.
fn psv_input_interps(psv: &[u8]) -> Vec<(u32, u8, u8, u8)> {
    let Ok(info_size) = u32_at(psv, 0).map(|n| n as usize) else {
        return Vec::new();
    };
    if info_size < 32 || psv.len() < 4 + info_size {
        return Vec::new();
    }
    let input_count = usize::from(psv[4 + 28]);
    let mut at = 4 + info_size;
    let Ok(binding_count) = u32_at(psv, at).map(|n| n as usize) else {
        return Vec::new();
    };
    at += 4;
    if binding_count > 0 {
        let Ok(entry_size) = u32_at(psv, at).map(|n| n as usize) else {
            return Vec::new();
        };
        at += 4;
        at = match binding_count
            .checked_mul(entry_size)
            .and_then(|n| at.checked_add(n))
        {
            Some(next) if next <= psv.len() => next,
            _ => return Vec::new(),
        };
    }
    let Ok(string_bytes) = u32_at(psv, at).map(|n| n as usize) else {
        return Vec::new();
    };
    at += 4;
    if at + string_bytes > psv.len() {
        return Vec::new();
    }
    at += string_bytes;
    let Ok(indices_count) = u32_at(psv, at).map(|n| n as usize) else {
        return Vec::new();
    };
    at += 4;
    if at + indices_count * 4 > psv.len() {
        return Vec::new();
    }
    at += indices_count * 4;
    if at + 4 > psv.len() {
        return Vec::new();
    }
    let Ok(elem_size) = u32_at(psv, at).map(|n| n as usize) else {
        return Vec::new();
    };
    at += 4;
    if elem_size < 14 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for k in 0..input_count {
        let Some(e) = psv.get(at + k * elem_size..at + (k + 1) * elem_size) else {
            break;
        };
        let alloc = e[10];
        if alloc & 0x40 == 0 {
            continue;
        }
        let row = u32::from(e[9]);
        let col = (alloc >> 4) & 3;
        let kind = e[11];
        let interp = e[13];
        out.push((row, col, kind, interp));
    }
    out
}

#[cfg(all(test, windows))]
mod tests {
    use super::super::validator::Validator;
    use super::*;

    fn fake_ps_container(inputs: &[SigElement]) -> Vec<u8> {
        let mut s = Shader::new(true);
        s.inputs = inputs.to_vec();
        s.outputs = vec![SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xF,
            half: false,
        }];
        let b = s.entry;
        let zero = s.m.f32(0);
        for c in 0..4u8 {
            let zero_i = s.m.i32(0);
            let col_k = s.m.i8(c);
            s.op(
                b,
                "storeOutput.f32",
                Attr::NoUnwind,
                s.t.void,
                5,
                &[zero_i, zero_i, col_k, zero],
            );
        }
        s.m.terminate(b, Term::Ret);
        s.finish()
    }

    #[test]
    fn unpack_vs_signature_matches_ps_with_per_primitive_outputs() {
        let emu = MeshEmu {
            threads: 32,
            max_vertices: 16,
            max_primitives: 8,
            vertex_dwords: 8,
            prim_dwords: 4,
            reads_group_id: false,
            vertex_outputs: vec![
                MeshOutput {
                    name: "SV_Position".into(),
                    index: 0,
                    kind: 3,
                    comp: 9,
                    interp: 4,
                    row: Some(0),
                    col: 0,
                    cols: 4,
                    dword_offset: 0,
                },
                MeshOutput {
                    name: "COLOR".into(),
                    index: 0,
                    kind: 0,
                    comp: 9,
                    interp: 2,
                    row: Some(1),
                    col: 0,
                    cols: 4,
                    dword_offset: 4,
                },
            ],
            prim_outputs: vec![MeshOutput {
                name: "PRIMDATA".into(),
                index: 0,
                kind: 0,
                comp: 9,
                interp: 1,
                row: Some(0),
                col: 0,
                cols: 4,
                dword_offset: 0,
            }],
        };

        // PS places COLOR0 at row 0 (colliding with MS SV_Position row 0!)
        // and PRIMDATA at row 1, without reading SV_Position.
        let ps = fake_ps_container(&[
            SigElement {
                name: "COLOR".into(),
                index: 0,
                kind: 0,
                comp: 9,
                interp: 2,
                row: Some(0),
                col: 0,
                cols: 4,
                usage: 0xF,
                half: false,
            },
            SigElement {
                name: "PRIMDATA".into(),
                index: 0,
                kind: 0,
                comp: 9,
                interp: 1,
                row: Some(1),
                col: 0,
                cols: 4,
                usage: 0xF,
                half: false,
            },
        ]);

        let mut vs = unpack_vertex_shader(&emu, Some(&ps)).unwrap();
        let c = Container::parse(&vs).unwrap();
        let osg1 = super::super::signature(c.part(FourCC::OSG1).unwrap().data).unwrap();
        assert_eq!(osg1.len(), 3);
        assert_eq!(osg1[0].name, "COLOR");
        assert_eq!(osg1[0].row, Some(0));
        assert_eq!(osg1[1].name, "PRIMDATA");
        assert_eq!(osg1[1].row, Some(1));
        assert_eq!(osg1[2].name, "SV_Position");
        assert_eq!(osg1[2].row, Some(2));
        assert_eq!(
            psv_input_interps(c.part(FourCC::PSV0).unwrap().data),
            vec![(0, 0, 1, 0)]
        );
        let ps_c = Container::parse(&ps).unwrap();
        assert_eq!(
            psv_input_interps(ps_c.part(FourCC::PSV0).unwrap().data),
            vec![(0, 0, 0, 2), (1, 0, 0, 1)]
        );

        // Depth-only pipeline (ps == None) still emits SV_Position.
        let mut depth_vs = unpack_vertex_shader(&emu, None).unwrap();
        let dc = Container::parse(&depth_vs).unwrap();
        let dosg = super::super::signature(dc.part(FourCC::OSG1).unwrap().data).unwrap();
        assert_eq!(dosg.len(), 1);
        assert_eq!(dosg[0].kind, 3);

        if let Ok(mut v) = Validator::load(Validator::default_path().as_deref()) {
            v.validate(&mut vs).unwrap();
            v.validate(&mut depth_vs).unwrap();
        }
    }

    use super::super::host::{EmulatedMeshDraw, Host};
    use super::super::{HostShader, emit_with};
    use crate::native::{HwStage, MeshIo, NativeProgram, PsInput, StageIo, Usage, VsOutput};
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;
    const S_BARRIER: u32 = 0xBF8A_0000;

    const fn sgpr(n: u8) -> u16 {
        n as u16
    }

    const fn vgpr(n: u8) -> u16 {
        256 + n as u16
    }

    const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
        0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
    }

    const fn vop1_lit(op: u16, vdst: u8, lit: u32) -> [u32; 2] {
        [vop1(op, vdst, 0xFF), lit]
    }

    const fn vop2(op: u16, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
        (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
    }

    const fn vopc(op: u16, src0: u16, vsrc1: u8) -> u32 {
        0x7C00_0000 | (op as u32 & 0xFF) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
    }

    const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
        [
            0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
            (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
        ]
    }

    const fn ds(op: u8, vdst: u8, vaddr: u8, vdata0: u8, vdata1: u8, offset: u16) -> [u32; 2] {
        [
            0xD800_0000 | (op as u32) << 18 | offset as u32,
            (vdst as u32) << 24 | (vdata1 as u32) << 16 | (vdata0 as u32) << 8 | vaddr as u32,
        ]
    }

    const fn exp(target: u8, en: u8, v: [u8; 4], done: bool, vm: bool) -> [u32; 2] {
        [
            0x3E << 26
                | (vm as u32) << 12
                | (done as u32) << 11
                | (target as u32 & 0x3F) << 4
                | (en as u32 & 0xF),
            (v[0] as u32) | (v[1] as u32) << 8 | (v[2] as u32) << 16 | (v[3] as u32) << 24,
        ]
    }

    const fn v_interp_mov_f32(vdst: u8, attr: u8, chan: u8) -> u32 {
        0xC800_0000 | (vdst as u32) << 18 | 2 << 16 | (attr as u32) << 10 | (chan as u32) << 8 | 2
    }

    fn sig_el(name: &str, kind: u8, comp: u8, row: Option<u32>, usage: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row,
            col: 0,
            cols: 4,
            usage,
            half: false,
        }
    }

    fn compile_ms(
        words: &[u32],
        io_outputs: Vec<VsOutput>,
        osg1: &[SigElement],
        psg1: &[SigElement],
        (threads, max_vertices, max_primitives, lds_bytes): (u32, u32, u32, u32),
    ) -> HostShader {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Ms,
            wave64: false,
            vgprs: 32,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Ms(MeshIo {
                outputs: io_outputs,
                clip_cntl: 0,
                threads,
                max_vertices,
                max_primitives,
                topology: 2,
                lds_bytes,
            }),
            code,
        };
        let osg_bytes = super::super::container::signature(osg1, false);
        let psg_bytes = super::super::container::signature(psg1, false);
        let container_bytes =
            crate::dxbc::build(&[(FourCC::OSG1, &osg_bytes), (FourCC::PSG1, &psg_bytes)]);
        let c = Container::parse(&container_bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let raised = io::raise(&p, &native).unwrap();
        let shader = emit_with(&c, &native, &p, &st, &raised, &res, MeshMode::Emulate).unwrap();
        if let Ok(mut v) = Validator::load(Validator::default_path().as_deref()) {
            v.validate(&mut shader.dxil.clone())
                .expect("emulated mesh CS validation failed");
        }
        shader
    }

    fn compile_flat_ps(semantic_name: &str) -> HostShader {
        let words = [
            v_interp_mov_f32(0, 0, 0),
            v_interp_mov_f32(1, 0, 1),
            v_interp_mov_f32(2, 0, 2),
            v_interp_mov_f32(3, 0, 3),
            exp(0, 0xF, [0, 1, 2, 3], true, true)[0],
            exp(0, 0xF, [0, 1, 2, 3], true, true)[1],
            S_ENDPGM,
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: 0xF,
                    attr: 0,
                    flat: true,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: Vec::new(),
            },
            code,
        };
        let isg1 =
            super::super::container::signature(&[sig_el(semantic_name, 0, 9, Some(0), 0xF)], true);
        let osg1 =
            super::super::container::signature(&[sig_el("SV_Target", 16, 9, Some(0), 0xF)], false);
        let container_bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
        let c = Container::parse(&container_bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let raised = io::raise(&p, &native).unwrap();
        let shader = emit_with(&c, &native, &p, &st, &raised, &res, MeshMode::Emulate).unwrap();
        if let Ok(mut v) = Validator::load(Validator::default_path().as_deref()) {
            v.validate(&mut shader.dxil.clone())
                .expect("PS validation failed");
        }
        shader
    }

    fn run_on_hosts(mut f: impl FnMut(&str, &Host)) {
        let _one = super::super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut ran = 0;
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                eprintln!("no SM 6.6 {label} device; skipped");
                continue;
            };
            ran += 1;
            f(label, &host);
        }
        assert!(
            ran > 0 || std::env::var("CI").is_err(),
            "no SM 6.6 host available"
        );
    }

    #[test]
    fn emulated_mesh_draws_triangle_and_culls_bit31_and_unexported() {
        let build_variant = |cull_prim0: bool| {
            let mut words = Vec::new();
            // v0 = tid (0..7)
            words.extend(vop3(0x365, 0, 193, 128, 0));
            // Default position: (-1, -1, 0.5, 1.0)
            words.extend(vop1_lit(0x001, 1, (-1.0f32).to_bits()));
            words.extend(vop1_lit(0x001, 2, (-1.0f32).to_bits()));
            words.extend(vop1_lit(0x001, 3, 0.5f32.to_bits()));
            words.extend(vop1_lit(0x001, 4, 1.0f32.to_bits()));
            // Default color: RED (1, 0, 0, 1); scratch constants v10=3.0, v11=1.0, v12=0.0
            words.extend(vop1_lit(0x001, 5, 1.0f32.to_bits()));
            words.extend(vop1_lit(0x001, 6, 0.0f32.to_bits()));
            words.extend(vop1_lit(0x001, 7, 0.0f32.to_bits()));
            words.extend(vop1_lit(0x001, 8, 1.0f32.to_bits()));
            words.extend(vop1_lit(0x001, 10, 3.0f32.to_bits()));
            words.extend(vop1_lit(0x001, 11, 1.0f32.to_bits()));
            words.extend(vop1_lit(0x001, 12, 0.0f32.to_bits()));
            // tid == 1 or tid == 4 -> pos.x = 3.0
            words.push(vopc(0x0C2, 129, 0));
            words.push(vop2(0x001, 1, vgpr(1), 10));
            words.push(vopc(0x0C2, 132, 0));
            words.push(vop2(0x001, 1, vgpr(1), 10));
            // tid == 2 or tid == 5 -> pos.y = 3.0
            words.push(vopc(0x0C2, 130, 0));
            words.push(vop2(0x001, 2, vgpr(2), 10));
            words.push(vopc(0x0C2, 133, 0));
            words.push(vop2(0x001, 2, vgpr(2), 10));
            // tid >= 3 (3 <= tid, i.e., 3 < tid + 1 or 2 < tid) -> green (0, 1, 0, 1)
            words.push(vopc(0x0C1, 130, 0)); // 2 < v0
            words.push(vop2(0x001, 5, vgpr(5), 12));
            words.push(vop2(0x001, 6, vgpr(6), 11));
            words.extend(exp(12, 0xF, [1, 2, 3, 4], false, false));
            words.extend(exp(32, 0xF, [5, 6, 7, 8], false, false));
            // v9 = (3 | 4<<10 | 5<<20) | 0x8000_0000 (bit-31 culled green tri)
            words.extend(vop1_lit(0x001, 9, 0x8050_1003));
            if !cull_prim0 {
                words.extend(vop1_lit(0x001, 13, 0x0020_0400));
                words.push(vopc(0x0C2, 128, 0)); // v0 == 0
                words.push(vop2(0x001, 9, vgpr(9), 13));
            }
            // Only tid < 2 (2 > v0) exports `exp prim`; slot 2 relies on prologue cull.
            words.push(vopc(0x0C4, 130, 0)); // 2 > v0
            words.push(0xBE80_3C6A); // s_and_saveexec_b32 s0, vcc_lo
            words.push(0xBF88_0002); // s_cbranch_execz +2
            words.extend(exp(20, 0x1, [9, 0, 0, 0], true, false));
            words.push(0x887E_007E); // s_or_b32 exec_lo, exec_lo, s0
            words.push(S_ENDPGM);

            compile_ms(
                &words,
                vec![VsOutput {
                    usage: Usage::Generic,
                    row: 0,
                    param: 0,
                    fp16: false,
                    pack2: false,
                }],
                &[
                    sig_el("SV_Position", 3, 9, None, 0xF),
                    sig_el("COLOR", 0, 9, Some(0), 0xF),
                ],
                &[],
                (8, 6, 3, 0),
            )
        };

        let ms_active = build_variant(false);
        let ms_culled = build_variant(true);
        let ps = compile_flat_ps("COLOR");
        let emu_active = ms_active.mesh_emu.as_ref().unwrap().clone();
        let emu_culled = ms_culled.mesh_emu.as_ref().unwrap().clone();

        run_on_hosts(|label, host| {
            let pipes_active = host.emulated_mesh_pipelines(&ms_active, &ps).unwrap();
            let pixels = host
                .draw_emulated_mesh(
                    &pipes_active,
                    &EmulatedMeshDraw {
                        cs_roots: Vec::new(),
                        ps_roots: Vec::new(),
                        size: (8, 8),
                        groups: 1,
                        max_primitives: emu_active.max_primitives,
                        group_bytes: emu_active.group_bytes(),
                        base_group: 0,
                        slot_map: None,
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: active draw failed: {e}"));
            assert!(
                pixels.iter().all(|&p| p == [1.0, 0.0, 0.0, 1.0]),
                "{label}: expected full red, got {:?}",
                pixels[0]
            );

            let pipes_culled = host.emulated_mesh_pipelines(&ms_culled, &ps).unwrap();
            let pixels = host
                .draw_emulated_mesh(
                    &pipes_culled,
                    &EmulatedMeshDraw {
                        cs_roots: Vec::new(),
                        ps_roots: Vec::new(),
                        size: (8, 8),
                        groups: 1,
                        max_primitives: emu_culled.max_primitives,
                        group_bytes: emu_culled.group_bytes(),
                        base_group: 0,
                        slot_map: None,
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: culled draw failed: {e}"));
            assert!(
                pixels.iter().all(|&p| p == [0.0, 0.0, 0.0, 0.0]),
                "{label}: expected all culled, got {:?}",
                pixels[0]
            );
        });
    }

    #[test]
    fn emulated_mesh_group_id_supports_base_group_and_slot_map() {
        let mut words = Vec::new();
        // s2 = group_id (from v5 via v_readfirstlane_b32)
        words.push(vop1(0x002, 2, vgpr(5)));
        // v0 = tid (0..3)
        words.extend(vop3(0x365, 0, 193, 128, 0));
        // v5 = float(s2), v6 = 0, v7 = 0, v8 = 1.0 -> COLOR0.r = group_id
        words.push(vop1(0x006, 5, sgpr(2)));
        words.extend(vop1_lit(0x001, 6, 0.0f32.to_bits()));
        words.extend(vop1_lit(0x001, 7, 0.0f32.to_bits()));
        words.extend(vop1_lit(0x001, 8, 1.0f32.to_bits()));
        // If s2 == 36 (group 9 * mesh_stride 4): draw right triangle (0, 1), (2, 1), (0, -3)
        // covering top-right; otherwise draw left triangle (-1, 1), (0, 1), (-1, -3) covering top-left.
        // v10 = 36, v11 = s2; compare vcc = (v11 == 36)
        words.push(vop1(0x001, 11, sgpr(2)));
        words.extend(vop1_lit(0x001, 10, 36));
        // Base left x (-1.0), vs right x (0.0) in v12
        words.extend(vop1_lit(0x001, 1, (-1.0f32).to_bits()));
        words.extend(vop1_lit(0x001, 12, 0.0f32.to_bits()));
        words.push(vopc(0x0C2, vgpr(10), 11));
        words.push(vop2(0x001, 1, vgpr(1), 12)); // base x: -1 or 0
        // If tid == 1: x += 1.0 (left: -1->0) or +2.0 (right: 0->2)
        words.extend(vop1_lit(0x001, 13, 1.0f32.to_bits()));
        words.extend(vop1_lit(0x001, 14, 2.0f32.to_bits()));
        words.push(vopc(0x0C2, vgpr(10), 11));
        words.push(vop2(0x001, 13, vgpr(13), 14)); // width: 1.0 or 2.0
        words.extend(vop2_add_f32_helper(14, 1, 13));
        words.push(vopc(0x0C2, 129, 0)); // tid == 1
        words.push(vop2(0x001, 1, vgpr(1), 14));
        // y: default 1.0, tid == 2 -> -3.0; z = 0.5, w = 1.0
        words.extend(vop1_lit(0x001, 2, 1.0f32.to_bits()));
        words.extend(vop1_lit(0x001, 15, (-3.0f32).to_bits()));
        words.push(vopc(0x0C2, 130, 0)); // tid == 2
        words.push(vop2(0x001, 2, vgpr(2), 15));
        words.extend(vop1_lit(0x001, 3, 0.5f32.to_bits()));
        words.extend(vop1_lit(0x001, 4, 1.0f32.to_bits()));

        words.extend(exp(12, 0xF, [1, 2, 3, 4], false, false));
        words.extend(exp(32, 0xF, [5, 6, 7, 8], false, false));
        words.extend(vop1_lit(0x001, 9, 0x0020_0400));
        words.extend(exp(20, 0x1, [9, 0, 0, 0], true, false));
        words.push(S_ENDPGM);

        let ms = compile_ms(
            &words,
            vec![VsOutput {
                usage: Usage::Generic,
                row: 0,
                param: 0,
                fp16: false,
                pack2: false,
            }],
            &[
                sig_el("SV_Position", 3, 9, None, 0xF),
                sig_el("COLOR", 0, 9, Some(0), 0xF),
            ],
            &[],
            (4, 3, 1, 0),
        );
        let emu = ms.mesh_emu.as_ref().unwrap();
        assert!(emu.reads_group_id);
        let ps = compile_flat_ps("COLOR");

        run_on_hosts(|label, host| {
            let pipes = host.emulated_mesh_pipelines(&ms, &ps).unwrap();
            // Chunked direct dispatch: 1 group starting at base_group = 5 -> top-left = 5.0
            let direct = host
                .draw_emulated_mesh(
                    &pipes,
                    &EmulatedMeshDraw {
                        cs_roots: Vec::new(),
                        ps_roots: Vec::new(),
                        size: (8, 8),
                        groups: 1,
                        max_primitives: emu.max_primitives,
                        group_bytes: emu.group_bytes(),
                        base_group: 5,
                        slot_map: None,
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: base_group draw failed: {e}"));
            assert_eq!(direct[0], [20.0, 0.0, 0.0, 1.0], "{label}: direct[0]");

            // Multi-record indirect dispatch: 2 local groups reading slot_map [2, 9]
            let indirect = host
                .draw_emulated_mesh(
                    &pipes,
                    &EmulatedMeshDraw {
                        cs_roots: Vec::new(),
                        ps_roots: Vec::new(),
                        size: (8, 8),
                        groups: 2,
                        max_primitives: emu.max_primitives,
                        group_bytes: emu.group_bytes(),
                        base_group: INDIRECT_SLOT_MAP_FLAG,
                        slot_map: Some(&[2, 9]),
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: slot_map draw failed: {e}"));
            assert_eq!(indirect[0], [8.0, 0.0, 0.0, 1.0], "{label}: top-left");
            assert_eq!(indirect[7], [36.0, 0.0, 0.0, 1.0], "{label}: top-right");
        });
    }

    const fn vop2_add_f32_helper(vdst: u8, src0_v: u8, vsrc1: u8) -> [u32; 1] {
        [vop2(0x003, vdst, vgpr(src0_v), vsrc1)]
    }

    #[test]
    fn emulated_mesh_passes_per_primitive_psg1_to_ps() {
        let mut words = Vec::new();
        words.extend(vop3(0x365, 0, 193, 128, 0));
        words.extend(vop1_lit(0x001, 1, (-1.0f32).to_bits()));
        words.extend(vop1_lit(0x001, 2, (-1.0f32).to_bits()));
        words.extend(vop1_lit(0x001, 3, 0.5f32.to_bits()));
        words.extend(vop1_lit(0x001, 4, 1.0f32.to_bits()));
        words.extend(vop1_lit(0x001, 10, 3.0f32.to_bits()));
        words.push(vopc(0x0C2, 129, 0));
        words.push(vop2(0x001, 1, vgpr(1), 10));
        words.push(vopc(0x0C2, 130, 0));
        words.push(vop2(0x001, 2, vgpr(2), 10));
        words.extend(exp(12, 0xF, [1, 2, 3, 4], false, false));

        words.extend(vop1_lit(0x001, 5, 0.25f32.to_bits()));
        words.extend(vop1_lit(0x001, 6, 0.5f32.to_bits()));
        words.extend(vop1_lit(0x001, 7, 0.75f32.to_bits()));
        words.extend(vop1_lit(0x001, 8, 1.0f32.to_bits()));
        // Per-primitive parameter 0 (`param = 0x80` -> PSG1 PRIMCOLOR at row 0)
        words.extend(exp(32, 0xF, [5, 6, 7, 8], false, false));

        words.extend(vop1_lit(0x001, 9, 0x0020_0400));
        words.extend(exp(20, 0x1, [9, 0, 0, 0], true, false));
        words.push(S_ENDPGM);

        let ms = compile_ms(
            &words,
            vec![VsOutput {
                usage: Usage::Generic,
                row: 0,
                param: 0x80,
                fp16: false,
                pack2: false,
            }],
            &[sig_el("SV_Position", 3, 9, None, 0xF)],
            &[sig_el("PRIMCOLOR", 0, 9, Some(0), 0xF)],
            (4, 3, 1, 0),
        );
        let ps = compile_flat_ps("PRIMCOLOR");
        let emu = ms.mesh_emu.as_ref().unwrap().clone();

        run_on_hosts(|label, host| {
            let pipes = host.emulated_mesh_pipelines(&ms, &ps).unwrap();
            let pixels = host
                .draw_emulated_mesh(
                    &pipes,
                    &EmulatedMeshDraw {
                        cs_roots: Vec::new(),
                        ps_roots: Vec::new(),
                        size: (8, 8),
                        groups: 1,
                        max_primitives: emu.max_primitives,
                        group_bytes: emu.group_bytes(),
                        base_group: 0,
                        slot_map: None,
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: draw failed: {e}"));
            assert!(
                pixels.iter().all(|&p| p == [0.25, 0.5, 0.75, 1.0]),
                "{label}: expected [0.25, 0.5, 0.75, 1.0], got {:?}",
                pixels[0]
            );
        });
    }

    #[test]
    fn emulated_mesh_supports_lds_barrier_and_quad_swizzle() {
        let mut words = Vec::new();
        // v0 = tid (0..3)
        words.extend(vop3(0x365, 0, 193, 128, 0));
        // v14 = tid * 4 (LDS byte offset)
        words.push(vop2(0x01A, 14, 130, 0)); // v_lshlrev_b32 v14, 2, v0
        // v15 = float(tid + 1) * 0.25 -> thread 2 writes 0.75 into lds[8]
        words.push(vop2(0x025, 15, 129, 0)); // v_add_nc_u32 v15, 1, v0
        words.push(vop1(0x006, 15, vgpr(15))); // v_cvt_f32_u32 v15, v15
        words.extend(vop1_lit(0x001, 16, 0.25f32.to_bits()));
        words.push(vop2(0x008, 15, vgpr(15), 16)); // v_mul_f32 v15, v15, v16
        words.extend(ds(0x0D, 0, 14, 15, 0, 0)); // ds_write_b32 v14, v15
        words.push(S_BARRIER);
        // Each lane reads ((tid + 3) & 3) * 4 -> lane 3 reads lds[8] (0.75)
        words.push(vop2(0x025, 14, 131, 0)); // v_add_nc_u32 v14, 3, v0
        words.push(vop2(0x01B, 14, 131, 14)); // v_and_b32 v14, 3, v14
        words.push(vop2(0x01A, 14, 130, 14)); // v_lshlrev_b32 v14, 2, v14
        words.extend(ds(0x36, 5, 14, 0, 0, 0)); // ds_read_b32 v5, v14
        // Broadcast quad lane 3's value (0.75) to all 4 lanes in the quad
        words.extend(ds(0x35, 5, 5, 0, 0, 0x80FF)); // ds_swizzle_b32 v5, v5 offset:0x80FF

        // Position + COLOR(v5, 0, 0, 1)
        words.extend(vop1_lit(0x001, 1, (-1.0f32).to_bits()));
        words.extend(vop1_lit(0x001, 2, (-1.0f32).to_bits()));
        words.extend(vop1_lit(0x001, 3, 0.5f32.to_bits()));
        words.extend(vop1_lit(0x001, 4, 1.0f32.to_bits()));
        words.extend(vop1_lit(0x001, 10, 3.0f32.to_bits()));
        words.push(vopc(0x0C2, 129, 0));
        words.push(vop2(0x001, 1, vgpr(1), 10));
        words.push(vopc(0x0C2, 130, 0));
        words.push(vop2(0x001, 2, vgpr(2), 10));
        words.extend(vop1_lit(0x001, 6, 0.0f32.to_bits()));
        words.extend(vop1_lit(0x001, 7, 0.0f32.to_bits()));
        words.extend(vop1_lit(0x001, 8, 1.0f32.to_bits()));
        words.extend(exp(12, 0xF, [1, 2, 3, 4], false, false));
        words.extend(exp(32, 0xF, [5, 6, 7, 8], false, false));
        words.extend(vop1_lit(0x001, 9, 0x0020_0400));
        words.extend(exp(20, 0x1, [9, 0, 0, 0], true, false));
        words.push(S_ENDPGM);

        let ms = compile_ms(
            &words,
            vec![VsOutput {
                usage: Usage::Generic,
                row: 0,
                param: 0,
                fp16: false,
                pack2: false,
            }],
            &[
                sig_el("SV_Position", 3, 9, None, 0xF),
                sig_el("COLOR", 0, 9, Some(0), 0xF),
            ],
            &[],
            (4, 3, 1, 16),
        );
        let ps = compile_flat_ps("COLOR");
        let emu = ms.mesh_emu.as_ref().unwrap().clone();

        run_on_hosts(|label, host| {
            let pipes = host.emulated_mesh_pipelines(&ms, &ps).unwrap();
            let pixels = host
                .draw_emulated_mesh(
                    &pipes,
                    &EmulatedMeshDraw {
                        cs_roots: Vec::new(),
                        ps_roots: Vec::new(),
                        size: (8, 8),
                        groups: 1,
                        max_primitives: emu.max_primitives,
                        group_bytes: emu.group_bytes(),
                        base_group: 0,
                        slot_map: None,
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: draw failed: {e}"));
            assert!(
                pixels.iter().all(|&p| p == [0.75, 0.0, 0.0, 1.0]),
                "{label}: expected [0.75, 0.0, 0.0, 1.0], got {:?}",
                pixels[0]
            );
        });
    }
}
