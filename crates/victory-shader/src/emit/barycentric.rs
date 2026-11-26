//! A geometry-stage fallback for devices without `SV_Barycentrics`.
//!
//! The pixel shader uses a private ordinary varying instead of the special
//! system value. This pass-through stage assigns the three triangle vertices
//! `(1,0,0)`, `(0,1,0)` and `(0,0,1)`; the pixel shader's interpolation mode
//! then produces the same perspective- or screen-linear barycentrics.

use super::bitcode::{Attr, Term, V};
use super::container::SigElement;
use super::shader::{Geometry, Shader};
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

/// Private semantic used only when the host cannot provide the system value.
pub(crate) const FALLBACK_SEMANTIC: &str = "VICTORY_BARYCENTRIC_FALLBACK";

/// Floating-point bits of one channel of a vertex's one-hot weight.
pub(super) fn weight_bits(channel: u8, vertex: u8) -> u32 {
    if channel == vertex {
        1.0f32.to_bits()
    } else {
        0.0f32.to_bits()
    }
}

/// Whether `ps_dxil` has an input synthesized by the barycentric fallback.
pub(crate) fn is_fallback_input(e: &SigElement) -> bool {
    e.kind == 0 && e.name == FALLBACK_SEMANTIC
}

/// Find or add the pixel input for one barycentric interpolation mode.
pub(crate) fn input_id(
    inputs: &mut Vec<SigElement>,
    mode: u8,
    fallback: bool,
) -> std::result::Result<usize, &'static str> {
    let is_bary = |e: &SigElement| e.kind == 28 || e.name == FALLBACK_SEMANTIC;
    let existing = inputs
        .iter()
        .position(|e| is_bary(e) && (e.interp == mode || e.usage == 0));
    let id = if fallback {
        let slot = existing.map_or_else(
            || inputs.iter().filter(|e| is_bary(e)).count(),
            |id| inputs[..id].iter().filter(|e| is_bary(e)).count(),
        );
        if slot >= 2 {
            return Err("barycentrics of a third interpolation mode");
        }
        let row = existing
            .and_then(|id| inputs[id].row)
            .unwrap_or_else(|| {
                inputs
                    .iter()
                    .filter_map(|e| e.row)
                    .max()
                    .map_or(0, |row| row + 1)
            });
        match existing {
            Some(id) => {
                let e = &mut inputs[id];
                e.name = FALLBACK_SEMANTIC.into();
                e.index = slot as u32;
                e.kind = 0;
                e.comp = 9;
                e.interp = mode;
                e.row = Some(row);
                e.col = 0;
                e.cols = 3;
                e.usage = 0x7;
                id
            }
            None => {
                inputs.push(SigElement {
                    name: FALLBACK_SEMANTIC.into(),
                    index: slot as u32,
                    kind: 0,
                    comp: 9,
                    interp: mode,
                    row: Some(row),
                    col: 0,
                    cols: 3,
                    usage: 0x7,
                    half: false,
                });
                inputs.len() - 1
            }
        }
    } else {
        match existing {
            Some(id) => id,
            None => {
                let index = inputs.iter().filter(|e| e.kind == 28).count() as u32;
                if index >= 2 {
                    return Err("barycentrics of a third interpolation mode");
                }
                inputs.push(SigElement {
                    name: "SV_Barycentrics".into(),
                    index,
                    kind: 28,
                    comp: 9,
                    interp: mode,
                    row: None,
                    col: 0,
                    cols: 3,
                    usage: 0,
                    half: false,
                });
                inputs.len() - 1
            }
        }
    };
    let e = &mut inputs[id];
    if e.col != 0 || e.cols < 3 {
        return Err("a console barycentrics element without i and j");
    }
    e.interp = mode;
    if !fallback {
        e.row = None;
    }
    Ok(id)
}

/// Whether `ps_dxil` needs the geometry-stage fallback.
pub fn requires_barycentric_fallback(ps_dxil: &[u8]) -> Result<bool, ShaderError> {
    let c = Container::parse(ps_dxil)?;
    let inputs = match c.part(FourCC::ISG1) {
        Some(part) => super::signature(part.data)?,
        None => Vec::new(),
    };
    Ok(inputs.iter().any(is_fallback_input))
}

/// A triangle pass-through geometry shader that adds the fallback barycentric
/// varyings requested by `ps_dxil`.
pub fn barycentric_gs(vs_dxil: &[u8], ps_dxil: &[u8]) -> Result<Option<Vec<u8>>, ShaderError> {
    let vs = Container::parse(vs_dxil)?;
    let ps = Container::parse(ps_dxil)?;
    let signature = |c: &Container, fourcc| -> Result<Vec<SigElement>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => super::signature(part.data),
            None => Ok(Vec::new()),
        }
    };
    let outputs = signature(&vs, FourCC::OSG1)?;
    let wanted = signature(&ps, FourCC::ISG1)?;
    let barycentrics: Vec<_> = wanted
        .iter()
        .filter(|e| is_fallback_input(e))
        .cloned()
        .collect();
    if barycentrics.is_empty() {
        return Ok(None);
    }
    let modes = ps
        .part(FourCC::PSV0)
        .map(|part| super::mesh_emu::psv_input_interps(part.data))
        .unwrap_or_default();
    build(outputs, &wanted, &modes).map(Some)
}

fn build(
    mut outputs: Vec<SigElement>,
    ps_inputs: &[SigElement],
    ps_modes: &[(u32, u8, u8, u8)],
) -> Result<Vec<u8>, ShaderError> {
    let unsupported = |what| ShaderError::Unsupported { offset: 0, what };
    if outputs.iter().any(|e| e.row.is_none()) {
        return Err(unsupported(
            "a barycentric fallback output without a register",
        ));
    }
    let Some(_position) = outputs
        .iter()
        .position(|e| e.kind == 3 && e.comp == 9 && e.cols == 4)
    else {
        return Err(unsupported(
            "a barycentric fallback without a four-component position",
        ));
    };
    for e in &mut outputs {
        // Match the interpolation classes accepted by the existing
        // pass-through rectangle stage. Pixel inputs carry the requested
        // centroid/sample qualifier themselves.
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
        e.usage = (1 << e.cols) - 1;
    }

    let pass_through_count = outputs.len();
    let gs_inputs = outputs.clone();
    let mut gs_outputs = outputs;
    let primitive = ps_inputs
        .iter()
        .find(|e| e.kind == 10 && e.row.is_some())
        .filter(|_| !gs_outputs.iter().any(|o| o.kind == 10));
    if let Some(e) = primitive {
        let row = gs_outputs
            .iter()
            .filter_map(|o| o.row)
            .max()
            .map_or(0, |r| r + 1);
        gs_outputs.push(SigElement {
            name: e.name.clone(),
            index: e.index,
            kind: e.kind,
            comp: e.comp,
            interp: 1,
            row: Some(row),
            col: 0,
            cols: e.cols,
            usage: (1 << e.cols) - 1,
            half: false,
        });
    }
    let primitive_output = primitive.map(|_| gs_outputs.len() - 1);

    let first_bary_row = gs_outputs
        .iter()
        .filter_map(|e| e.row)
        .max()
        .map_or(0, |r| r + 1);
    let mut bary_output_ids = Vec::new();
    for (row, bary) in (first_bary_row..).zip(ps_inputs.iter().filter(|e| is_fallback_input(e))) {
        if bary.comp != 9 || bary.col != 0 || bary.cols != 3 {
            return Err(unsupported("a malformed barycentric fallback pixel input"));
        }
        let mut output = bary.clone();
        output.interp = ps_modes
            .iter()
            .find(|&&(row, col, kind, _)| {
                Some(row) == bary.row && col == bary.col && kind == bary.kind
            })
            .map_or(2, |&(_, _, _, mode)| mode);
        output.row = Some(row);
        output.col = 0;
        output.cols = 3;
        output.usage = 0x7;
        bary_output_ids.push(gs_outputs.len());
        gs_outputs.push(output);
    }

    let scalars: u32 = gs_outputs.iter().map(|e| u32::from(e.cols)).sum();
    if scalars * 3 > 1024 {
        return Err(unsupported(
            "a barycentric fallback with more output components than a geometry shader emits",
        ));
    }
    let mut s = Shader::new(false);
    s.geometry = Some(Geometry { max_vertices: 3 });
    s.inputs = gs_inputs;
    s.outputs = gs_outputs;
    let entry = s.entry;
    let stream = s.m.i8(0);
    let primitive_id =
        primitive_output.map(|_| s.op(entry, "primitiveID.i32", Attr::ReadNone, s.t.i32, 108, &[]));

    let output_plan = s.outputs.clone();
    for vertex in 0..3u32 {
        for (id, output) in output_plan.iter().enumerate() {
            if id < pass_through_count {
                for col in 0..output.cols {
                    let value = load(&mut s, id, col, vertex);
                    store(&mut s, id, col, value);
                }
            } else if Some(id) == primitive_output {
                store(
                    &mut s,
                    id,
                    0,
                    primitive_id.expect("primitive output has an id"),
                );
            } else if bary_output_ids.contains(&id) {
                for col in 0..3u8 {
                    let value = s.m.f32(weight_bits(col, vertex as u8));
                    store(&mut s, id, col, value);
                }
            }
        }
        s.op(entry, "emitStream", Attr::NoUnwind, s.t.void, 97, &[stream]);
    }
    s.op(entry, "cutStream", Attr::NoUnwind, s.t.void, 98, &[stream]);
    s.m.terminate(entry, Term::Ret);
    Ok(s.finish())
}

/// Column `col` of input `id` at vertex `vertex` of the triangle.
fn load(s: &mut Shader, id: usize, col: u8, vertex: u32) -> V {
    let (name, ty) = if s.inputs[id].comp == 9 {
        ("loadInput.f32", s.t.f32)
    } else {
        ("loadInput.i32", s.t.i32)
    };
    let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), s.m.i32(vertex)];
    let entry = s.entry;
    s.op(entry, name, Attr::ReadNone, ty, 4, &args)
}

fn store(s: &mut Shader, id: usize, col: u8, value: V) {
    let suffix = if s.outputs[id].comp == 9 {
        ".f32"
    } else {
        ".i32"
    };
    let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), value];
    let entry = s.entry;
    s.op(
        entry,
        &format!("storeOutput{suffix}"),
        Attr::NoUnwind,
        s.t.void,
        5,
        &args,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dxbc::Container;

    fn element(
        name: &str,
        index: u32,
        kind: u8,
        comp: u8,
        interp: u8,
        row: u32,
        cols: u8,
    ) -> SigElement {
        SigElement {
            name: name.into(),
            index,
            kind,
            comp,
            interp,
            row: Some(row),
            col: 0,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        }
    }

    fn vertex_shader() -> Vec<u8> {
        let mut s = Shader::new(false);
        s.outputs = vec![
            element("SV_Position", 0, 3, 9, 4, 0, 4),
            element("TEXCOORD", 0, 0, 9, 2, 1, 2),
        ];
        let zero = s.m.f32(0);
        for (id, output) in s.outputs.clone().iter().enumerate() {
            for col in 0..output.cols {
                store(&mut s, id, col, zero);
            }
        }
        s.m.terminate(s.entry, Term::Ret);
        s.finish()
    }

    fn pixel_shader(bary_count: usize) -> Vec<u8> {
        let mut s = Shader::new(true);
        s.inputs = (0..bary_count)
            .map(|i| {
                element(
                    FALLBACK_SEMANTIC,
                    i as u32,
                    0,
                    9,
                    [2, 4][i],
                    2 + i as u32,
                    3,
                )
            })
            .collect();
        s.outputs = vec![element("SV_Target", 0, 16, 9, 0, 0, 4)];
        let zero = s.m.f32(0);
        for col in 0..4 {
            store(&mut s, 0, col, zero);
        }
        s.m.terminate(s.entry, Term::Ret);
        s.finish()
    }

    #[test]
    fn fallback_gs_passes_through_and_emits_each_requested_interpolation() {
        let vs = vertex_shader();
        let ps = pixel_shader(2);
        let gs = barycentric_gs(&vs, &ps).unwrap().unwrap();
        let c = Container::parse(&gs).unwrap();
        assert_eq!(c.part(FourCC::DXIL).unwrap().data[..4], [0x66, 0, 2, 0]);
        let psv = c.part(FourCC::PSV0).unwrap().data;
        assert_eq!(psv[28..34], [2, 0, 3, 0, 2, 4]);
        let sig = |fourcc| super::super::signature(c.part(fourcc).unwrap().data).unwrap();
        let inputs = sig(FourCC::ISG1);
        let outputs = sig(FourCC::OSG1);
        assert_eq!(inputs.len(), 2);
        assert_eq!(outputs.len(), 4);
        assert_eq!(outputs[0].name, "SV_Position");
        assert_eq!(outputs[1].name, "TEXCOORD");
        assert_eq!(
            outputs[2..]
                .iter()
                .map(|e| (e.name.as_str(), e.index, e.cols))
                .collect::<Vec<_>>(),
            [(FALLBACK_SEMANTIC, 0, 3), (FALLBACK_SEMANTIC, 1, 3)]
        );
        let psv = c.part(FourCC::PSV0).unwrap().data;
        let input_count = usize::from(psv[4 + 28]);
        let output_count = usize::from(psv[4 + 29]);
        let mut at = 4 + usize::try_from(u32::from_le_bytes(psv[..4].try_into().unwrap())).unwrap();
        let bindings =
            usize::try_from(u32::from_le_bytes(psv[at..at + 4].try_into().unwrap())).unwrap();
        at += 4;
        if bindings != 0 {
            let elem_size =
                usize::try_from(u32::from_le_bytes(psv[at..at + 4].try_into().unwrap())).unwrap();
            at += 4 + bindings * elem_size;
        }
        let string_bytes =
            usize::try_from(u32::from_le_bytes(psv[at..at + 4].try_into().unwrap())).unwrap();
        at += 4 + string_bytes;
        let index_count =
            usize::try_from(u32::from_le_bytes(psv[at..at + 4].try_into().unwrap())).unwrap();
        at += 4 + index_count * 4;
        let element_size =
            usize::try_from(u32::from_le_bytes(psv[at..at + 4].try_into().unwrap())).unwrap();
        at += 4;
        let modes: Vec<_> = (0..output_count)
            .map(|i| psv[at + (input_count + i) * element_size + 13])
            .collect();
        assert_eq!(&modes[2..], &[2, 4]);
        for (input, output) in inputs.iter().zip(&outputs) {
            assert_eq!((&input.name, input.index), (&output.name, output.index));
        }
        #[cfg(windows)]
        if let Ok(mut validator) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            validator.validate(&mut gs.clone()).unwrap();
        }
    }

    #[test]
    fn no_fallback_input_does_not_add_a_geometry_shader() {
        let vs = vertex_shader();
        let ps = pixel_shader(0);
        assert!(barycentric_gs(&vs, &ps).unwrap().is_none());
    }
}
