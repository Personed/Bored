//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, turn-range reduction for `v_sin_*`/`v_cos_*`, and the LDS
//! float atomics `ds_min_f32`/`ds_max_f32`.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    vop3_mod(op, vdst, src0, src1, src2, 0, 0, 0, false)
}

/// VOP3 `op vdst, src0, src1, src2` with explicit `abs`, `neg`, `omod`, `clamp`.
#[allow(clippy::too_many_arguments)]
const fn vop3_mod(
    op: u16,
    vdst: u8,
    src0: u16,
    src1: u16,
    src2: u16,
    abs: u8,
    neg: u8,
    omod: u8,
    clamp: bool,
) -> [u32; 2] {
    [
        0x35 << 26
            | (op as u32 & 0x3FF) << 16
            | (clamp as u32) << 15
            | (abs as u32 & 7) << 8
            | vdst as u32,
        (neg as u32 & 7) << 29
            | (omod as u32 & 3) << 27
            | (src0 as u32 & 0x1FF)
            | (src1 as u32 & 0x1FF) << 9
            | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
fn build_valu_cs(body: &[u32]) -> HostShader {
    build_cs(body, 1, 0)
}

/// [`build_valu_cs`] with `threads` threads in x (thread index in `v0`
/// on entry, before `body`) and `lds_bytes` of LDS (a multiple of 512).
fn build_cs(body: &[u32], threads: u32, lds_bytes: u32) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (5 << 1) | ((lds_bytes / 512) << 15), // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        emulate_mesh: false,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
        mesh_emu: None,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// One GPU test at a time. With these tests on the hardware adapter and WARP
/// from several test threads at once, now and then none of their dispatches
/// ever completes and the test binary hangs; and two GPU tests at once on
/// WARP can end the test process with an access violation. Other GPU test
/// modules take this lock too.
pub(super) static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    run_cases_on(shader, cases, &[("hardware", false), ("warp", true)], check);
}

fn run_cases_on(
    shader: &HostShader,
    cases: &[[u32; 4]],
    devices: &[(&str, bool)],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for &(label, warp) in devices {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    if ran == 0 && devices.iter().all(|&(_, warp)| !warp) {
        // A case for the hardware adapter alone, on a machine without one.
        eprintln!("no SM 6.6 hardware device; a hardware-only case not run");
        return;
    }
    assert!(ran > 0, "no SM 6.6 device available ({devices:?})");
}

/// These cases require a WaveSize(32) pipeline. D3D12 shader-model 6.6 support
/// alone does not guarantee that the selected adapter accepts that wave size.
fn hardware_supports_wave_size(lanes: u32) -> bool {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no hardware D3D12 device available for WaveSize({lanes}); skipped");
        return false;
    };
    match host.wave_lane_range() {
        Ok((min, max)) => {
            let supported = (min..=max).contains(&lanes);
            if !supported {
                eprintln!(
                    "hardware adapter supports wave sizes {min}..={max}, not WaveSize({lanes}); skipped"
                );
            }
            supported
        }
        Err(error) => {
            eprintln!("could not query hardware wave-size support ({error}); running test");
            true
        }
    }
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

/// MTBUF `op vdata, off, s[srsrc:srsrc+3], 0 format:format` (at byte 0).
const fn mtbuf(op: u8, vdata: u8, srsrc: u8, format: u8) -> [u32; 2] {
    [
        0x3A << 26 | (format as u32 & 0x7F) << 19 | (op as u32 & 7) << 16,
        0x80 << 24
            | (op as u32 >> 3 & 1) << 21
            | (srsrc as u32 >> 2 & 0x1F) << 16
            | (vdata as u32) << 8,
    ]
}

/// Stores `v1` to the buffer, then reads it back into `v0` with the tbuffer
/// load `op` in buffer format `format`.
fn tbuffer_load_of_v1(op: u8, format: u8) -> HostShader {
    let mut body = buffer_store_dword(1, 8, 0).to_vec();
    body.extend(mtbuf(op, 0, 8, format));
    build_valu_cs(&body)
}


#[test]
fn normalized_buffer_formats_load_their_ends_exactly() {
    // tbuffer_load_format_x.
    let x = 0;
    // (format, dword stored, value loaded)
    let cases: &[(u8, u32, f32)] = &[
        (1, 0xFF, 1.0),   // 8_UNORM
        (7, 0xFFFF, 1.0), // 16_UNORM
        (2, 0x7F, 1.0),   // 8_SNORM: the top, the one above the lowest, the lowest
        (2, 0x81, -1.0),
        (2, 0x80, -1.0),
        (8, 0x7FFF, 1.0), // 16_SNORM
        (8, 0x8001, -1.0),
        (8, 0x8000, -1.0),
    ];
    for &(format, stored, want) in cases {
        let shader = tbuffer_load_of_v1(x, format);
        run_cases(&shader, &[[stored, 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(
                got,
                want.to_bits(),
                "{backend}: format {format} of {stored:#x} = {got:#010x}, want {want}",
            );
        });
    }
}

/// Codes of the normalized formats sampled below: every 8-bit code; for
/// 16 bits a stride, the RX 6700 XT's division misses (257, 261, ...) and
/// the ends.
fn normalized_codes(bits: u32) -> Vec<u32> {
    if bits <= 10 {
        return (0..1 << bits).collect();
    }
    let edges = [1, 2, 3, 255, 256, 257, 261, 265, 269, 511, 512, 1000, 0x7FFF, 0x8000, 0x8001, 0xFFFE];
    (0..=0xFFFF).step_by(257).chain(edges).collect()
}

/// Every normalized code loads as the float nearest its exact quotient,
/// as the console's buffer unit converts it, not as the host divides (the
/// RX 6700 XT gave 125 of the 8-bit codes one ulp high and 16-bit codes
/// such as 257 one ulp low): unsigned and signed, 8, 10 (the x channel of
/// 2_10_10_10) and 16 bits, the signed formats' two lowest codes -1.0.
#[test]
fn normalized_buffer_formats_load_correctly_rounded_quotients() {
    let x = 0;
    // (format, bits, signed): 8_UNORM, 16_UNORM, 8_SNORM, 16_SNORM,
    // 2_10_10_10_UNORM and _SNORM.
    for (format, bits, signed) in [(1u8, 8u32, false), (7, 16, false), (2, 8, true), (8, 16, true), (50, 10, false), (51, 10, true)] {
        let shader = tbuffer_load_of_v1(x, format);
        let cases: Vec<[u32; 4]> = normalized_codes(bits).iter().map(|&a| [a, 0, 0, 0]).collect();
        run_cases(&shader, &cases, |backend, _, [a, ..], got| {
            let want = if signed {
                let top = (1i32 << (bits - 1)) - 1;
                let s = ((a << (32 - bits)) as i32) >> (32 - bits);
                (s.max(-top) as f32 / top as f32).to_bits()
            } else {
                (a as f32 / ((1u32 << bits) - 1) as f32).to_bits()
            };
            let sign = if signed { "signed" } else { "unsigned" };
            assert_eq!(got, want, "{backend}: {bits}-bit {sign} normalized {a:#x} loads as {got:#010x}, want {want:#010x}");
        });
    }
}

/// A title truncates `x * (2^n - 1)` back to the code it stored (the
/// material index a mesh program reads from a vertex colour's alpha, as
/// `v_mul_legacy_f32` by 255.0 then `v_cvt_u32_f32`): every code comes back
/// as itself, on hardware and WARP, which a quotient one ulp low breaks.
#[test]
fn normalized_codes_truncate_back_to_themselves() {
    let x = 0;
    for (format, bits) in [(1u8, 8u32), (7, 16)] {
        let scale = (((1u32 << bits) - 1) as f32).to_bits();
        let mut body = buffer_store_dword(1, 8, 0).to_vec();
        body.extend(mtbuf(x, 0, 8, format));
        // v_mul_legacy_f32 v0, literal, v0; v_cvt_u32_f32 v0, v0.
        body.extend([0x07 << 25 | 0xFF, scale]);
        body.push(vop1(0x07, 0, vgpr(0)));
        let shader = build_valu_cs(&body);
        let cases: Vec<[u32; 4]> = normalized_codes(bits).iter().map(|&a| [a, 0, 0, 0]).collect();
        run_cases(&shader, &cases, |backend, _, [a, ..], got| {
            assert_eq!(got, a, "{backend}: {bits}-bit code {a} comes back as {got}");
        });
    }
}

#[test]
fn packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves.
    // The same GFX10 source field (248) has a different bit pattern for a
    // packed floating-point and a packed integer instruction.
    let cases = [
        // v_pk_add_f16 v0, 0x3118, v1
        ([0xcc0f4000, 0x1a0202f8], 0x0000_3118),
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        // v_pk_add_u16 v0, 0x3e22f983, v1
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ];
    for (words, expected) in cases {
        let shader = build_valu_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: {words:08x?}");
        });
    }
}

/// The float `v_cvt_f32_f64` gives for `x`: Rust's rounding (to nearest
/// even), with results below the smallest normal float flushed to signed
/// zero as the console's default float mode does.
fn cvt_f32_f64_reference(x: f64) -> u32 {
    let r = x as f32;
    if r.is_subnormal() || (r == 0.0 && x != 0.0) {
        r.to_bits() & 0x8000_0000
    } else {
        r.to_bits()
    }
}

#[test]
#[ignore = "expects results below the smallest normal float flushed to zero; the IEEE lowering (v_cvt_f32_f64_matches_ieee754) is kept until the console's mode is checked"]
fn v_cvt_f32_f64_rounds_to_nearest_even() {
    // v_cvt_f32_f64 v0, v[1:2] (VOP1 opcode 0x0f)
    let shader = build_valu_cs(&[vop1(0x00f, 0, vgpr(1))]);
    let tie_even = 1.0f64 + f64::from(f32::EPSILON) / 2.0; // ties to 1.0
    let tie_odd = 1.0f64 + 1.5 * f64::from(f32::EPSILON); // ties up to 1 + 2 ulp
    let values = [
        0.0,
        -0.0,
        1.0,
        -2.5,
        std::f64::consts::PI,
        1.0e-3,
        123_456_789.123,
        tie_even,
        tie_odd,
        1.0 + f64::from(f32::EPSILON) * 0.75,
        f64::from(f32::MAX),
        f64::from(f32::MAX) * 1.000_000_1, // rounds to infinity
        1.0e300,
        -1.0e300,
        1.0e-40, // a float denormal: flushed
        -1.0e-300,
        f64::from(f32::MIN_POSITIVE),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let inputs: Vec<[u32; 4]> = values
        .iter()
        .map(|x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&shader, &inputs, |backend, i, _, got| {
        let x = values[i];
        if x.is_nan() {
            assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: cvt({x}) = {got:#010x}, want NaN"
            );
            return;
        }
        let want = cvt_f32_f64_reference(x);
        assert_eq!(
            got, want,
            "{backend}: cvt({x:e}) = {got:#010x}, want {want:#010x}"
        );
    });
}

/// The lane `v_permlane16_b32` (`cross` false) or `v_permlanex16_b32` reads
/// for `lane`, with selector dwords `s1` (lanes 0-7 of a row) and `s2`.
fn permlane_source(lane: u32, s1: u32, s2: u32, cross: bool) -> u32 {
    let sel = u64::from(s1) | u64::from(s2) << 32;
    let pick = (sel >> (4 * (lane & 15)) & 15) as u32;
    let row = if cross {
        (lane & !31) | ((lane & 16) ^ 16)
    } else {
        lane & !15
    };
    row | pick
}

#[test]
fn v_permlane16_and_x16_read_the_lanes_their_selectors_pick() {
    if !hardware_supports_wave_size(32) {
        return;
    }
    // 32 lanes, each with its lane number in v0; v1 = permute(v0, s0, s1);
    // lane `read` of v1 into s12, then into v0 for the store.
    let selectors: [(u32, u32); 3] = [
        (0x7654_3210, 0xfedc_ba98), // identity within the row
        (0x89ab_cdef, 0x0123_4567), // reversed
        (0x1111_1111, 0xf0f0_f0f0), // repeats
    ];
    for (op, cross) in [(0x377u16, false), (0x378, true)] {
        for read in [0u8, 5, 15, 16, 22, 31] {
            let mut body = Vec::new();
            body.extend_from_slice(&vop3(op, 1, vgpr(0), sgpr(0), sgpr(1)));
            // v_readlane_b32 s12, v1, read (inline constant 128 + read)
            body.extend_from_slice(&vop3(0x360, 12, vgpr(1), 128 + u16::from(read), 0));
            body.push(vop1(0x001, 0, sgpr(12))); // v_mov_b32 v0, s12
            let shader = build_cs(&body, 32, 0);
            let inputs: Vec<[u32; 4]> = selectors.iter().map(|&(a, b)| [a, b, 0, 0]).collect();
            // Cross-lane code declares WaveSize(32), which WARP's pipeline
            // creation refuses: hardware only.
            run_cases_on(
                &shader,
                &inputs,
                &[("hardware", false)],
                |backend, i, inp, got| {
                    let want = permlane_source(u32::from(read), inp[0], inp[1], cross);
                    assert_eq!(
                        got, want,
                        "{backend}: op {op:#x} selectors {:08x?}, lane {read}",
                        selectors[i]
                    );
                },
            );
        }
    }
}

#[test]
fn directed_f64_ceil_and_floor_write_both_words() {
    let values = [
        0.0,
        -0.0,
        f64::from_bits(1),
        f64::from_bits(0x8000_0000_0000_0001),
        0.75,
        -0.75,
        1.0,
        -1.0,
        1.5,
        -1.5,
        // Fractional fields of 33, 32 and 31 bits.
        524_288.0,
        f64::from_bits(524_288.0f64.to_bits() + 1),
        -524_288.75,
        1_048_576.125,
        -1_048_576.125,
        2_097_152.625,
        -2_097_152.625,
        4_503_599_627_370_495.5,
        -4_503_599_627_370_495.5,
        4_503_599_627_370_496.0,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let inputs: Vec<_> = values
        .iter()
        .map(|value| {
            let bits = value.to_bits();
            [bits as u32, (bits >> 32) as u32, 0, 0]
        })
        .collect();
    let operations = [(0x018, f64::ceil as fn(f64) -> f64), (0x01a, f64::floor)];
    for (op, expected) in operations {
        for source in [vgpr(1), sgpr(0)] {
            for (word, shift) in [(6, 0), (7, 32)] {
                let shader = build_valu_cs(&[vop1(op, 6, source), vop1(0x001, 0, vgpr(word))]);
                run_cases(&shader, &inputs, |backend, case, _, got| {
                    let want = (expected(values[case]).to_bits() >> shift) as u32;
                    assert_eq!(
                        got, want,
                        "{backend}: op {op:#x}, source {source}, case {case}, v{word}"
                    );
                });
            }
        }
    }
}

#[test]
fn directed_f64_vop3_negated_literal_rounds_in_both_directions() {
    for (opcode, expected) in [(0x19a, 0xc02c_0000), (0x198, 0xc02a_0000)] {
        let [op, sources] = vop3(opcode, 6, 255, 0, 0);
        let shader = build_valu_cs(&[op, sources | 1 << 29, 0x402b_0000, vop1(0x001, 0, vgpr(7))]);
        let container = crate::Container::parse(&shader.dxil).unwrap();
        let features = container.part(crate::FourCC::SFI0).unwrap();
        let bits = u64::from_le_bytes(features.data.try_into().unwrap());
        assert_eq!(bits & 0x8001, 0, "SFI0 doubles and int64 feature bits");
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: opcode {opcode:#x}, -13.5");
        });
    }
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

/// RDNA2 orders -0 below +0 in its maximums and minimums; DXIL's FMax and
/// FMin compare and select, which can give either zero.
#[test]
fn max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    // Each case has a -0 and a +0 among its first two operands.
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f32", 0x110, pz),
        ("v_min_f32", 0x10F, nz),
        ("v_max3_f32", 0x154, pz),
        ("v_min3_f32", 0x151, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    }
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

/// `v_cmp_lg_*` is "less or greater": ordered and not equal, so false when
/// either operand is a NaN (RDNA2 ISA; the same on gfx1036 hardware).
#[test]
fn v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    let nan = f32::NAN.to_bits();
    let (inf, ninf) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([inf, ninf, 0, 0], 1),
        ([inf, inf, 0, 0], 0),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([nan, nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0), // a signalling NaN
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x4000, 0x3C00, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x3C00, 0x7E00, 0, 0], 0),
        ([0x7E00, 0x7E00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),           // a signalling NaN
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0), // only the low halves are read
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// RDNA2's flt16_to_snorm16 / flt16_to_unorm16: clamp, scale by 32767 or
/// 65535, round the exact product to nearest even; NaN gives 0.
fn cvt_norm16_reference(bits: u16, signed: bool) -> u16 {
    let x = f16_to_f32(bits);
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    // f64 holds the product of an f16 and a 16-bit scale exactly.
    let p = f64::from(x.clamp(lo, 1.0)) * scale;
    p.round_ties_even() as i32 as u16
}

#[test]
fn v_cvt_norm_i16_and_u16_f16_round_the_exact_product() {
    // Every 251st f16 pattern, plus the values where an f32 product would
    // round before the rounding to an integer, ties, and the edges.
    let mut bits: Vec<u16> = (0..=u16::MAX).step_by(251).collect();
    bits.extend([
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3801, 0xB801, 0x37FF,
        0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01,
        0x3A9B, 0xBA9B, 0x2E66, 0x1400,
    ]);
    for (op, signed, what) in [
        (0x063u16, true, "v_cvt_norm_i16_f16"),
        (0x064, false, "v_cvt_norm_u16_f16"),
    ] {
        // v_cvt_norm_*16_f16 v0, v1 (VOP1); v0's high half starts at 0.
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = bits.iter().map(|&b| [u32::from(b), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = cvt_norm16_reference(bits[i], signed);
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                bits[i],
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_cvt_u16_and_i16_f16_saturate() {
    // v_mov_b32 v0, 0; v_cvt_u16_f16 / v_cvt_i16_f16 v0, v1 (VOP1 0x052 / 0x053).
    let cases: &[(u16, u16, u16)] = &[
        // (f16 input, u16 result, i16 result)
        (0x0000, 0, 0),          // 0
        (0x8000, 0, 0),          // -0
        (0x3E00, 1, 1),          // 1.5 -> 1 (toward zero)
        (0xBE00, 0, 0xFFFF),     // -1.5 -> 0 / -1
        (0x57D0, 125, 125),      // 125.0
        (0x7800, 32768, 32767),  // 32768 saturates i16
        (0x77FF, 32752, 32752),  // 32752, the largest f16 below 32768
        (0xF800, 0, 0x8000),     // -32768 fits i16
        (0xF801, 0, 0x8000),     // -32800 saturates i16
        (0x7BFF, 65504, 32767),  // max finite f16
        (0xFBFF, 0, 0x8000),     // -65504
        (0x7C00, 0xFFFF, 32767), // +inf
        (0xFC00, 0, 0x8000),     // -inf
        (0x7E00, 0, 0),          // NaN
        (0xFE00, 0, 0),          // -NaN
        (0x0001, 0, 0),          // smallest subnormal
    ];
    for (op, what, pick) in [
        (0x052u16, "v_cvt_u16_f16", 0usize),
        (0x053, "v_cvt_i16_f16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_i32 width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    run_cases(
        &v_signed,
        &[[0x8000_0000, 4, 32, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: v_bfe_i32 width 32 should mask to 0");
        },
    );

    // s_bfe_i32 width is 7-bit (src1[22:16]), so width=32 keeps all 32 bits.
    let mut s_cases = signed_cases.to_vec();
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    let s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn f16_omod_applies_before_clamp() {
    // VOP3 v_add_f16 (0x132 = 0x100 + VOP2 0x32): v0 = (v1 + v2) * omod [clamp]
    let mul2 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 1, false));
    let mul4 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 2, false));
    let div2 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, false));
    let div2_clamp = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, true));

    // f16 constants: 0.25=0x3400, 0.5=0x3800, 0.75=0x3A00, 1.0=0x3C00, 1.5=0x3E00, 2.0=0x4000, -0.5=0xB800
    run_cases(
        &mul2,
        &[[0x3400, 0x3800, 0, 0]], // (0.25 + 0.5) * 2 = 1.5 (0x3E00)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x3E00, "{backend}: v_add_f16 mul:2");
        },
    );
    run_cases(
        &mul4,
        &[[0x3400, 0x3400, 0, 0]], // (0.25 + 0.25) * 4 = 2.0 (0x4000)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x4000, "{backend}: v_add_f16 mul:4");
        },
    );
    run_cases(
        &div2,
        &[[0x3E00, 0x3E00, 0, 0]], // (1.5 + 1.5) / 2 = 1.5 (0x3E00)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x3E00, "{backend}: v_add_f16 div:2");
        },
    );
    // Ordering check: (0.75 + 0.75) = 1.5; div:2 before clamp gives 0.75 (0x3A00),
    // whereas clamp before div:2 would give 0.5 (0x3800).
    let clamp_cases: &[(u16, u16, u16)] = &[
        (0x3A00, 0x3A00, 0x3A00), // (0.75 + 0.75) / 2 = 0.75 -> 0.75
        (0x3E00, 0x3E00, 0x3C00), // (1.5 + 1.5) / 2 = 1.5 -> clamped to 1.0
        (0xB800, 0xB800, 0x0000), // (-0.5 + -0.5) / 2 = -0.5 -> clamped to 0.0
    ];
    let inputs: Vec<[u32; 4]> = clamp_cases
        .iter()
        .map(|&(a, b, _)| [u32::from(a), u32::from(b), 0, 0])
        .collect();
    run_cases(&div2_clamp, &inputs, |backend, i, _, got| {
        let want = clamp_cases[i].2;
        assert_eq!(
            got as u16, want,
            "{backend}: v_add_f16 div:2 clamp case {i}: got {:#06x}, want {want:#06x}",
            got as u16,
        );
    });
}

#[test]
fn cvt_16bit_int_from_f16_saturates_and_honors_modifiers() {
    // VOP3 v_cvt_u16_f16 = 0x1D2 (0x180 + VOP1 0x52)
    // VOP3 v_cvt_i16_f16 = 0x1D3 (0x180 + VOP1 0x53)
    let cvt_u16 = build_valu_cs(&vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    let cvt_u16_abs = build_valu_cs(&vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 1, 0, 0, false));
    let cvt_i16 = build_valu_cs(&vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    let cvt_i16_neg = build_valu_cs(&vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 1, 0, false));

    let u16_cases: &[(u16, u16)] = &[
        (0xBC00, 0),      // -1.0 -> 0
        (0xFC00, 0),      // -inf -> 0
        (0x7E00, 0),      // NaN -> 0
        (0x57BC, 123),    // 123.75 -> 123
        (0x7BFF, 65504),  // 65504.0 (max finite f16) -> 65504
        (0x7C00, 0xFFFF), // +inf -> 65535 (saturated)
    ];
    let u16_in: Vec<[u32; 4]> = u16_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cvt_u16, &u16_in, |backend, i, inp, got| {
        let want = u16_cases[i].1;
        assert_eq!(
            got as u16, want,
            "{backend}: v_cvt_u16_f16({:#06x}) = {:#06x}, want {want:#06x}",
            inp[0], got as u16,
        );
    });
    run_cases(
        &cvt_u16_abs,
        &[[0xD7BC, 0, 0, 0]], // |-123.75| -> 123
        |backend, _, _, got| {
            assert_eq!(got as u16, 123, "{backend}: v_cvt_u16_f16 |src0|");
        },
    );

    let i16_cases: &[(u16, u16)] = &[
        (0xFC00, 0x8000),           // -inf -> -32768
        (0xF8E2, 0x8000),           // -40000.0 -> -32768 (saturated, not wrapped)
        (0xD7BC, (-123i16) as u16), // -123.75 -> -123
        (0x7E00, 0),                // NaN -> 0
        (0x57BC, 123),              // 123.75 -> 123
        (0x78E2, 0x7FFF),           // +40000.0 -> 32767 (saturated, not wrapped)
        (0x7C00, 0x7FFF),           // +inf -> 32767
    ];
    let i16_in: Vec<[u32; 4]> = i16_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cvt_i16, &i16_in, |backend, i, inp, got| {
        let want = i16_cases[i].1;
        assert_eq!(
            got as u16, want,
            "{backend}: v_cvt_i16_f16({:#06x}) = {:#06x}, want {want:#06x}",
            inp[0], got as u16,
        );
    });
    run_cases(
        &cvt_i16_neg,
        &[[0x57BC, 0, 0, 0], [0x78E2, 0, 0, 0]], // -(123.75) -> -123; -(40000.0) -> -32768
        |backend, i, _, got| {
            let want = [(-123i16) as u16, 0x8000][i];
            assert_eq!(got as u16, want, "{backend}: v_cvt_i16_f16 -src0 case {i}");
        },
    );
}

const S_BARRIER: u32 = 0xBF8A_0000;

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// DS `op vdst, addr, data0` at offset 0.
const fn ds(op: u8, vdst: u8, addr: u8, data0: u8) -> [u32; 2] {
    [
        0x36 << 26 | (op as u32) << 18,
        (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32,
    ]
}

const DS_WRITE_B32: u8 = 0x0D;
const DS_MIN_F32: u8 = 0x12;
const DS_MAX_F32: u8 = 0x13;
const DS_MIN_RTN_F32: u8 = 0x32;
const DS_MAX_RTN_F32: u8 = 0x33;
const DS_READ_B32: u8 = 0x36;

/// What RDNA2 writes for `ds_max_f32` / `ds_min_f32` over a dword holding
/// `old` with `data` (bits), as measured on gfx1036 with f32 denormals kept:
/// an sNaN, data first, then memory, is written quieted; otherwise a NaN
/// loses to a number (two quiet NaNs give the data); numbers compare as
/// floats with -0 below +0.
fn lds_min_max_ref(old: u32, data: u32, max: bool) -> u32 {
    let nan = |x: u32| x & 0x7FFF_FFFF > 0x7F80_0000;
    let snan = |x: u32| nan(x) && x & 0x40_0000 == 0;
    let key = |x: u32| {
        let b = x as i32;
        if b < 0 { b ^ 0x7FFF_FFFF } else { b }
    };
    if snan(data) {
        return data | 0x40_0000;
    }
    if snan(old) {
        return old | 0x40_0000;
    }
    match (nan(data), nan(old)) {
        (true, true) | (false, true) => return data,
        (true, false) => return old,
        (false, false) => {}
    }
    let take = if max {
        key(data) > key(old)
    } else {
        key(data) < key(old)
    };
    if take { data } else { old }
}

#[test]
fn ds_min_max_f32_order_floats_and_return_the_old_value() {
    if !hardware_supports_wave_size(32) {
        return;
    }
    let values = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        2.5,
        -2.5,
        1e-30,
        -1e-30,
        f32::from_bits(1), // denormal
        f32::from_bits(0x8000_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
    ];
    // Quiet and signalling NaNs of both signs, in memory and as data.
    let nans = [
        0x7FC0_0000u32,
        0xFFC0_0000,
        0x7F80_0001,
        0xFFA0_0000,
        0x7FBF_FFFF,
    ];
    let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).chain(nans).collect();
    let mut cases: Vec<[u32; 4]> = Vec::new();
    for &old in &bits {
        for &data in &bits {
            cases.push([old, data, 0, 0]);
        }
    }
    for (rtn, plain, max, what) in [
        (DS_MAX_RTN_F32, DS_MAX_F32, true, "ds_max"),
        (DS_MIN_RTN_F32, DS_MIN_F32, false, "ds_min"),
    ] {
        // v5 = 0; LDS[v5] = v1; the atomic with v2; v0 = the old value
        // (rtn) or the dword afterwards (read back).
        let mut returns_old = vec![vop1(0x001, 5, 0x80)];
        returns_old.extend(ds(DS_WRITE_B32, 0, 5, 1));
        returns_old.extend(ds(rtn, 0, 5, 2));
        let mut returns_new = vec![vop1(0x001, 5, 0x80)];
        returns_new.extend(ds(DS_WRITE_B32, 0, 5, 1));
        returns_new.extend(ds(plain, 0, 5, 2));
        returns_new.extend(ds(DS_READ_B32, 0, 5, 0));
        for (body, old_result) in [(returns_old, true), (returns_new, false)] {
            // One thread with LDS declares WaveSize(32), which WARP's
            // pipeline creation refuses: hardware only (the 64-thread test
            // below runs on WARP too).
            let shader = build_cs(&body, 1, 512);
            run_cases_on(
                &shader,
                &cases,
                &[("hardware", false)],
                |backend, _, inp, got| {
                    let (old, data) = (f32::from_bits(inp[0]), f32::from_bits(inp[1]));
                    let want = if old_result {
                        inp[0]
                    } else {
                        lds_min_max_ref(inp[0], inp[1], max)
                    };
                    assert_eq!(
                        got,
                        want,
                        "{backend}: {what}_f32 on {old:e} [{:#010x}] with {data:e} [{:#010x}] gave {got:#010x}, want {want:#010x} ({})",
                        inp[0],
                        inp[1],
                        if old_result {
                            "old value"
                        } else {
                            "dword after"
                        },
                    );
                },
            );
        }
    }
}

#[test]
fn ds_min_max_f32_are_atomic_across_waves() {
    // 64 threads (two host waves of 32 or one of 64): LDS[0] = v1, then
    // every thread applies (thread - 32) as a float; v0 = LDS[0].
    for (op, max, init, want) in [
        (DS_MAX_F32, true, f32::NEG_INFINITY, 31.0f32),
        (DS_MAX_F32, true, 100.0, 100.0),
        (DS_MIN_F32, false, f32::INFINITY, -32.0),
        (DS_MIN_F32, false, -0.0, -32.0),
    ] {
        let mut body = vec![vop1(0x001, 5, 0x80)];
        body.extend(ds(DS_WRITE_B32, 0, 5, 1));
        body.push(S_BARRIER);
        body.push(vop2(0x26, 6, vgpr(0), 3)); // v_sub_nc_u32 v6, v0, v3
        body.push(vop1(0x005, 6, vgpr(6))); // v_cvt_f32_i32
        body.extend(ds(op, 0, 5, 6));
        body.push(S_BARRIER);
        body.extend(ds(DS_READ_B32, 0, 5, 0));
        let shader = build_cs(&body, 64, 512);
        run_cases(
            &shader,
            &[[init.to_bits(), 0, 32, 0]],
            |backend, _, _, got| {
                assert_eq!(
                    f32::from_bits(got),
                    want,
                    "{backend}: {} from {init} over 64 threads",
                    if max { "max" } else { "min" },
                );
            },
        );
    }
}

#[test]
fn v_cmp_class_f16_matches_rdna2() {
    // v_cmp_class_f16_e64 s6, v1, v2 (VOP3 opcode 0x08f); v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp_lo = build_valu_cs(&[
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);
    // Same with op_sel:[1,0] (high half of v1) + neg on src0 (bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x08f, 6, vgpr(1), vgpr(2), 0);
    w0 |= 1 << 11; // opsel[0] = 1 (high half)
    w1 |= 1 << 29; // neg[0] = 1
    let cmp_hi_neg = build_valu_cs(&[
        w0,
        w1,
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary16 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7C01, 0), // sNaN (+sNaN, quiet bit 0x0200 clear)
        (0xFDFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7E00, 1), // +qNaN
        (0xFE01, 1), // -qNaN
        (0xFC00, 2), // -inf
        (0xBC00, 3), // -normal (-1.0)
        (0x8001, 4), // -denormal
        (0x8000, 5), // -0.0
        (0x0000, 6), // +0.0
        (0x0001, 7), // +denormal
        (0x3C00, 8), // +normal (+1.0)
        (0x7C00, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_lo, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16({:#06x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // High half with neg: high half 0x3C00 (+1.0) negated -> -normal (class 3);
    // high half 0x7C01 (+sNaN) negated -> still sNaN (class 0, not quieted).
    let hi_cases: &[([u32; 4], u32)] = &[
        ([0x3C00_0000, 1 << 3, 0, 0], 1),
        ([0x3C00_0000, 1 << 8, 0, 0], 0),
        ([0x7C01_FFFF, 1 << 0, 0, 0], 1),
        ([0x7C01_FFFF, 1 << 1, 0, 0], 0),
    ];
    let hi_in: Vec<[u32; 4]> = hi_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_hi_neg, &hi_in, |backend, i, inp, got| {
        let want = hi_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16(-hi({:#010x}), {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn v_cvt_f32_f64_matches_ieee754() {
    // v_cvt_f32_f64 v0, v[1:2] (VOP1 opcode 0x00f -> VOP3 opcode 0x18f).
    let plain = build_valu_cs(&[vop1(0x00f, 0, vgpr(1))]);
    // Same with abs + neg on src0 (VOP3: abs[0] = bit 8 of w0, neg[0] = bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x18f, 0, vgpr(1), 0, 0);
    w0 |= 1 << 8;
    w1 |= 1 << 29;
    let neg_abs = build_valu_cs(&[w0, w1]);

    let values: &[f64] = &[
        0.0,
        -0.0,
        1.0,
        -1.0,
        3.5,
        -123456.75,
        // Tie-to-even: even mantissa stays, odd mantissa rounds up.
        f64::from_bits(0x3FF0_0000_1000_0000), // 1.0 + 0.5 ulp(f32) -> 1.0
        f64::from_bits(0x3FF0_0000_3000_0000), // (1.0 + 1 ulp) + 0.5 ulp -> 1.0 + 2 ulp
        f64::from_bits(0x3FF0_0000_1000_0001), // > 0.5 ulp -> rounds up
        // Largest finite f32 and overflow to +Inf / -Inf.
        f64::from(f32::MAX),
        f64::from(f32::MAX) * 1.000_001,
        -f64::from(f32::MAX) * 1.000_001,
        1e300,
        // Subnormal range (exp64 in [874, 896]), including carry into smallest normal.
        f64::from(f32::MIN_POSITIVE),
        f64::from(f32::MIN_POSITIVE) * 0.5,
        f64::from(f32::MIN_POSITIVE) * 0.75,
        f64::from(f32::from_bits(0x007F_FFFF)),
        f64::from_bits(0x380F_FFFF_F000_0000), // max subnormal + 0.5 ulp -> min normal
        // exp64 == 873: exact 2^-150 ties to 0; strictly greater rounds to min subnormal (1).
        f64::from_bits(0x3690_0000_0000_0000),
        f64::from_bits(0x3690_0000_0000_0001),
        f64::from_bits(0xB690_0000_0000_0001),
        // Deep underflow -> +/-0.0.
        1e-200,
        -1e-200,
        // Infinities and NaNs.
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7FF8_0000_0000_0000), // +qNaN
        f64::from_bits(0x7FF0_0000_0000_0001), // +sNaN -> quieted
    ];
    let inputs: Vec<[u32; 4]> = values
        .iter()
        .map(|&x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&plain, &inputs, |backend, i, _, got| {
        let x = values[i];
        let want = (x as f32).to_bits();
        if x.is_nan() {
            assert!(
                f32::from_bits(got).is_nan() && (got & 0x0040_0000) != 0,
                "{backend}: v_cvt_f32_f64({:#018x}) = {got:#010x}, expected qNaN",
                x.to_bits(),
            );
        } else {
            assert_eq!(
                got,
                want,
                "{backend}: v_cvt_f32_f64({x} / {:#018x}) = {got:#010x}, want {want:#010x}",
                x.to_bits(),
            );
        }
    });

    // -|x| via VOP3 abs + neg modifiers.
    let mod_values: &[f64] = &[2.5, -2.5, 0.0, f64::INFINITY];
    let mod_in: Vec<[u32; 4]> = mod_values
        .iter()
        .map(|&x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&neg_abs, &mod_in, |backend, i, _, got| {
        let x = mod_values[i];
        let want = (-(x.abs()) as f32).to_bits();
        assert_eq!(
            got, want,
            "{backend}: v_cvt_f32_f64(-|{x}|) = {got:#010x}, want {want:#010x}",
        );
    });
}

#[test]
fn whole_wave_f64_sum_and_cvt_f32_f64() {
    if !hardware_supports_wave_size(32) {
        return;
    }
    let s_or_saveexec_b32 =
        |sdst: u8, src0: u8| 0xBE80_0000 | (sdst as u32) << 16 | 0x3D << 8 | src0 as u32;
    let v_mov_dpp8 =
        |vdst: u8, vsrc: u8, sel: u32| [vop1(0x001, vdst, 0xE9), vsrc as u32 | (sel << 8)];
    let v_mov_row_mirror = |vdst: u8, vsrc: u8| {
        [
            vop1(0x001, vdst, 0xFA),
            vsrc as u32 | (0x140 << 8) | (1 << 19) | (0xF << 24) | (0xF << 28),
        ]
    };

    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80));
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Per-lane f64 input in v[6:7] = (s0, s1).
    words.push(vop1(0x001, 6, sgpr(0)));
    words.push(vop1(0x001, 7, sgpr(1)));
    // Whole-wave f64 sum over Wave32 with s_mov_b32 copy before EXEC restore.
    words.push(s_or_saveexec_b32(12, 0xC1));
    words.extend_from_slice(&vop3(0x101, 2, 0x80, vgpr(6), sgpr(12)));
    words.extend_from_slice(&vop3(0x101, 3, 0x80, vgpr(7), sgpr(12)));
    for sel in [14570689u32, 9658386, 2340] {
        words.extend_from_slice(&v_mov_dpp8(4, 2, sel));
        words.extend_from_slice(&v_mov_dpp8(5, 3, sel));
        words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0)); // v_add_f64
    }
    words.extend_from_slice(&v_mov_row_mirror(4, 2));
    words.extend_from_slice(&v_mov_row_mirror(5, 3));
    words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    words.extend_from_slice(&vop3(0x378, 4, vgpr(2), 0x80, 0x80)); // v_permlanex16_b32
    words.extend_from_slice(&vop3(0x378, 5, vgpr(3), 0x80, 0x80));
    words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    words.extend_from_slice(&vop3(0x360, 14, vgpr(2), 0x80, 0)); // v_readlane_b32 s14, v2, 0
    words.extend_from_slice(&vop3(0x360, 15, vgpr(3), 0x80, 0)); // v_readlane_b32 s15, v3, 0
    words.push(s_mov_b32(16, 14));
    words.push(s_mov_b32(17, 15));
    words.push(s_mov_b32(126, 12)); // restore exec_lo
    words.push(vop1(0x00f, 0, sgpr(16))); // v_cvt_f32_f64 v0, s[16:17]
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 12,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [4, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        emulate_mesh: false,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: Vec::new(),
        mesh_emu: None,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }

    let cases: &[(f64, f32)] = &[(1.25, 5.0), (-0.75, -3.0)];
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            // WARP does not support fixed WaveSize(32) on SM 6.6.
            Err(e) if warp => {
                eprintln!("{label}: compute_pipeline skipped ({e})");
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        for &(x, want) in cases {
            let b = x.to_bits();
            let inputs = [b as u32, (b >> 32) as u32, 0, 0];
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            assert_eq!(
                out[0][0],
                want.to_bits(),
                "{label}: wave f64 sum({x}) = {:#010x}, want {:#010x}",
                out[0][0],
                want.to_bits(),
            );
        }
    }
}

/// SOPK `op sdst, simm16`.
const fn sopk(op: u8, sdst: u8, simm16: i16) -> u32 {
    0xB000_0000 | (op as u32 & 0x1F) << 23 | (sdst as u32 & 0x7F) << 16 | (simm16 as u16) as u32
}

#[test]
fn signed_scalar_add_sub_scc_matches_rdna2() {
    // s_add_i32 s6, s0, s1 (SOP2 0x02); s_cselect_b32 s7, 1, 0 (SOP2 0x0A); v_mov_b32 v0, s7
    let add_scc = build_valu_cs(&[
        sop2(0x02, 6, 0, 1),
        sop2(0x0A, 7, 0x81, 0x80),
        vop1(0x001, 0, sgpr(7)),
    ]);
    // s_sub_i32 s6, s0, s1 (SOP2 0x03); s_cselect_b32 s7, 1, 0; v_mov_b32 v0, s7
    let sub_scc = build_valu_cs(&[
        sop2(0x03, 6, 0, 1),
        sop2(0x0A, 7, 0x81, 0x80),
        vop1(0x001, 0, sgpr(7)),
    ]);
    // s_mov_b32 s6, s0; s_addk_i32 s6, +1 (SOPK 0x0F); s_cselect_b32 s7, 1, 0; v_mov_b32 v0, s7
    let addk_pos1 = build_valu_cs(&[
        s_mov_b32(6, 0),
        sopk(0x0F, 6, 1),
        sop2(0x0A, 7, 0x81, 0x80),
        vop1(0x001, 0, sgpr(7)),
    ]);
    // s_mov_b32 s6, s0; s_addk_i32 s6, -32768 (SOPK 0x0F); s_cselect_b32 s7, 1, 0; v_mov_b32 v0, s7
    let addk_neg_min = build_valu_cs(&[
        s_mov_b32(6, 0),
        sopk(0x0F, 6, i16::MIN),
        sop2(0x0A, 7, 0x81, 0x80),
        vop1(0x001, 0, sgpr(7)),
    ]);

    let add_cases: &[(i32, i32)] = &[
        (0, 0),
        (10, 20),
        (-1, 1), // unsigned carry, no signed overflow
        (i32::MAX, 0),
        (i32::MAX, 1),        // positive overflow
        (i32::MAX, i32::MAX), // positive overflow
        (i32::MIN, 0),
        (i32::MIN, -1),       // negative overflow
        (i32::MIN, i32::MIN), // negative overflow (result 0)
        (i32::MIN, i32::MAX), // -1, no overflow
    ];
    let add_in: Vec<[u32; 4]> = add_cases
        .iter()
        .map(|&(a, b)| [a as u32, b as u32, 0, 0])
        .collect();
    run_cases(&add_scc, &add_in, |backend, i, _, got| {
        let (a, b) = add_cases[i];
        let want = u32::from(a.overflowing_add(b).1);
        assert_eq!(
            got, want,
            "{backend}: s_add_i32({a}, {b}) SCC = {got}, want {want}",
        );
    });

    let sub_cases: &[(i32, i32)] = &[
        (0, 0),
        (0, 1),        // unsigned borrow, no signed overflow
        (0, i32::MIN), // 0 - (-2^31) overflows i32
        (i32::MIN, 0),
        (i32::MIN, 1),        // negative overflow
        (i32::MIN, i32::MIN), // 0, no overflow
        (i32::MAX, -1),       // positive overflow
        (i32::MAX, i32::MIN), // positive overflow
        (-5, -3),             // -2, no overflow
        (-1, i32::MAX),       // -2^31 (i32::MIN), no overflow
    ];
    let sub_in: Vec<[u32; 4]> = sub_cases
        .iter()
        .map(|&(a, b)| [a as u32, b as u32, 0, 0])
        .collect();
    run_cases(&sub_scc, &sub_in, |backend, i, _, got| {
        let (a, b) = sub_cases[i];
        let want = u32::from(a.overflowing_sub(b).1);
        assert_eq!(
            got, want,
            "{backend}: s_sub_i32({a}, {b}) SCC = {got}, want {want}",
        );
    });

    let addk_pos_cases: &[i32] = &[0, -1, i32::MAX - 1, i32::MAX, i32::MIN];
    let addk_pos_in: Vec<[u32; 4]> = addk_pos_cases
        .iter()
        .map(|&a| [a as u32, 0, 0, 0])
        .collect();
    run_cases(&addk_pos1, &addk_pos_in, |backend, i, _, got| {
        let a = addk_pos_cases[i];
        let want = u32::from(a.overflowing_add(1).1);
        assert_eq!(
            got, want,
            "{backend}: s_addk_i32({a}, +1) SCC = {got}, want {want}",
        );
    });

    let addk_neg_cases: &[i32] = &[0, i32::MAX, i32::MIN + 32768, i32::MIN + 32767, i32::MIN];
    let addk_neg_in: Vec<[u32; 4]> = addk_neg_cases
        .iter()
        .map(|&a| [a as u32, 0, 0, 0])
        .collect();
    run_cases(&addk_neg_min, &addk_neg_in, |backend, i, _, got| {
        let a = addk_neg_cases[i];
        let want = u32::from(a.overflowing_add(i32::from(i16::MIN)).1);
        assert_eq!(
            got, want,
            "{backend}: s_addk_i32({a}, -32768) SCC = {got}, want {want}",
        );
    });
}

/// VOP2 `op vdst, src0, vsrc1`.
#[test]
fn v_pk_fmac_f16_matches_rdna2() {
    // VOP2 `v_pk_fmac_f16 v0, v1, v2` (opcode 0x3c), initialized from v3 (root constant s2).
    let vop2_cs = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)),          // v_mov_b32 v0, v3 (accumulator)
        vop2(0x3c, 0, vgpr(1), 2),        // v_pk_fmac_f16 v0, v1, v2
    ]);
    // VOP2 with inline constant 1.0 (0xF2): GFX10 puts 1.0 in lo f16 and 0.0 in hi f16.
    let vop2_inline_cs = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)),          // v_mov_b32 v0, v3
        vop2(0x3c, 0, 0xF2, 2),           // v_pk_fmac_f16 v0, 1.0, v2
    ]);
    // VOP3 `v_pk_fmac_f16 v0, |v1|, -v2 clamp` (opcode 0x13c): abs[0], neg[1], clamp.
    let [mut w0, mut w1] = vop3(0x13c, 0, vgpr(1), vgpr(2), 0);
    w0 |= (1 << 8) | (1 << 15); // abs[0], clamp
    w1 |= 1 << 30;              // neg[1]
    let vop3_mod_cs = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)), // v_mov_b32 v0, v3
        w0,
        w1,
    ]);

    let pack = |lo: u16, hi: u16| u32::from(lo) | (u32::from(hi) << 16);
    // f16 constants: 0.0=0x0000, 0.5=0x3800, 1.0=0x3C00, 2.0=0x4000, 3.0=0x4200,
    // 4.0=0x4400, 5.0=0x4500, 7.0=0x4700, 10.0=0x4900, -2.0=0xC000, -3.0=0xC200.
    let vop2_cases: &[([u32; 4], u32)] = &[
        // lo: 2.0 * 3.0 + 1.0 = 7.0; hi: -2.0 * 4.0 + 10.0 = 2.0
        (
            [pack(0x4000, 0xC000), pack(0x4200, 0x4400), pack(0x3C00, 0x4900), 0],
            pack(0x4700, 0x4000),
        ),
        // lo: 0.5 * 4.0 + 3.0 = 5.0; hi: 0.0 * 7.0 + (-3.0) = -3.0
        (
            [pack(0x3800, 0x0000), pack(0x4400, 0x4700), pack(0x4200, 0xC200), 0],
            pack(0x4500, 0xC200),
        ),
    ];
    let vop2_in: Vec<[u32; 4]> = vop2_cases.iter().map(|c| c.0).collect();
    run_cases(&vop2_cs, &vop2_in, |backend, i, inp, got| {
        let want = vop2_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_pk_fmac_f16({:#010x}, {:#010x}, acc={:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });

    // Inline 1.0: lo = 1.0 * 3.0 + 2.0 = 5.0; hi = 0.0 * 4.0 + 7.0 = 7.0 (unchanged).
    run_cases(
        &vop2_inline_cs,
        &[[0, pack(0x4200, 0x4400), pack(0x4000, 0x4700), 0]],
        |backend, _, _, got| {
            let want = pack(0x4500, 0x4700);
            assert_eq!(
                got, want,
                "{backend}: v_pk_fmac_f16(1.0, ...) = {got:#010x}, want {want:#010x}",
            );
        },
    );

    // VOP3 abs[0], neg[1], clamp:
    // lo: |-2.0| * (-0.5) + 1.5(=0x3E00) = 0.5 (in [0, 1]) -> 0x3800
    // hi: |-2.0| * (-2.0) + 1.0 = -3.0 -> clamped to 0.0 (0x0000)
    // case 2: lo: |-2.0| * (-(-1.0)) + 0.5 = 2.5 -> clamped to 1.0 (0x3C00)
    //         hi: |0.5| * (-(-0.5)) + 0.25(=0x3400) = 0.5 (0x3800)
    let vop3_cases: &[([u32; 4], u32)] = &[
        (
            [pack(0xC000, 0xC000), pack(0x3800, 0x4000), pack(0x3E00, 0x3C00), 0],
            pack(0x3800, 0x0000),
        ),
        (
            [pack(0xC000, 0x3800), pack(0xBC00, 0xB800), pack(0x3800, 0x3400), 0],
            pack(0x3C00, 0x3800),
        ),
    ];
    let vop3_in: Vec<[u32; 4]> = vop3_cases.iter().map(|c| c.0).collect();
    run_cases(&vop3_mod_cs, &vop3_in, |backend, i, inp, got| {
        let want = vop3_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_pk_fmac_f16_e64(|{:#010x}|, -{:#010x}, acc={:#010x}) clamp = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_frexp_f32_and_f16_match_rdna2() {
    // VOP1 opcodes: 0x3f=v_frexp_exp_i32_f32, 0x40=v_frexp_mant_f32,
    //               0x59=v_frexp_mant_f16,    0x5a=v_frexp_exp_i16_f16.
    let exp_f32 = build_valu_cs(&[vop1(0x3f, 0, vgpr(1))]);
    let mant_f32 = build_valu_cs(&[vop1(0x40, 0, vgpr(1))]);
    // For f16 tests, initialize v0 = 0xDEAD_0000 so we also verify high-half preservation.
    let mant_f16 = build_valu_cs(&[
        vop1(0x001, 0, vgpr(2)), // v_mov_b32 v0, v2 (sentinel upper half)
        vop1(0x59, 0, vgpr(1)),
    ]);
    let exp_f16 = build_valu_cs(&[
        vop1(0x001, 0, vgpr(2)), // v_mov_b32 v0, v2 (sentinel upper half)
        vop1(0x5a, 0, vgpr(1)),
    ]);

    // (input f32 bits, expected mantissa f32 bits, expected i32 exponent)
    let f32_cases: &[(u32, u32, i32)] = &[
        (0.0f32.to_bits(), 0.0f32.to_bits(), 0),
        ((-0.0f32).to_bits(), (-0.0f32).to_bits(), 0),
        (0.5f32.to_bits(), 0.5f32.to_bits(), 0),
        (1.0f32.to_bits(), 0.5f32.to_bits(), 1),
        ((-6.0f32).to_bits(), (-0.75f32).to_bits(), 3),
        (0.125f32.to_bits(), 0.5f32.to_bits(), -2),
        (f32::INFINITY.to_bits(), f32::INFINITY.to_bits(), 0),
        (f32::NEG_INFINITY.to_bits(), f32::NEG_INFINITY.to_bits(), 0),
        // Subnormals: 2^-127 = 0.5 * 2^-126; 3 * 2^-149 = 0.75 * 2^-147; -2^-149 = -0.5 * 2^-148
        (0x0040_0000, 0.5f32.to_bits(), -126),
        (0x0000_0003, 0.75f32.to_bits(), -147),
        (0x8000_0001, (-0.5f32).to_bits(), -148),
    ];
    let f32_in: Vec<[u32; 4]> = f32_cases.iter().map(|&(x, _, _)| [x, 0, 0, 0]).collect();
    run_cases(&mant_f32, &f32_in, |backend, i, inp, got| {
        let want = f32_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_frexp_mant_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    run_cases(&exp_f32, &f32_in, |backend, i, inp, got| {
        let want = f32_cases[i].2 as u32;
        assert_eq!(
            got, want,
            "{backend}: v_frexp_exp_i32_f32({:#010x}) = {} ({got:#010x}), want {}",
            inp[0], got as i32, f32_cases[i].2,
        );
    });

    // NaN: mantissa stays NaN, exponent is 0.
    let qnan = 0x7FC0_0000u32;
    run_cases(&mant_f32, &[[qnan, 0, 0, 0]], |backend, _, _, got| {
        assert!(
            f32::from_bits(got).is_nan(),
            "{backend}: v_frexp_mant_f32(NaN) = {got:#010x}, want NaN",
        );
    });
    run_cases(&exp_f32, &[[qnan, 0, 0, 0]], |backend, _, _, got| {
        assert_eq!(got, 0, "{backend}: v_frexp_exp_i32_f32(NaN) = {got}, want 0");
    });

    // f16 cases: (input f16, expected mantissa f16, expected i16 exponent)
    // 0x0000=+0, 0x8000=-0, 0x3800=0.5, 0x3C00=1.0, 0xC600=-6.0 (-0.75 * 2^3, -0.75=0xBA00),
    // 0x7BFF=65504 (max f16 -> mant=0x3BFF, exp=16), 0x0001=2^-24 (min subnormal -> 0.5 * 2^-23),
    // 0x8003=-3*2^-24 (-0.75 * 2^-22), 0x7C00=+inf, 0xFC00=-inf.
    let sentinel = 0xDEAD_0000u32;
    let f16_cases: &[(u16, u16, i16)] = &[
        (0x0000, 0x0000, 0),
        (0x8000, 0x8000, 0),
        (0x3800, 0x3800, 0),
        (0x3C00, 0x3800, 1),
        (0xC600, 0xBA00, 3),
        (0x7BFF, 0x3BFF, 16),
        (0x0001, 0x3800, -23),
        (0x8003, 0xBA00, -22),
        (0x7C00, 0x7C00, 0),
        (0xFC00, 0xFC00, 0),
    ];
    let f16_in: Vec<[u32; 4]> = f16_cases
        .iter()
        .map(|&(x, _, _)| [u32::from(x), sentinel, 0, 0])
        .collect();
    run_cases(&mant_f16, &f16_in, |backend, i, inp, got| {
        let want = sentinel | u32::from(f16_cases[i].1);
        assert_eq!(
            got, want,
            "{backend}: v_frexp_mant_f16({:#06x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    run_cases(&exp_f16, &f16_in, |backend, i, inp, got| {
        let want = sentinel | u32::from(f16_cases[i].2 as u16);
        assert_eq!(
            got, want,
            "{backend}: v_frexp_exp_i16_f16({:#06x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });

    // f16 NaN: mantissa is f16 NaN (upper half preserved), exponent is 0.
    run_cases(&mant_f16, &[[0x7E00, sentinel, 0, 0]], |backend, _, _, got| {
        assert_eq!(got & 0xFFFF_0000, sentinel);
        let lo = (got & 0x7FFF) as u16;
        assert!(lo > 0x7C00, "{backend}: v_frexp_mant_f16(NaN) lo={lo:#06x}, want NaN");
    });
    run_cases(&exp_f16, &[[0x7E00, sentinel, 0, 0]], |backend, _, _, got| {
        assert_eq!(got, sentinel, "{backend}: v_frexp_exp_i16_f16(NaN) = {got:#010x}, want 0");
    });
}

/// VOP2 `op vdst, src0, vsrc1`.
#[test]
fn v_ldexp_f16_matches_rdna2() {
    // VOP2 `v_ldexp_f16 v0, v1, v2` (opcode 0x3b), preserving upper half from v3.
    let vop2_cs = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)),   // v_mov_b32 v0, v3 (sentinel upper half)
        vop2(0x3b, 0, vgpr(1), 2), // v_ldexp_f16 v0, v1, v2
    ]);
    // VOP3 `v_ldexp_f16 v0, |v1|, v2 clamp` (opcode 0x13b) with op_sel:[1, 0, 0, 1]
    // (reads high half of v1, low half of v2, writes high half of v0).
    let [mut w0, w1] = vop3(0x13b, 0, vgpr(1), vgpr(2), 0);
    w0 |= (1 << 8) | (1 << 11) | (1 << 14) | (1 << 15); // abs[0], opsel[0], opsel[3], clamp
    let vop3_cs = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)), // v_mov_b32 v0, v3 (sentinel lower half)
        w0,
        w1,
    ]);

    let sentinel = 0xCAFE_0000u32;
    // (f16 input, i16 exponent, expected f16 output)
    let cases: &[(u16, i16, u16)] = &[
        (0x3C00, 0, 0x3C00),         // 1.0 * 2^0 = 1.0
        (0x3E00, 3, 0x4A00),         // 1.5 * 2^3 = 12.0
        (0xC200, -2, 0xBA00),        // -3.0 * 2^-2 = -0.75
        (0x3C00, -15, 0x0200),       // 1.0 * 2^-15 = f16 subnormal 0x0200
        (0x0001, 24, 0x3C00),        // min f16 subnormal (2^-24) * 2^24 = 1.0
        (0x7BFF, 0, 0x7BFF),         // max finite f16 * 2^0 stays 65504 (no premature inf)
        (0x3C00, 16, 0x7C00),        // 1.0 * 2^16 = 65536 -> +inf (RTE overflow)
        (0xBC00, 1000, 0xFC00),      // -1.0 * 2^1000 -> -inf
        (0x0001, i16::MAX, 0x7C00),  // min subnormal * 2^32767 -> +inf
        (0x7BFF, i16::MIN, 0x0000),  // max finite * 2^-32768 -> +0.0
        (0xFBFF, i16::MIN, 0x8000),  // -max finite * 2^-32768 -> -0.0
        (0x0000, i16::MAX, 0x0000),  // +0.0 * 2^32767 = +0.0 (not NaN)
        (0x8000, i16::MAX, 0x8000),  // -0.0 * 2^32767 = -0.0 (not NaN)
        (0x7C00, i16::MIN, 0x7C00),  // +inf * 2^-32768 = +inf (not NaN)
        (0xFC00, i16::MIN, 0xFC00),  // -inf * 2^-32768 = -inf (not NaN)
    ];
    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(a, e, _)| [u32::from(a), u32::from(e as u16), sentinel, 0])
        .collect();
    run_cases(&vop2_cs, &inputs, |backend, i, _, got| {
        let (a, e, want_h) = cases[i];
        let want = sentinel | u32::from(want_h);
        assert_eq!(
            got, want,
            "{backend}: v_ldexp_f16({a:#06x}, {e}) = {got:#010x}, want {want:#010x}",
        );
    });

    // NaN stays NaN (upper half preserved).
    run_cases(&vop2_cs, &[[0x7E00, 4, sentinel, 0]], |backend, _, _, got| {
        assert_eq!(got & 0xFFFF_0000, sentinel);
        let lo = (got & 0x7FFF) as u16;
        assert!(lo > 0x7C00, "{backend}: v_ldexp_f16(NaN, 4) lo={lo:#06x}, want NaN");
    });

    // VOP3 abs[0] + opsel:[1,0,0,1] + clamp:
    // |-0.25(=0xB400)| * 2^1 = 0.5 (0x3800) into high half; |-1.0(=0xBC00)| * 2^2 = 4.0 -> clamped 1.0 (0x3C00)
    let lo_sentinel = 0x0000_1234u32;
    let vop3_cases: &[([u32; 4], u32)] = &[
        ([0xB400_0000, 1, lo_sentinel, 0], (0x3800u32 << 16) | lo_sentinel),
        ([0xBC00_0000, 2, lo_sentinel, 0], (0x3C00u32 << 16) | lo_sentinel),
    ];
    let vop3_in: Vec<[u32; 4]> = vop3_cases.iter().map(|c| c.0).collect();
    run_cases(&vop3_cs, &vop3_in, |backend, i, _, got| {
        let want = vop3_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_ldexp_f16_e64 case {i} = {got:#010x}, want {want:#010x}",
        );
    });
}

/// VOP2 `op vdst, src0, vsrc1` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
/// VOP3P `op vdst, src0, src1, src2` with its packed modifiers: `op_sel` and
/// `op_sel_hi` pick each source's half for the low and high lanes of the
/// operation (bit k for source k), `neg_lo`/`neg_hi` negate them.
#[allow(clippy::too_many_arguments)]
const fn vop3p_mods(
    op: u8,
    vdst: u8,
    [src0, src1, src2]: [u16; 3],
    op_sel: u8,
    op_sel_hi: u8,
    neg_lo: u8,
    neg_hi: u8,
    clamp: bool,
) -> [u32; 2] {
    [
        0x33 << 26
            | (op as u32 & 0x7F) << 16
            | (clamp as u32) << 15
            | (op_sel_hi as u32 >> 2 & 1) << 14
            | (op_sel as u32 & 7) << 11
            | (neg_hi as u32 & 7) << 8
            | vdst as u32,
        (src0 as u32 & 0x1FF)
            | (src1 as u32 & 0x1FF) << 9
            | (src2 as u32 & 0x1FF) << 18
            | (op_sel_hi as u32 & 3) << 27
            | (neg_lo as u32 & 7) << 29,
    ]
}

/// Inputs `[src0, src1, src2, _]` and the f32 bits RDNA2 gives for
/// `src0.f16[0] * src1.f16[0] + src0.f16[1] * src1.f16[1] + src2.f32`: f16
/// products are exact in f32, so 65504 * 65504 does not overflow as it would
/// in f16.
const DOT2_F32_F16_CASES: &[([u32; 4], u32)] = &[
    // (1, 2) . (3, 4) + 0.5 = 11.5
    ([0x4000_3c00, 0x4400_4200, 0x3f00_0000, 0], 0x4138_0000),
    // (-1.5, 0.25) . (2, -8) - 1 = -6
    ([0x3400_be00, 0xc800_4000, 0xbf80_0000, 0], 0xc0c0_0000),
    // (65504, 65504) . (65504, 0) + 0 = 4290774016
    ([0x7bff_7bff, 0x0000_7bff, 0, 0], 0x4f7f_c004),
    // (inf, 0) . (1, 0) + 0 = inf
    ([0x0000_7c00, 0x0000_3c00, 0, 0], 0x7f80_0000),
];

#[test]
fn v_dot2_f32_f16_sums_f16_products_in_f32() {
    // v0 = v_dot2_f32_f16 v1, v2, v3 (no modifiers: op_sel_hi = 0b111)
    let body = vop3p_mods(0x13, 0, [vgpr(1), vgpr(2), vgpr(3)], 0, 0b111, 0, 0, false);
    let shader = build_valu_cs(&body);
    let inputs: Vec<[u32; 4]> = DOT2_F32_F16_CASES.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = DOT2_F32_F16_CASES[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2_f32_f16({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_dot2_f32_f16_applies_op_sel_and_negation_per_half() {
    // src0's halves swapped (op_sel bit 0 set, op_sel_hi bit 0 clear), src1's
    // low half negated (neg_lo bit 1), src0's high-lane half negated (neg_hi
    // bit 0): (1, 2) . (3, 4) + 0.5 becomes 2 * -3 + -1 * 4 + 0.5 = -9.5.
    let body = vop3p_mods(
        0x13,
        0,
        [vgpr(1), vgpr(2), vgpr(3)],
        0b001,
        0b110,
        0b010,
        0b001,
        false,
    );
    let shader = build_valu_cs(&body);
    let cases: &[([u32; 4], u32)] = &[
        ([0x4000_3c00, 0x4400_4200, 0x3f00_0000, 0], 0xc118_0000),
        // (-1.5, 0.25) . (2, -8) - 1 with the same modifiers:
        // 0.25 * -2 + 1.5 * -8 - 1 = -13.5
        ([0x3400_be00, 0xc800_4000, 0xbf80_0000, 0], 0xc158_0000),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2_f32_f16 with modifiers({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_dot2_f32_f16_clamps_to_zero_one() {
    let body = vop3p_mods(0x13, 0, [vgpr(1), vgpr(2), vgpr(3)], 0, 0b111, 0, 0, true);
    let shader = build_valu_cs(&body);
    let cases: &[([u32; 4], u32)] = &[
        // 11.5 clamps to 1.0, -6 to 0.0
        ([0x4000_3c00, 0x4400_4200, 0x3f00_0000, 0], 0x3f80_0000),
        ([0x3400_be00, 0xc800_4000, 0xbf80_0000, 0], 0),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2_f32_f16 clamp({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_dot2c_f32_f16_accumulates_into_its_destination() {
    // v3 = v_dot2c_f32_f16 v1, v2 (v3 += v1 . v2), then v0 = v3.
    let body = [vop2(0x02, 3, vgpr(1), 2), vop1(0x001, 0, vgpr(3))];
    let shader = build_valu_cs(&body);
    let inputs: Vec<[u32; 4]> = DOT2_F32_F16_CASES.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = DOT2_F32_F16_CASES[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2c_f32_f16({:#010x}, {:#010x}) + {:#010x} = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

/// Inputs where rounding the sum of the two products before adding the
/// addend gives a different f32 than RDNA2, which rounds the exact
/// `p0 + p1 + addend` once (measured on gfx1036 in #465).
const DOT2_ROUNDED_ONCE_CASES: &[([u32; 4], u32)] = &[
    // (1, 2^-12) . (1, 2^-12) - 1 = 2^-24; 1 + 2^-24 rounded first gives 0.
    ([0x0c00_3c00, 0x0c00_3c00, 0xbf80_0000, 0], 0x3380_0000),
    // (1, 2^-14) . (1, 2^-14) + 2^24 = 2^24 + 1 + 2^-28: above the tie
    // between 2^24 and 2^24 + 2, so it rounds up; 1 + 2^-28 rounded first
    // lands on the tie, which rounds to even (2^24).
    ([0x0400_3c00, 0x0400_3c00, 0x4b80_0000, 0], 0x4b80_0001),
    // (-0, -0) . (1, 1) - 0 = -0: the sign of an all-negative-zero sum.
    ([0x8000_8000, 0x3c00_3c00, 0x8000_0000, 0], 0x8000_0000),
];

#[test]
fn v_dot2_f32_f16_rounds_the_exact_sum_once() {
    let body = vop3p_mods(0x13, 0, [vgpr(1), vgpr(2), vgpr(3)], 0, 0b111, 0, 0, false);
    let shader = build_valu_cs(&body);
    let inputs: Vec<[u32; 4]> = DOT2_ROUNDED_ONCE_CASES.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = DOT2_ROUNDED_ONCE_CASES[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2_f32_f16({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_dot2c_f32_f16_rounds_the_exact_sum_once() {
    let body = [vop2(0x02, 3, vgpr(1), 2), vop1(0x001, 0, vgpr(3))];
    let shader = build_valu_cs(&body);
    let inputs: Vec<[u32; 4]> = DOT2_ROUNDED_ONCE_CASES.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = DOT2_ROUNDED_ONCE_CASES[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2c_f32_f16({:#010x}, {:#010x}) + {:#010x} = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_sad_family_matches_rdna2() {
    let vop3_clamp = |op: u16| -> [u32; 2] {
        let [mut w0, w1] = vop3(op, 0, vgpr(1), vgpr(2), vgpr(3));
        w0 |= 1 << 15; // clamp = 1
        [w0, w1]
    };

    // v_sad_u8 (0x15a)
    let sad_u8 = build_valu_cs(&vop3(0x15a, 0, vgpr(1), vgpr(2), vgpr(3)));
    let sad_u8_clamp = build_valu_cs(&vop3_clamp(0x15a));
    // Bytes: |0x10-0x20|=16, |0xFF-0x00|=255, |0x05-0x05|=0, |0x80-0x70|=16 -> sum = 287
    run_cases(
        &sad_u8,
        &[
            [0x8005_FF10, 0x7005_0020, 100, 0],
            [0xFFFF_FFFF, 0x0000_0000, u32::MAX - 10, 0], // wraps: (2^32 - 11) + 1020 = 1009
        ],
        |backend, i, _, got| {
            let want = [387u32, 1009][i];
            assert_eq!(got, want, "{backend}: v_sad_u8 case {i}");
        },
    );
    run_cases(
        &sad_u8_clamp,
        &[[0xFFFF_FFFF, 0x0000_0000, u32::MAX - 10, 0]], // saturates to u32::MAX
        |backend, _, _, got| {
            assert_eq!(got, u32::MAX, "{backend}: v_sad_u8 clamp");
        },
    );

    // v_msad_u8 (0x171): zero bytes in src0 contribute 0 regardless of src1.
    let msad_u8 = build_valu_cs(&vop3(0x171, 0, vgpr(1), vgpr(2), vgpr(3)));
    // Byte 0: src0=0x00 -> skipped (even though src1=0xFF)
    // Byte 1: |0x10-0x00| = 16 (src1=0 is NOT skipped!)
    // Byte 2: src0=0x00 -> skipped
    // Byte 3: |0x80-0x70| = 16 -> sum = 32
    run_cases(
        &msad_u8,
        &[[0x8000_1000, 0x70AA_00FF, 5, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 37, "{backend}: v_msad_u8");
        },
    );

    // v_sad_u16 (0x15c)
    let sad_u16 = build_valu_cs(&vop3(0x15c, 0, vgpr(1), vgpr(2), vgpr(3)));
    let sad_u16_clamp = build_valu_cs(&vop3_clamp(0x15c));
    // Low: |0x0100 - 0x0300| = 0x200; High: |0xFFFF - 0x0001| = 0xFFFE -> sum = 0x101FE
    run_cases(
        &sad_u16,
        &[[0xFFFF_0100, 0x0001_0300, 2, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0x10200, "{backend}: v_sad_u16");
        },
    );
    run_cases(
        &sad_u16_clamp,
        &[[0xFFFF_FFFF, 0x0000_0000, u32::MAX - 5, 0]],
        |backend, _, _, got| {
            assert_eq!(got, u32::MAX, "{backend}: v_sad_u16 clamp");
        },
    );

    // v_sad_u32 (0x15d)
    let sad_u32 = build_valu_cs(&vop3(0x15d, 0, vgpr(1), vgpr(2), vgpr(3)));
    let sad_u32_clamp = build_valu_cs(&vop3_clamp(0x15d));
    run_cases(
        &sad_u32,
        &[
            [10, 30, 5, 0],
            [0xFFFF_FFFF, 0, 10, 0], // wraps: (2^32 - 1) + 10 = 9
        ],
        |backend, i, _, got| {
            let want = [25u32, 9][i];
            assert_eq!(got, want, "{backend}: v_sad_u32 case {i}");
        },
    );
    run_cases(
        &sad_u32_clamp,
        &[[0xFFFF_FFFF, 0, 10, 0]],
        |backend, _, _, got| {
            assert_eq!(got, u32::MAX, "{backend}: v_sad_u32 clamp");
        },
    );
}
