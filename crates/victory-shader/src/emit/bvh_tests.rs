//! The ray intersection emulation (`lower/bvh.rs`) and title memory through
//! the address map (`lower/global.rs`), against a CPU model of the
//! instruction: Microsoft's software model in the XDXR traversal headers
//! (`SoftwareBvhIntersectRay`, `TestXboxBvh4Node`, `TestXboxBvh4Leaf`),
//! transcribed. On a GPU, a program reads rays through the map, tests
//! them against hand-built nodes and stores the results through the map.

const INF: u32 = 0x7F80_0000;
const ONE: u32 = 0x3F80_0000;
const INVALID: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
struct Ray {
    origin: [f32; 3],
    dir: [f32; 3],
    inv: [f32; 3],
    extent: f32,
}

impl Ray {
    fn new(origin: [f32; 3], dir: [f32; 3], extent: f32) -> Ray {
        Ray {
            origin,
            dir,
            inv: dir.map(|d| 1.0 / d),
            extent,
        }
    }
}

/// HLSL's min and max: a NaN operand gives the other.
fn min(a: f32, b: f32) -> f32 {
    if a.is_nan() || b < a { b } else { a }
}

fn max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b > a { b } else { a }
}

fn half(h: u32) -> f32 {
    let h = h & 0xFFFF;
    let sign = if h >> 15 != 0 { -1.0 } else { 1.0 };
    let (e, m) = ((h >> 10) & 0x1F, h & 0x3FF);
    match e {
        0 => sign * m as f32 * (2f32).powi(-24),
        31 => f32::from_bits((h >> 15) << 31 | 0x7F80_0000 | m << 13),
        _ => f32::from_bits((h >> 15) << 31 | (e + 112) << 23 | m << 13),
    }
}

fn ray_box(r: &Ray, lo: [f32; 3], hi: [f32; 3]) -> f32 {
    if (0..3).any(|i| lo[i] > hi[i]) {
        return f32::INFINITY;
    }
    let (mut near, mut far) = (0.0f32, r.extent);
    for i in 0..3 {
        if r.dir[i] == 0.0 {
            if r.origin[i] < lo[i] || r.origin[i] > hi[i] {
                return f32::INFINITY;
            }
        } else {
            let t1 = (lo[i] - r.origin[i]) * r.inv[i];
            let t2 = (hi[i] - r.origin[i]) * r.inv[i];
            near = max(near, min(t1, t2));
            far = min(far, max(t1, t2));
            if far < near {
                return f32::INFINITY;
            }
        }
    }
    near
}

fn test_box(n: &[u32], small: bool, r: &Ray, sort: bool) -> [u32; 4] {
    let f = |w: u32| f32::from_bits(w);
    let clamp = |x: f32| max(min(x, f32::MAX), -f32::MAX);
    let mut d = [0.0f32; 4];
    let mut p = [n[0], n[1], n[2], n[3]];
    for k in 0..4 {
        let (lo, hi) = if small {
            let w = [n[4 + 3 * k], n[5 + 3 * k], n[6 + 3 * k]];
            (
                [half(w[0]), half(w[0] >> 16), half(w[1])].map(clamp),
                [half(w[1] >> 16), half(w[2]), half(w[2] >> 16)].map(clamp),
            )
        } else {
            let b = &n[4 + 6 * k..];
            ([f(b[0]), f(b[1]), f(b[2])], [f(b[3]), f(b[4]), f(b[5])])
        };
        d[k] = ray_box(r, lo, hi);
        if d[k].is_nan() || d[k] > r.extent {
            p[k] = INVALID;
        }
    }
    if sort {
        for (x, y) in [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)] {
            if d[x] > d[y] {
                d.swap(x, y);
                p.swap(x, y);
            }
        }
    }
    p
}

const TRIANGLES: [[usize; 3]; 4] = [[0, 1, 2], [1, 3, 2], [2, 3, 4], [2, 4, 0]];

fn yzx(v: [f32; 3]) -> [f32; 3] {
    [v[1], v[2], v[0]]
}

fn zxy(v: [f32; 3]) -> [f32; 3] {
    [v[2], v[0], v[1]]
}

fn break_tie(ay: f32, by: f32, cy: f32, winding: bool, missed: &mut bool) {
    if by.abs() == 0.0 && cy.abs() == 0.0 {
        *missed |= ay > 0.0;
    } else {
        let below = by < 0.0 || (by.abs() == 0.0 && cy > 0.0);
        *missed |= below ^ winding;
    }
}

fn test_leaf(n: &[u32], tri: usize, r: &Ray, ij: bool) -> [u32; 4] {
    let vert = |k: usize| [0, 1, 2].map(|j| f32::from_bits(n[3 * k + j]));
    let [ia, ib, ic] = TRIANGLES[tri];
    let (mut v0, mut v1, mut v2) = (vert(ia), vert(ib), vert(ic));
    let remap = n[15] >> (8 * tri);
    let (mut dir, mut origin) = (r.dir, r.origin);
    let mag = dir.map(f32::abs);
    let m = max(mag[0], max(mag[1], mag[2]));
    let sw: Option<fn([f32; 3]) -> [f32; 3]> = if mag[0] == m {
        Some(yzx)
    } else if mag[1] == m {
        Some(zxy)
    } else {
        None
    };
    if let Some(s) = sw {
        (dir, origin, v0, v1, v2) = (s(dir), s(origin), s(v0), s(v1), s(v2));
    }
    let sub = |a: [f32; 3]| [a[0] - origin[0], a[1] - origin[1], a[2] - origin[2]];
    let (v0, v1, v2) = (sub(v0), sub(v1), sub(v2));
    let shear = |v: [f32; 3]| {
        [
            v[0] * dir[2] - dir[0] * v[2],
            v[1] * dir[2] - dir[1] * v[2],
            v[2],
        ]
    };
    let (a, b, c) = (shear(v0), shear(v1), shear(v2));
    let u = c[0] * b[1] - c[1] * b[0];
    let v = a[0] * c[1] - a[1] * c[0];
    let w = b[0] * a[1] - b[1] * a[0];
    let t_num = u * a[2] + v * b[2] + w * c[2];
    let t_den = u * dir[2] + v * dir[2] + w * dir[2];
    let mut missed = (u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0);
    missed |= t_den.abs() == 0.0;
    missed |= t_num * f32::from_bits(t_den.to_bits() & 0x8000_0000 | ONE) <= 0.0;
    missed |= t_num.is_nan();
    let winding = t_den > 0.0;
    if u.abs() == 0.0 {
        break_tie(a[1], b[1], c[1], winding, &mut missed);
    }
    if v.abs() == 0.0 {
        break_tie(b[1], c[1], a[1], winding, &mut missed);
    }
    if w.abs() == 0.0 {
        break_tie(c[1], a[1], b[1], winding, &mut missed);
    }
    if missed {
        return [INF, ONE, 0, 0];
    }
    let pick = |s: u32| match s & 3 {
        1 => v,
        2 => w,
        _ => u,
    };
    let (mut i, mut j) = (pick(remap), pick(remap >> 2));
    if (i < 0.0 && t_den > 0.0) || (i > 0.0 && t_den < 0.0) {
        (i, j) = (-i, -j);
    }
    if ij {
        [t_num.to_bits(), t_den.to_bits(), i.to_bits(), j.to_bits()]
    } else {
        [t_num.to_bits(), t_den.to_bits(), n[15], 1]
    }
}

/// The model of `image_bvh_intersect_ray` on `node` (a pointer into
/// `bvh`, a buffer of dwords at the descriptor's base).
fn intersect(bvh: &[u32], node: u32, r: &Ray, sort: bool, ij: bool) -> [u32; 4] {
    let at = ((node >> 3) * 16) as usize;
    let mut n = bvh[at..].to_vec();
    n.resize(28, 0);
    match node & 7 {
        t @ 0..=3 => test_leaf(&n, t as usize, r, ij),
        t => test_box(&n, t == 4, r, sort),
    }
}

fn f3(v: [f32; 3]) -> [u32; 3] {
    v.map(f32::to_bits)
}

/// Six halves of a box: min x y, min z max x, max y z.
fn box16(lo: [f32; 3], hi: [f32; 3]) -> [u32; 3] {
    let h = |x: f32| -> u32 {
        // Exact for the small dyadic values the tests use.
        let b = x.to_bits();
        if x == 0.0 {
            return (b >> 16) & 0x8000;
        }
        let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
        assert!(
            (1..31).contains(&e) && b & 0x1FFF == 0,
            "{x} is no exact half"
        );
        (b >> 16) & 0x8000 | (e as u32) << 10 | (b >> 13) & 0x3FF
    };
    [
        h(lo[0]) | h(lo[1]) << 16,
        h(lo[2]) | h(hi[0]) << 16,
        h(hi[1]) | h(hi[2]) << 16,
    ]
}

/// The test BVH: a leaf (unit 0), a float box node (units 2-3) and a
/// half box node (unit 4), with their pointers.
fn scene() -> (Vec<u32>, [u32; 4], u32, u32) {
    let mut bvh = vec![0u32; 16 * 6];
    let verts: [[f32; 3]; 5] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    for (k, v) in verts.iter().enumerate() {
        bvh[3 * k..3 * k + 3].copy_from_slice(&f3(*v));
    }
    // TRI0 i=V j=W, TRI1 i=W j=U, TRI2 i=U j=V, TRI3 i=V j=U.
    bvh[15] = 0x09 | 0x02 << 8 | 0x04 << 16 | 0x01 << 24;
    let boxes: [([f32; 3], [f32; 3]); 4] = [
        ([0.0, 0.0, -3.0], [1.0, 1.0, -2.0]),
        ([0.0, 0.0, -1.0], [1.0, 1.0, 0.0]),
        // Inverted: never hit.
        ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, -6.0], [1.0, 1.0, -5.0]),
    ];
    let children = [0x105, 0x204, 0x300, 0x401];
    let full = 2 * 16;
    bvh[full..full + 4].copy_from_slice(&children);
    for (k, (lo, hi)) in boxes.iter().enumerate() {
        let b = full + 4 + 6 * k;
        bvh[b..b + 3].copy_from_slice(&f3(*lo));
        bvh[b + 3..b + 6].copy_from_slice(&f3(*hi));
    }
    let small = 4 * 16;
    bvh[small..small + 4].copy_from_slice(&children);
    for (k, (lo, hi)) in boxes.iter().enumerate() {
        let b = small + 4 + 3 * k;
        bvh[b..b + 3].copy_from_slice(&box16(*lo, *hi));
    }
    let leaf = [0, 1, 2, 3];
    (bvh, leaf, (2 << 3) | 5, (4 << 3) | 4)
}

/// 32 (node, ray) cases: every triangle, shared edges, parallel and
/// backward rays, back faces, both box kinds with sorting, a short extent
/// and misses, then seeded random rays at the boxes.
fn cases() -> Vec<(u32, Ray)> {
    let (_, tri, full, small) = scene();
    let down = |x, y| Ray::new([x, y, 5.0], [0.0, 0.0, -1.0], 100.0);
    let mut c = vec![
        (tri[0], down(0.25, 0.25)),
        (tri[1], down(0.25, 0.25)),
        (tri[0], down(0.75, 0.75)),
        (tri[1], down(0.75, 0.75)),
        // On the edge TRI0 and TRI1 share: exactly one owns it.
        (tri[0], down(0.5, 0.5)),
        (tri[1], down(0.5, 0.5)),
        (tri[2], down(0.25, 0.5)),
        (tri[3], Ray::new([-1.0, 0.25, 0.25], [1.0, 0.0, 0.0], 100.0)),
        // In the triangle's plane, and moving away from it.
        (tri[0], Ray::new([-1.0, 0.25, 0.0], [1.0, 0.0, 0.0], 100.0)),
        (
            tri[0],
            Ray::new([0.25, 0.25, -5.0], [0.0, 0.0, -1.0], 100.0),
        ),
        // From below: the back face.
        (tri[0], Ray::new([0.25, 0.25, -5.0], [0.0, 0.0, 1.0], 100.0)),
        (
            tri[1],
            Ray::new([0.5, 0.25, 4.0], [0.125, 0.25, -1.0], 100.0),
        ),
        (full, down(0.5, 0.5)),
        (full, Ray::new([0.5, 0.5, 5.0], [0.0, 0.0, -1.0], 6.0)),
        (full, down(1.5, 0.5)),
        (full, Ray::new([0.5, 0.5, -2.5], [1.0, 0.0, 0.0], 100.0)),
        (small, down(0.5, 0.5)),
        (small, Ray::new([0.5, 0.5, 5.0], [0.0, 0.0, -1.0], 6.0)),
        (small, down(1.5, 0.5)),
        (small, Ray::new([0.5, 0.5, -2.5], [1.0, 0.0, 0.0], 100.0)),
    ];
    let mut seed = 0x1234_5678u32;
    let mut rand = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0
    };
    while c.len() < 32 {
        let node = if c.len() % 2 == 0 { full } else { small };
        // Over the boxes' footprint (x, y in [0, 1]), give or take.
        let origin = [0.5 + rand() * 0.35, 0.5 + rand() * 0.35, rand() + 4.0];
        let dir = [rand() * 0.25, rand() * 0.25, -1.0 - rand().abs()];
        c.push((node, Ray::new(origin, dir, 2.0 + rand().abs() * 4.0)));
    }
    c
}

#[test]
fn model_hits_and_orders_as_expected() {
    let (bvh, tri, full, _) = scene();
    let r = Ray::new([0.25, 0.25, 5.0], [0.0, 0.0, -1.0], 100.0);
    let hit = intersect(&bvh, tri[0], &r, true, true);
    let t = f32::from_bits(hit[0]) / f32::from_bits(hit[1]);
    assert_eq!(t, 5.0);
    assert_eq!(intersect(&bvh, tri[1], &r, true, true), [INF, ONE, 0, 0]);
    // Barycentrics: TRI0 remaps i to V, j to W.
    let (i, j) = (
        f32::from_bits(hit[2]) / f32::from_bits(hit[1]),
        f32::from_bits(hit[3]) / f32::from_bits(hit[1]),
    );
    assert!(
        (i - 0.25).abs() < 1e-6 && (j - 0.25).abs() < 1e-6,
        "{i} {j}"
    );
    let mid = Ray::new([0.5, 0.5, 5.0], [0.0, 0.0, -1.0], 100.0);
    let owners = [tri[0], tri[1]]
        .iter()
        .filter(|&&n| intersect(&bvh, n, &mid, true, true)[0] != INF)
        .count();
    assert_eq!(owners, 1, "a shared edge belongs to one triangle");
    let mid = Ray::new([0.5, 0.5, 5.0], [0.0, 0.0, -1.0], 100.0);
    assert_eq!(
        intersect(&bvh, full, &mid, true, true),
        [0x204, 0x105, 0x401, INVALID]
    );
    assert_eq!(
        intersect(&bvh, full, &mid, false, true),
        [0x105, 0x204, INVALID, 0x401]
    );
}

#[test]
fn the_cases_cover_hits_misses_and_boxes() {
    let (bvh, ..) = scene();
    let (mut hits, mut misses, mut box_hits, mut box_misses) = (0, 0, 0, 0);
    for (node, r) in cases() {
        let got = intersect(&bvh, node, &r, true, true);
        match (node & 7 > 3, got[0] == INF || got == [INVALID; 4]) {
            (false, false) => hits += 1,
            (false, true) => misses += 1,
            (true, false) => box_hits += 1,
            (true, true) => box_misses += 1,
        }
    }
    assert!(hits >= 5 && misses >= 4, "{hits} hits, {misses} misses");
    assert!(
        box_hits >= 10 && box_misses >= 2,
        "{box_hits} box hits, {box_misses} misses"
    );
}

#[cfg(windows)]
mod gpu {
    use super::super::host::{Dispatch, Host, Root, Stride, View};
    use super::super::lower::intersect::TEST_MEMORY;
    use super::super::{HostShader, lower, validator};
    use super::*;
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, ir, resource, structure};

    const BVH: u64 = 0x2_0000_0100;

    /// Per lane: 12 dwords of ray at `48 * lane` of the record buffer
    /// (node, extent, origin, direction, inverse direction), read through
    /// the descriptor the table's SRV entry builds; the intersection with
    /// the descriptor `s[16:19]`, 4 dwords stored at `16 * lane` through
    /// the UAV entry's descriptor. An access whose address is title
    /// memory lowers through the host's directory, which the test has no
    /// copies for, so the reads and the write go through descriptors.
    /// `desc1`/`desc3` are the descriptor's flag dwords, written as
    /// literals like the console compiler does.
    fn program(desc0: u32, desc1: u32, desc3: u32) -> Vec<u32> {
        let w = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
            0x1602_00B0, // v_mul_u32_u24 v1, 48, v0
            0xE038_1000,
            0x0002_0201, // buffer_load_dwordx4 v[2:5], v1, s[8:11], 0 offen
            0xE038_1010,
            0x0002_0601, // buffer_load_dwordx4 v[6:9], v1, s[8:11], 0 offen offset:16
            0xE038_1020,
            0x0002_0A01, // buffer_load_dwordx4 v[10:13], v1, s[8:11], 0 offen offset:32
            0xBE90_03FF,
            desc0, // s_mov_b32 s16, desc0
            0xBE91_03FF,
            desc1,       // s_mov_b32 s17, desc1
            0xBE92_03C1, // s_mov_b32 s18, -1
            0xBE93_03FF,
            desc3,       // s_mov_b32 s19, desc3
            0xF198_9F01, // image_bvh_intersect_ray v[14:17], v[2:12], s[16:19]
            0x0004_0E02,
            0x3402_0084, // v_lshlrev_b32 v1, 4, v0
            0xE078_1000,
            0x0003_0E01, // buffer_store_dwordx4 v[14:17], v1, s[12:15], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        let names: Vec<_> =
            crate::gfx10::decode(&w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                .expect("decode")
                .iter()
                .map(|i| i.name().unwrap_or("?"))
                .collect();
        assert_eq!(
            names,
            [
                "s_mov_b32",
                "s_load_dwordx4",
                "s_load_dwordx4",
                "v_mul_u32_u24",
                "buffer_load_dwordx4",
                "buffer_load_dwordx4",
                "buffer_load_dwordx4",
                "s_mov_b32",
                "s_mov_b32",
                "s_mov_b32",
                "s_mov_b32",
                "image_bvh_intersect_ray",
                "v_lshlrev_b32",
                "buffer_store_dwordx4",
                "s_endpgm"
            ]
        );
        w
    }

    fn build(sort: bool, ij: bool) -> HostShader {
        let base = BVH >> 8;
        let desc0 = base as u32;
        let desc1 = (base >> 32) as u32 & 0xFF | u32::from(sort) << 31;
        let desc3 = 0x88FF_FFFF | u32::from(ij) << 24;
        let table = |kind, at| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: at,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 4,
            slot: 1,
        };
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 32,
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
            used: vec![
                table(ResourceType::Srv, 0),
                table(ResourceType::Srv, 4),
                table(ResourceType::Uav, 8),
            ],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1,
                threads: [32, 1, 1],
            }),
            code: program(desc0, desc1, desc3)
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect(),
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
        assert!(res.scalar.iter().all(|s| s.1.param != resource::CODE));
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
            crumbs: None,
            mesh_stride: 0,
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
            address_map: lowered.address_map,
            descriptor_words: lowered.descriptor_words,
            root_uavs: lowered.root_uavs,
            root_indexed_views: Vec::new(),
        };
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut shader.dxil.clone())
                .expect("DXIL validation failed");
        } else {
            eprintln!("no dxil.dll; validation skipped");
        }
        shader
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn bvh_program_tracks_and_lowers() {
        build(true, true);
        build(false, true);
    }

    #[test]
    fn bvh_intersection_matches_the_model_on_the_gpu() {
        let (bvh, ..) = scene();
        let cases = cases();
        let rays: Vec<u32> = cases
            .iter()
            .flat_map(|(node, r)| {
                let mut w = vec![*node, r.extent.to_bits()];
                w.extend(f3(r.origin));
                w.extend(f3(r.dir));
                w.extend(f3(r.inv));
                w.push(0);
                w
            })
            .collect();
        let _one = super::super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                eprintln!("no SM 6.6 {label} device; skipped");
                continue;
            };
            for (sort, ij) in [(true, true), (false, true)] {
                let shader = build(sort, ij);
                let (root, pso) = host
                    .compute_pipeline(&shader)
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                // Heap slot 1 holds the nodes from title address BVH on.
                TEST_MEMORY.set(Some((1, BVH)));
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(vec![0; 5]))],
                            views: &[
                                View::Buffer {
                                    data: &rays,
                                    stride: Stride::Raw,
                                },
                                View::Buffer {
                                    data: &bvh,
                                    stride: Stride::Raw,
                                },
                                View::RawUav(128),
                            ],
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                TEST_MEMORY.set(None);
                let got = &out[2];
                for (lane, (node, r)) in cases.iter().enumerate() {
                    let want = intersect(&bvh, *node, r, sort, ij);
                    let g = &got[4 * lane..4 * lane + 4];
                    let what = format!(
                        "{label} sort={sort} ij={ij} lane {lane} node {node:#x}: got {g:x?}, want {want:x?}"
                    );
                    if node & 7 > 3 || want[0] == INF {
                        assert_eq!(g, want, "{what}");
                        continue;
                    }
                    assert_ne!(g[0], INF, "{what}");
                    let f = |x: u32| f32::from_bits(x);
                    let (gd, wd) = (f(g[1]), f(want[1]));
                    assert!(close(f(g[0]) / gd, f(want[0]) / wd), "{what}: t");
                    if ij {
                        assert!(close(f(g[2]) / gd, f(want[2]) / wd), "{what}: i");
                        assert!(close(f(g[3]) / gd, f(want[3]) / wd), "{what}: j");
                    } else {
                        assert_eq!(g[2..], want[2..], "{what}");
                    }
                }
            }
        }
    }
}
