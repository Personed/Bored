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

/// The model of `image_bvh_intersect_ray` on `node` (a pointer into
/// `bvh`, a buffer of dwords at the descriptor's base).
fn intersect(bvh: &[u32], node: u32, r: &Ray, sort: bool, _ij: bool) -> [u32; 4] {
    // The reference is the crate's own model (watertight: a ray exactly
    // on a shared edge hits both triangles), which the upstream card
    // tests prove against the hardware.
    let at = ((node >> 3) * 16) as usize;
    let mut n = [0u32; 28];
    let take = (bvh.len() - at).min(28);
    n[..take].copy_from_slice(&bvh[at..at + take]);
    let ray = crate::bvh::Ray {
        extent: r.extent,
        origin: r.origin,
        direction: r.dir,
        inverse: r.inv,
    };
    crate::bvh::intersect(&n, node & 7, &ray, sort)
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
        // Inverted bounds: per-axis min/max normalizes the box.
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
        // On the edge TRI0 and TRI1 share: watertight means both hit it.
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
    // Watertight: a ray exactly on the shared edge hits both triangles.
    let owners = [tri[0], tri[1]]
        .iter()
        .filter(|&&n| intersect(&bvh, n, &mid, true, true)[0] != INF)
        .count();
    assert_eq!(owners, 2, "a ray on a shared edge hits both triangles");
    let mid = Ray::new([0.5, 0.5, 5.0], [0.0, 0.0, -1.0], 100.0);
    // The inverted box's per-axis min/max normalizes it: the ray enters
    // all four boxes, sorted by entry distance or in child order.
    assert_eq!(
        intersect(&bvh, full, &mid, true, true),
        [0x300, 0x204, 0x105, 0x401]
    );
    assert_eq!(
        intersect(&bvh, full, &mid, false, true),
        [0x105, 0x204, 0x300, 0x401]
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
            0x3402_0086, // v_lshlrev_b32 v1, 6, v0
            0xE038_1000,
            0x8002_0201, // buffer_load_dwordx4 v[2:5], v1, s[8:11], 0 offen
            0xE038_1010,
            0x8002_0601, // buffer_load_dwordx4 v[6:9], v1, s[8:11], 0 offen offset:16
            0xE038_1020,
            0x8002_0A01, // buffer_load_dwordx4 v[10:13], v1, s[8:11], 0 offen offset:32
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
            0x8003_0E01, // buffer_store_dwordx4 v[14:17], v1, s[12:15], 0 offen
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
                "v_lshlrev_b32",
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
            used: vec![table(ResourceType::Srv, 0), table(ResourceType::Uav, 8)],
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
        // One buffer like the upstream tests: 64-byte records, then the
        // nodes. A record's node pointer counts the records region in, so
        // `(pointer >> 3) << 6` lands on the node's byte in the buffer.
        let records = cases.len() * 64;
        let data: Vec<u32> = cases
            .iter()
            .flat_map(|(node, r)| {
                let mut w = vec![(records / 8) as u32 + node, r.extent.to_bits()];
                w.extend(f3(r.origin));
                w.extend(f3(r.dir));
                w.extend(f3(r.inv));
                w.extend([0; 5]);
                w
            })
            .chain(bvh.iter().copied())
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
                // Heap slot 0: the records and nodes; the buffer's bytes
                // map to title address BVH on, so a node read lands on
                // its byte past the records region. Set before `build`:
                // the hook fires while the DXIL is lowered.
                TEST_MEMORY.set(Some((0, BVH)));
                let shader = build(sort, ij);
                let (root, pso) = host
                    .compute_pipeline(&shader)
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(vec![0; 5]))],
                            views: &[
                                View::Buffer {
                                    data: &data,
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
                let got = &out[0];
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
