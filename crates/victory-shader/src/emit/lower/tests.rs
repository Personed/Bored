use super::{Lowered, Numeric, Parts, buffer_format, half_shift, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

#[test]
fn packed_pixel_inputs_find_their_second_row() {
    use super::super::container::SigElement;
    use super::half_input_partners;
    use crate::native::{PsInput, Usage};
    let entry = |row, row2, pack2| PsInput {
        usage: Usage::Generic,
        row,
        mask: 0xF,
        attr: row,
        flat: false,
        fp16: true,
        pack2,
        row2,
    };
    let element = |row| SigElement {
        name: "TEXCOORD".into(),
        index: row,
        kind: 0,
        comp: 9,
        interp: 2,
        row: Some(row),
        col: 0,
        cols: 4,
        usage: 0,
        half: true,
    };
    // Rows 0 and 3 pack and leave their second row out; 5 names its
    // own (6). Unnamed 16-bit rows: 1, 4, 6, 7.
    let io = StageIo::Ps {
        inputs: vec![entry(0, 0, true), entry(3, 3, true), entry(5, 6, true)],
        input_addr: 0,
        db_shader_control: 0,
        patches: vec![],
        rsrc2: 0,
    };
    let sig: Vec<_> = [0, 1, 3, 4, 5, 6, 7].map(element).into();
    let p = half_input_partners(&io, &sig);
    assert_eq!(p.get(&0), Some(&1));
    assert_eq!(p.get(&3), Some(&4));
    assert_eq!(p.get(&5), None);
}

/// The packed pedestrian layouts of Grand Theft Auto VI: an element
/// behind another one in its register follows in the first parameter
/// with a dword left, and is not a half of its register's row.
#[test]
fn packed_elements_behind_another_take_the_first_free_dword() {
    use super::super::container::SigElement;
    use super::packed_half;
    let element = |index, row, col, cols| SigElement {
        name: "TEXCOORD".into(),
        index,
        kind: 0,
        comp: 9,
        interp: 2,
        row: Some(row),
        col,
        cols,
        usage: 0,
        half: true,
    };
    // MS_VS_PedTransformPreSkinD_Packed 005fe88ec1bdc0c7 and its pixel
    // shader 93667197d3d05e1c: TEXCOORD6 is one value at register 4
    // column 3; parameter 3 (rows 6 and 7) carries it in dword 3.
    let sig = vec![
        element(1, 2, 0, 4),
        element(2, 3, 0, 4),
        element(4, 4, 0, 3),
        element(6, 4, 3, 1),
        element(3, 5, 0, 4),
        element(5, 6, 0, 4),
        element(7, 7, 0, 2),
        element(9, 9, 0, 3),
        element(11, 10, 0, 3),
        element(15, 12, 0, 4),
    ];
    let packs = [
        (2, Some(3)),
        (4, Some(5)),
        (6, Some(7)),
        (9, Some(10)),
        (12, None),
    ];
    let at = |a, h| packed_half(&sig, &packs, a, h);
    // Parameter 2: TEXCOORD4's three values, a zero half, TEXCOORD3.
    assert_eq!(at(4, 2), Some((4, 2)));
    assert_eq!(at(4, 3), None);
    assert_eq!(at(4, 4), Some((5, 0)));
    assert_eq!(at(4, 7), Some((5, 3)));
    // Parameter 3: TEXCOORD5, TEXCOORD7, then TEXCOORD6 alone.
    assert_eq!(at(6, 3), Some((6, 3)));
    assert_eq!(at(6, 5), Some((7, 1)));
    assert_eq!(at(6, 6), Some((4, 3)));
    assert_eq!(at(6, 7), None);
    // Parameters without such an element are as before.
    assert_eq!(at(9, 3), None);
    assert_eq!(at(9, 4), Some((10, 0)));
    assert_eq!(at(12, 3), Some((12, 3)));
    assert_eq!(at(12, 4), None);
    // The common layout (29805d23a495cdf8): TEXCOORD7 is two values at
    // register 1 columns 2-3; parameter 0 is TEXCOORD0, TEXCOORD1,
    // TEXCOORD7.
    let sig = vec![
        element(0, 1, 0, 2),
        element(7, 1, 2, 2),
        element(1, 2, 0, 4),
        element(2, 3, 0, 4),
        element(4, 4, 0, 3),
    ];
    let packs = [(3, Some(4)), (1, Some(2))];
    let at = |a, h| packed_half(&sig, &packs, a, h);
    assert_eq!(at(1, 1), Some((1, 1)));
    assert_eq!(at(1, 2), Some((2, 0)));
    assert_eq!(at(1, 5), Some((2, 3)));
    assert_eq!(at(1, 6), Some((1, 2)));
    assert_eq!(at(1, 7), Some((1, 3)));
    assert_eq!(at(3, 6), Some((4, 2)));
    assert_eq!(at(3, 7), None);
    // A row no element holds gives nothing, as before.
    assert_eq!(packed_half(&sig, &[(8, None)], 8, 0), None);
}

/// Whether a synthetic compute program translates with these input registers.
fn lowers(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        crumbs: None,
    })
    .map(|_| ())
}

/// Lowers a compute program with the root parameters `layout` (in user
/// SGPRs from s0 on) and the used resources `used`.
fn lower_with(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<Lowered, crate::ShaderError> {
    lower_wave(words, layout, used, false)
}

/// [`lower_with`] in wave64 if `wave64`.
fn lower_wave(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    wave64: bool,
) -> Result<Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 256,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            // The layout's user SGPRs (at least two), 512 bytes of LDS.
            rsrc2: 1 << 15 | sgprs.max(2) << 1,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, wave64, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        crumbs: None,
    })
}

/// A counted loop (`s5` from 0 to 4) in a compute program with one root
/// constant in `s0`, lowered with breadcrumbs for `crumbs`.
fn counted_loop(crumbs: Option<u32>) -> Lowered {
    let words = [
        0xBE85_0380u32, // s_mov_b32 s5, 0
        0x8005_8105,    // s_add_u32 s5, s5, 1
        0xBF0A_8405,    // s_cmp_lt_u32 s5, 4
        0xBF85_FFFD,    // s_cbranch_scc1 -3
        S_ENDPGM,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![root(RootType::Constants, 0, 0)],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15 | 2 << 1,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        crumbs,
    })
    .unwrap()
}

#[test]
fn breadcrumbs_add_a_loop_store_and_one_user_data_dword() {
    let plain = counted_loop(None);
    assert!(
        plain
            .user_data
            .iter()
            .all(|u| u.root != crate::emit::BREADCRUMB_ROOT)
    );
    let crumbs = counted_loop(Some(7));
    // The store (and the heap handle it goes through) are in the module.
    assert!(crumbs.dxil.len() > plain.dxil.len());
    assert!(crumbs.wave_ops && !plain.wave_ops);
    // The first dword past the console's (one root constant: s0).
    let heap_index: Vec<_> = crumbs
        .user_data
        .iter()
        .filter(|u| u.root == crate::emit::BREADCRUMB_ROOT)
        .collect();
    assert_eq!(heap_index.len(), 1);
    assert_eq!(heap_index[0].index, 1);
    // Everything else is unchanged.
    assert_eq!(
        crumbs.user_data.len(),
        plain.user_data.len() + 1,
        "{:?}",
        crumbs.user_data
    );
}

/// Root parameter `slot` in user SGPRs from `user_data` on.
fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: if root == RootType::Descriptor { 2 } else { 1 },
        base_register: 0,
        user_data,
        slot,
    }
}

#[test]
fn buffer_cache_invalidations_lower_as_no_ops() {
    // MUBUF opcode 0x071/0x072 with no data operand: the title asks the
    // host to invalidate its cached view of a buffer.
    assert!(
        lowers(&[0xE1C4_0000, 0, S_ENDPGM], 0).is_ok(),
        "buffer_gl0_inv"
    );
    assert!(
        lowers(&[0xE1C8_0000, 0, S_ENDPGM], 0).is_ok(),
        "buffer_gl1_inv"
    );
}

#[test]
fn a_scalar_carry_chain_reaches_the_select() {
    // The high dword of a 64-bit add or subtract, then a select on the
    // carry (borrow) the high half leaves in SCC, and a branch on it
    // (only code with an effect needs its operands).
    let add = [
        0xBE80_03C1, // s_mov_b32 s0, -1
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8000_8100, // s_add_u32 s0, s0, 1
        0x8201_8001, // s_addc_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];
    let sub = [
        0xBE80_0380, // s_mov_b32 s0, 0
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8080_8100, // s_sub_u32 s0, s0, 1
        0x8281_8001, // s_subb_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];

    for words in [add, sub] {
        lowers(&words, 0).unwrap();
    }
}

#[test]
fn v_cndmask_accepts_uniform_and_constant_masks() {
    // Uniform select (`s_cselect_b32`) and inline constant masks in `v_cndmask_b32`
    // (resolved in `Structured::other` or directly as constants).
    let words = [
        0xBF06_8000u32, // s_cmp_eq_u32 s0, 0
        0x8500_80C1,    // s_cselect_b32 s0, -1, 0
        0xD501_0001,
        0x0001_0080, // v_cndmask_b32 v1, 0, 1, s0
        0xD501_0002,
        0x0201_02C1, // v_cndmask_b32 v2, 1, v1, -1
        0xD834_0000,
        0x0000_0200, // ds_write_b32 v0, v2
        S_ENDPGM,
    ];
    lowers(&words, 0x8001).unwrap();
}

fn perm_features(selector: u32) -> u64 {
    // Synthetic compute shader:
    // v0 is the thread ID and can supply a dynamic selector.
    // Store the result in LDS to keep the permutation live.
    let words = [
        0x7E02_02FFu32,
        0x817F_34AB,
        0x7E04_02FF,
        0xFE01_56CD,
        0xD744_0003,
        257 | (258 << 9) | (selector << 18),
        0x3408_0082,
        0xD834_0000,
        0x0000_0304,
        S_ENDPGM,
    ];

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15,
            threads: [64, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, false, 256).unwrap();

    assert!(
        p.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name() == "v_perm_b32")
    );

    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();

    let io = io::raise(&p, &native).unwrap();

    let emitted = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        crumbs: None,
    })
    .unwrap();

    let container = crate::Container::parse(&emitted.dxil).unwrap();

    let features = container.part(crate::FourCC::SFI0).unwrap();

    u64::from_le_bytes(features.data.try_into().unwrap())
}

#[test]
fn perm_constant_selector_needs_no_int64() {
    assert_eq!(perm_features(128) & 0x8000, 0);
}

#[test]
fn perm_dynamic_selector_needs_no_int64() {
    assert_eq!(
        perm_features(256) & 0x8000,
        0,
        "dynamic v_perm_b32 must not require INT64"
    );
}

#[test]
fn scalar_loads_through_a_heap_buffer_that_vector_accesses_use_read_a_buffer_view() {
    // A descriptor from the resource heap (s0) at an index from a root
    // constant (s1), read by a scalar buffer load, then used by nothing
    // else, by a vector buffer load, or by a vector buffer store. A slot
    // that vector accesses use is a buffer view (an SRV if they only
    // read it, a UAV if one writes it), so the scalar load must not read
    // it through a constant buffer handle (a descriptor of the wrong
    // type). With no other use, nothing says it isn't a CBV.
    #[derive(Clone, Copy)]
    enum Also {
        Nothing,
        Load,
        Store,
    }
    let program = |also: Also| {
        let mut words = vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0x8F06_8501, // s_lshl_b32 s6, s1, 5
            0xF408_0202,
            0x0C00_0000, // s_load_dwordx4 s[8:11], s[4:5], s6
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF420_0304,
            0xFA00_0000, // s_buffer_load_dword s12, s[8:11], 0x0
            0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
            0x7E02_020C, // v_mov_b32 v1, s12
        ];
        match also {
            Also::Nothing => {}
            Also::Load => words.extend([
                0xE030_1000,
                0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 offen
                0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
                0x4A02_0302, // v_add_nc_u32 v1, v2, v1
            ]),
            Also::Store => words.extend([
                0xE070_1000,
                0x8002_0100, // buffer_store_dword v1, v0, s[8:11], 0 offen
            ]),
        }
        words.extend([
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ]);
        words
    };
    let layout = || {
        vec![
            root(RootType::ResourceHeap, 0, 0),
            root(RootType::Constants, 1, 1),
        ]
    };
    let cbuffers = |also| {
        lower_with(&program(also), layout(), vec![])
            .unwrap()
            .cbuffer_handles
    };
    assert_eq!(cbuffers(Also::Nothing), 1, "no other use: a CBV");
    assert_eq!(cbuffers(Also::Load), 0, "buffer_load: an SRV");
    assert_eq!(cbuffers(Also::Store), 0, "buffer_store: a UAV");
}

#[test]
fn a_descriptor_selected_by_scc_picks_its_heap_index() {
    // Two buffer descriptors loaded from the table at s0, one picked by
    // a root constant (s1) with s_cselect_b64 per half, and loaded
    // through by index and stored to LDS so the load is needed.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xF408_0302,
        0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
        0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
        0xE030_2000,
        0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    // Both are structured buffers of dwords in the table's range.
    let used = || {
        vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }]
    };
    lower_with(&words, layout(), used()).unwrap();
}

#[test]
fn image_atomic_64_bit_lowers_with_int64_feature() {
    // Table address at s0 loaded into s[2:3], s_load_dwordx8 s[0:7], s[2:3], null
    // then MIMG image_atomic_add v[0:1], v[2:3], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D
    let words = [
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF40C_0001,
        0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
        0xF044_0308,
        0x0000_0002, // image_atomic_add v[0:1], v[2:3], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D
        S_ENDPGM,
    ];
    let layout = || vec![root(RootType::Table, 0, 0)];
    let used = || {
        vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        }]
    };
    let l = lower_with(&words, layout(), used()).unwrap();
    #[cfg(windows)]
    if let Ok(mut v) = crate::emit::validator::Validator::load(
        crate::emit::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut l.dxil.clone()).unwrap();
    }
    let container = crate::Container::parse(&l.dxil).unwrap();
    let features = container.part(crate::FourCC::SFI0).unwrap();
    let sfi0 = u64::from_le_bytes(features.data.try_into().unwrap());
    assert_ne!(
        sfi0 & 0x8000,
        0,
        "64-bit image atomic must declare INT64 feature"
    );
    assert_ne!(
        sfi0 & 0x40_0000,
        0,
        "64-bit image atomic must declare ATOMIC_INT64_ON_TYPED_RESOURCE"
    );
    assert_ne!(
        sfi0 & 0x1000_0000,
        0,
        "64-bit image atomic from descriptor heap must declare ATOMIC_INT64_ON_HEAP_RESOURCE"
    );
}

#[test]
fn image_atomic_32_and_64_handles_do_not_collide_in_cache() {
    // Table address at s0 loaded into s[2:3], s_load_dwordx8 s[0:7], s[2:3], null
    // then 32-bit image_atomic_add v0, v2, s[0:7] dmask:0x1
    // then 64-bit image_atomic_add v[0:1], v[2:3], s[0:7] dmask:0x3
    let words = [
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF40C_0001,
        0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
        0xF044_0108,
        0x0000_0002, // image_atomic_add v0, v2, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D
        0xF044_0308,
        0x0000_0002, // image_atomic_add v[0:1], v[2:3], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D
        S_ENDPGM,
    ];
    let layout = || vec![root(RootType::Table, 0, 0)];
    let used = || {
        vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        }]
    };
    let _l = lower_with(&words, layout(), used()).unwrap();
    #[cfg(windows)]
    if let Ok(mut v) = crate::emit::validator::Validator::load(
        crate::emit::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut _l.dxil.clone()).unwrap();
    }
}

#[test]
fn a_descriptor_read_only_as_data_takes_the_kind_of_its_entrys_use() {
    // The table entry at s0 loaded twice: the second load's descriptor
    // reads a structured buffer, the first's record count (dword 2) is
    // only read as a number, as the console compiler does for a size.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xF408_0302,
        0xFA00_0000, // s_load_dwordx4 s[12:15], s[4:5], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_2000,
        0x8003_0100, // buffer_load_dword v1, v0, s[12:15], 0 idxen
        0x7E04_020A, // v_mov_b32 v2, s10
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        0xD834_0000,
        0x0000_0200, // ds_write_b32 v0, v2
        S_ENDPGM,
    ];
    let layout = vec![root(RootType::Table, 0, 0)];
    let used = vec![UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    lower_with(&words, layout, used).unwrap();
}

/// Tables of 2D textures at s0 and samplers at s1.
fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let range = |kind, user_data, slot| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
        user_data,
        slot,
    };
    (
        vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
        vec![
            range(ResourceType::Srv, 0, 0),
            range(ResourceType::Sampler, 1, 1),
        ],
    )
}

#[test]
fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
    use crate::resource::tests::{
        WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
    };
    for (words, wave64) in [
        (WATERFALL_FETCHED_WAVE32, false),
        (WATERFALL_FETCHED_WAVE64, true),
        (WATERFALL_PICKED_WAVE32, false),
    ] {
        let (layout, used) = texture_tables();
        let l = lower_wave(words, layout, used, wave64).unwrap();
        // The loop's reads of the first lane became each lane's own
        // dwords: no wave operation is left.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

/// [`texture_tables`] with a raw buffer UAV at dword 16 of the first.
pub(crate) fn texture_and_uav_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let (layout, mut used) = texture_tables();
    used.push(UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 16,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    });
    (layout, used)
}

#[test]
fn waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
    use crate::resource::tests::{WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
    for words in [WATERFALL_JOINED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32] {
        let (layout, used) = texture_and_uav_tables();
        let l = lower_wave(words, layout, used, false).unwrap();
        // One image handle, from the join's per-lane heap index.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

#[test]
fn descriptors_joined_by_nested_control_flow_lower_to_one_uniform_handle() {
    // The view table at s0 (2D textures at entries 0 and 1, a raw UAV at
    // dword 16) and a root constant at s1: entry 0 if s1 is 0, else
    // entry 1 if it is 1, else entry 0 again, the inner if/else joined
    // before the outer one. The join's edges are a load and a join.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBE88_03FF,
        0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_0380, // s_mov_b32 s11, 0
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF84_0003, // s_cbranch_scc0 -> 0x38
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBF82_0008, // s_branch -> 0x58
        0xBF06_8101, // 0x38: s_cmp_eq_u32 s1, 1
        0xBF84_0003, // s_cbranch_scc0 -> 0x4c
        0xF40C_0302,
        0xFA00_0020, // s_load_dwordx8 s[12:19], s[4:5], 0x20
        0xBF82_0002, // s_branch -> 0x54
        0xF40C_0302,
        0xFA00_0000, // 0x4c: s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBE86_0301, // 0x54: s_mov_b32 s6, s1
        0x7E04_02F0, // 0x58: v_mov_b32 v2, 0.5
        0x7E06_02F0, // v_mov_b32 v3, 0.5
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        S_ENDPGM,
    ];
    let (_, mut used) = texture_and_uav_tables();
    used.retain(|u| u.user_data == 0);
    let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    let l = lower_with(&words, layout, used).unwrap();
    // One image handle from the joins' heap index, the same in all lanes.
    assert_eq!(l.nonuniform_handles, 0);
    validate_dxil(&l.dxil);
}

#[test]
fn scalar_bit_tests_lower() {
    // Each tests one bit of a constant into SCC and branches on it (only
    // code with an effect needs its operands).
    let tail = [
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    let bit_tests: [&[u32]; 4] = [
        &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
        &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
        ],
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
        ],
    ];
    for head in bit_tests {
        let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
        lowers(&words, 0).unwrap();
    }
}

#[test]
#[ignore = "ours marks joins read through joins only with VICTORY_STRICT_JOINS (40 programs stop otherwise)"]
fn a_missing_value_stays_missing_through_two_live_joins() {
    let mut words = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85_0001, // skip the first overwrite
        0xBE86_0380, // s_mov_b32 s6, 0
        0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
        0xBF85_0001, // skip the second overwrite
        0xBE86_0381, // s_mov_b32 s6, 1
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // use the result to branch
        0xBE83_0380,
        S_ENDPGM,
    ];
    let group_ids_xy = 3 << 7;
    // Compare to a direct read so future timestamp support works in both cases.
    let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
        Ok(()) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    let direct = [&words[..2], &words[8..]].concat();
    assert_eq!(refusal(&words), refusal(&direct));
    words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
    lowers(&words, group_ids_xy).expect("dead timestamp");
}

#[cfg(windows)]
fn validate_dxil(dxil: &[u8]) {
    if let Ok(mut v) = super::super::validator::Validator::load(
        super::super::validator::Validator::default_path().as_deref(),
    ) {
        let mut copy = dxil.to_vec();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }
}

#[cfg(not(windows))]
fn validate_dxil(_dxil: &[u8]) {}

#[test]
fn a_live_descriptor_word_stays_missing_through_two_joins() {
    let mut words = [
        0xBE84_0300, // s_mov_b32 s4, s0 (table)
        0xBE85_0380, // s_mov_b32 s5, 0
        0x7E00_0280, // v_mov_b32 v0, 0
        0xBF06_8001, // s_cmp_eq_u32 s1, 0 (root constant)
        0xBF85_0002, // take the descriptor arm when SCC is set
        0xBE89_0391, // s_mov_b32 s9, 17
        0xBF82_0006, // skip the descriptor arm
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xE030_2000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        0xBF85_0001, // if SCC is set, keep the descriptor word live
        0xBE89_039D, // s_mov_b32 s9, 29
        0x7E04_0209, // v_mov_b32 v2, s9
        0xD834_0000,
        0x0000_0200, // ds_write_b32 v0, v2
        S_ENDPGM,
    ];
    let run = |words: &[u32]| {
        let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        let used = vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }];
        lower_wave(words, layout, used, false)
    };
    let reason = |words: &[u32]| match run(words) {
        Ok(_) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    let through_two_joins = reason(&words);
    assert_eq!(
        through_two_joins,
        Some("a buffer descriptor's stride or high address read as data")
    );
    words[13] = 0xBF82_0001; // skip the overwrite unconditionally: one join
    assert_eq!(through_two_joins, reason(&words));

    // Correlated overwrite: when SCC is set, always overwrite s9 = 29 before reading it.
    words[13] = 0xBF84_0001; // s_cbranch_scc0 +1 (skip overwrite only when SCC is clear)
    let lowered = run(&words).expect("overwritten descriptor word must lower");
    validate_dxil(&lowered.dxil);
}

#[test]
fn unknown_stride_load_stays_missing_when_live_and_lowers_when_overwritten() {
    let mut words = [
        0xBE84_0300, // s_mov_b32 s4, s0 (table)
        0xBE85_0380, // s_mov_b32 s5, 0
        0x7E00_0280, // v_mov_b32 v0, 0
        0xBF06_8001, // s_cmp_eq_u32 s1, 0 (root constant)
        0xBF85_0002, // take the unknown-stride load arm when SCC is set
        0x7E02_0291, // v_mov_b32 v1, 17
        0xBF82_0004, // skip the load arm
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xE030_2000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen (unknown stride -> Missing)
        0xBF85_0001, // if SCC is set, keep the missing load live
        0x7E02_029D, // v_mov_b32 v1, 29
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let run = |words: &[u32]| {
        let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        let used = vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0, // unknown stride
            dwords: 4,
            user_data: 0,
            slot: 0,
        }];
        lower_wave(words, layout, used, false)
    };
    let reason = |words: &[u32]| match run(words) {
        Ok(_) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    let two_joins = reason(&words);
    assert_eq!(two_joins, Some("a structured buffer of unknown stride"));
    words[11] = 0xBF82_0001; // s_branch +1: one join
    assert_eq!(two_joins, reason(&words));

    // Correlated overwrite: when SCC is set, always overwrite v1 = 29.
    words[11] = 0xBF84_0001; // s_cbranch_scc0 +1
    let lowered = run(&words).expect("overwritten unknown-stride load must lower");
    validate_dxil(&lowered.dxil);
}

#[test]
fn first_half_of_interpolation_stays_missing_when_live_and_lowers_when_overwritten() {
    let mut words = [
        0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x (Missing)
        0xBF06_8000,    // s_cmp_eq_u32 s0, 0
        0xBF84_0001,    // s_cbranch_scc0 +1 (skip p2 when s0 != 0)
        0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
        0xBF85_0001,    // s_cbranch_scc1 +1 (skip overwrite when s0 == 0)
        0x7E04_0280,    // v_mov_b32 v2, 0
        0xF800_180F,
        0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
        S_ENDPGM,
    ];
    let run = |words: &[u32]| {
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![root(RootType::Constants, 0, 0)],
            used: vec![],
            io: StageIo::Ps {
                rsrc2: 0,
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1111,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 1 << 1, // PERSP_CENTER
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, false, 4).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let outputs = vec![super::SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }];
        let inputs = vec![super::SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }];
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs,
            outputs,
            prims: vec![],
            mesh_stride: 0,
            crumbs: None,
        })
    };
    // Overwritten on the s0 != 0 branch: p1 never reaches exp.
    let lowered = run(&words).expect("overwritten interp half must lower");
    validate_dxil(&lowered.dxil);

    // Flip word 4 to s_cbranch_scc0 +1: when s0 != 0, p1 reaches exp live.
    words[4] = 0xBF84_0001;
    let Err(crate::ShaderError::Unsupported { what, .. }) = run(&words) else {
        panic!("live p1 must refuse");
    };
    assert_eq!(what, "the first half of an interpolation");

    // Unconditional skip (1 join): also refuses identically.
    words[4] = 0xBF82_0001;
    let Err(crate::ShaderError::Unsupported { what, .. }) = run(&words) else {
        panic!("1-join live p1 must refuse");
    };
    assert_eq!(what, "the first half of an interpolation");
}

#[test]
fn guarded_write_and_exec_mask_phi_lower_soundly() {
    // 1. EXEC phi fed by a scalar SGPR in one branch, then read via s_bcnt1_i32_b32.
    let exec_phi_words = [
        0xBE84_03BF, // s_mov_b32 s4, 0x3F
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF84_0001, // s_cbranch_scc0 +1
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xBE82_0F7E, // s_bcnt1_i32_b32 s2, exec_lo
        0x7E00_0202, // v_mov_b32 v0, s2
        0xD834_0000,
        0x0000_0000, // ds_write_b32 v0, v0
        S_ENDPGM,
    ];
    let lowered = lower_wave(
        &exec_phi_words,
        vec![root(RootType::Constants, 0, 0)],
        vec![],
        false,
    )
    .expect("EXEC phi read via mask_ok must lower");
    validate_dxil(&lowered.dxil);

    // 2. Guarded write (`write_placed`) preserving `17` on `!guard` lanes where `guard`
    //    lanes held an unknown-stride load (and overwriting `guard` lanes with `29`).
    let guarded_words = [
        0xBE84_0300, // s_mov_b32 s4, s0 (table)
        0xBE85_0380, // s_mov_b32 s5, 0
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0291, // v_mov_b32 v1, 17
        0x7D84_0080, // v_cmp_eq_u32 vcc_lo, 0, v0
        0xBE86_3C6A, // s_and_saveexec_b32 s6, vcc_lo
        0xBF88_0004, // s_cbranch_execz +4
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xE030_2000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen (unknown stride)
        0xBEFE_0306, // s_mov_b32 exec_lo, s6
        0xBE86_3C6A, // s_and_saveexec_b32 s6, vcc_lo
        0x7E02_029D, // v_mov_b32 v1, 29 (guarded by v0 == 0, keeps v1 = 17 on v0 != 0)
        0xBEFE_0306, // s_mov_b32 exec_lo, s6
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    // Lowers on fxceless through their `slot_stride` upload table (the
    // deferred strides port); until it lands, the unknown stride is
    // still refused by name.
    let e = match lower_wave(
        &guarded_words,
        vec![root(RootType::Table, 0, 0)],
        vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        false,
    ) {
        Ok(_) => panic!("the unknown-stride load lowered without slot strides"),
        Err(e) => e,
    };
    assert!(
        e.to_string()
            .contains("a structured buffer of unknown stride"),
        "{e}"
    );
}

#[test]
fn loop_missing_value_refuses_on_back_edge_and_lowers_when_infeasible() {
    // Loop header (word 4): s2 starts at 17, v7 = 0. Inside loop: if s0 == 0, s_memtime -> s2.
    // Latch uses VCC so loop-invariant scc (s0 == 0) survives across iterations.
    // After loop: if s0 == 0, overwrite s2 = 29, else read P_s2.
    let mut words = [
        0x7E00_0280, // 0: v_mov_b32 v0, 0
        0xBE82_0391, // 1: s_mov_b32 s2, 17
        0x7E0E_0280, // 2: v_mov_b32 v7, 0
        0xBF06_8000, // 3: s_cmp_eq_u32 s0, 0 (loop-invariant, preserved in SCC)
        // loop header (word 4):
        0xBF84_0003, // 4: s_cbranch_scc0 +3 (skip s_memtime when s0 != 0)
        0xF490_0180,
        0,           // 5-6: s_memtime s[6:7]
        0xBE82_0306, // 7: s_mov_b32 s2, s6
        // latch (word 8):
        0x4A0E_0E81, // 8: v_add_nc_u32 v7, 1, v7
        0x7D88_0E82, // 9: v_cmp_gt_u32 vcc_lo, 2, v7
        0xBF87_FFF9, // 10: s_cbranch_vccnz -7 (back to word 4)
        // after loop (word 11):
        0xBF84_0001, // 11: s_cbranch_scc0 +1 (skip overwrite when s0 != 0)
        0xBE82_039D, // 12: s_mov_b32 s2, 29
        0x7E02_0202, // 13: v_mov_b32 v1, s2
        0xD834_0000,
        0x0000_0100, // 14-15: ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let run =
        |words: &[u32]| lower_wave(words, vec![root(RootType::Constants, 0, 0)], vec![], false);
    let lowered = run(&words).expect("loop-invariant infeasible missing edge must lower");
    validate_dxil(&lowered.dxil);

    // Flip word 11 to s_cbranch_scc1 +1: s_memtime from back-edge reaches ds_write when s0 == 0.
    words[11] = 0xBF85_0001;
    let Err(crate::ShaderError::Unsupported { what, .. }) = run(&words) else {
        panic!("live loop back-edge missing value must refuse");
    };
    assert_eq!(what, "a timestamp read as data");
}

#[test]
fn old_register_missing_values_and_cached_floats_stay_refused() {
    let srv_used = || {
        vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }]
    };
    let expect_unsupported = |res: Result<Lowered, crate::ShaderError>, want: &str| match res {
        Ok(_) => panic!("expected refusal `{want}`, but lowered"),
        Err(crate::ShaderError::Unsupported { what, .. }) => assert_eq!(what, want),
        Err(error) => panic!("{error}"),
    };

    // 1a. Direct Val::Missing on `old` surviving into `!guard` of `write_placed`.
    let direct_old_guard = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0x7E00_0280, // v_mov_b32 v0, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xE030_2000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen (unknown stride)
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0 (v0 < 16)
        0xBE86_3C6A, // s_and_saveexec_b32 s6, vcc_lo
        0x7E02_029D, // v_mov_b32 v1, 29 (v0 >= 16 keeps old unknown-stride v1)
        0xBEFE_0306, // s_mov_b32 exec_lo, s6
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    expect_unsupported(
        lower_wave(
            &direct_old_guard,
            vec![root(RootType::Table, 0, 0)],
            srv_used(),
            false,
        ),
        "a structured buffer of unknown stride",
    );

    // 1b. Phi Val::Missing on `old` surviving into `!guard` of `write_placed`.
    let phi_old_guard = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0291, // v_mov_b32 v1, 17
        0x7D84_0080, // v_cmp_eq_u32 vcc_lo, 0, v0 (v0 == 0)
        0xBE86_3C6A, // s_and_saveexec_b32 s6, vcc_lo
        0xBF88_0004, // s_cbranch_execz +4
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xE030_2000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
        0xBEFE_0306, // s_mov_b32 exec_lo, s6
        0x7D8A_0080, // v_cmp_ne_u32 vcc_lo, 0, v0 (v0 != 0 -> !guard keeps v0 == 0!)
        0xBE86_3C6A, // s_and_saveexec_b32 s6, vcc_lo
        0x7E02_029D, // v_mov_b32 v1, 29
        0xBEFE_0306, // s_mov_b32 exec_lo, s6
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    expect_unsupported(
        lower_wave(
            &phi_old_guard,
            vec![root(RootType::Table, 0, 0)],
            srv_used(),
            false,
        ),
        "a structured buffer of unknown stride",
    );

    // 1c. Live Val::Missing on `old` kept by `write_16` (`keep == true`).
    let write16_old = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0x7E00_0280, // v_mov_b32 v0, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xE030_2000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
        0x6402_00F2, // v_add_f16 v1, 1.0, v0 (keeps high 16 bits of v1)
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    expect_unsupported(
        lower_wave(
            &write16_old,
            vec![root(RootType::Table, 0, 0)],
            srv_used(),
            false,
        ),
        "a structured buffer of unknown stride",
    );

    // 1d. Live Val::Missing on `old` kept by `place_bits` (SDWA `dst_unused: UNUSED_PRESERVE`).
    let sdwa_old = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0x7E00_0280, // v_mov_b32 v0, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xE030_2000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
        0x7E02_02F9,
        0x0006_1400, // v_mov_b32_sdwa v1, v0 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    expect_unsupported(
        lower_wave(
            &sdwa_old,
            vec![root(RootType::Table, 0, 0)],
            srv_used(),
            false,
        ),
        "a structured buffer of unknown stride",
    );

    // 1e. Direct Val::Missing on `outside` surviving `!guard` of `ds_add_rtn_u32`.
    let ds_rtn_outside = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0x7E00_0280, // v_mov_b32 v0, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xE030_2000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0 (v0 < 16)
        0xBE86_3C6A, // s_and_saveexec_b32 s6, vcc_lo
        0xD880_0000,
        0x0100_0000, // ds_add_rtn_u32 v1, v0, v0 (v0 >= 16 keeps outside = unknown-stride v1)
        0xBEFE_0306, // s_mov_b32 exec_lo, s6
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    expect_unsupported(
        lower_wave(
            &ds_rtn_outside,
            vec![root(RootType::Table, 0, 0)],
            srv_used(),
            false,
        ),
        "a structured buffer of unknown stride",
    );

    // 2. Live missing phi flowing through chained float ops (`self.floats` cache hit) to `ds_write`.
    let chained_float_phi = [
        0x7E00_0280, // v_mov_b32 v0, 0
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0002, // take s_memtime arm when s0 == 0
        0xBE82_0391, // s_mov_b32 s2, 17
        0xBF82_0003, // skip s_memtime
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xBE82_0306, // s_mov_b32 s2, s6
        0x0604_0002, // v_add_f32 v2, s2, v0 (inserts v2 into self.floats)
        0x0A06_04F4, // v_mul_f32 v3, 2.0, v2 (hits self.floats[&v2], inserts v3 into self.floats)
        0xD834_0000,
        0x0000_0300, // ds_write_b32 v0, v3
        S_ENDPGM,
    ];
    expect_unsupported(
        lower_wave(
            &chained_float_phi,
            vec![root(RootType::Constants, 0, 0)],
            vec![],
            false,
        ),
        "a timestamp read as data",
    );
}

/// The names `words` decode to, to check hand-written encodings.
fn names(words: &[u32]) -> Vec<&'static str> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, false, 256).unwrap();
    p.blocks
        .iter()
        .flat_map(|b| b.insts.iter().map(|i| i.name()))
        .collect()
}

/// The reason a program stops, or `None` if it translates.
fn refusal(r: Result<Lowered, crate::ShaderError>) -> Option<&'static str> {
    match r {
        Ok(_) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    }
}

/// A descriptor table at s0 whose first entry is a buffer of `kind`
/// and `stride`.
fn buffer_table(kind: ResourceType, stride: u32) -> (Vec<RootParam>, Vec<UsedResource>) {
    let used = UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    (vec![root(RootType::Table, 0, 0)], vec![used])
}

#[test]
fn scalar_loaded_buffer_descriptors_read_as_data_name_the_dword_and_kind() {
    // A buffer descriptor that scalar buffer loads read through, one
    // of whose dwords is also stored as a number.
    let program = |word: u32| {
        vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xF420_0304,
            0xFA00_0000, // s_buffer_load_dword s12, s[8:11], null
            word,        // v_mov_b32 v1, s10 (dword 2) or s8 (dword 0)
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ]
    };
    let (dword2, dword0) = (0x7E02_020A, 0x7E02_0208);
    assert_eq!(
        names(&program(dword2)),
        [
            "s_mov_b32",
            "s_mov_b32",
            "s_load_dwordx4",
            "s_buffer_load_dword",
            "v_mov_b32",
            "ds_write_b32",
            "s_endpgm"
        ]
    );
    let cases = [
        (
            ResourceType::Srv,
            0,
            "a scalar-loaded raw buffer's record count read as data",
        ),
        (
            ResourceType::Srv,
            16,
            "a scalar-loaded structured buffer's record count read as data",
        ),
        (
            ResourceType::Cbv,
            0,
            "a scalar-loaded constant buffer's record count read as data",
        ),
        (
            ResourceType::Uav,
            0,
            "a scalar-loaded buffer's record count read as data",
        ),
    ];
    for (kind, stride, want) in cases {
        let (layout, used) = buffer_table(kind, stride);
        let got = refusal(lower_with(&program(dword2), layout.clone(), used.clone()));
        assert_eq!(got, Some(want), "{kind:?}, stride {stride}");
        assert_eq!(
            refusal(lower_with(&program(dword0), layout, used)),
            Some("a scalar-loaded buffer's address or format read as data"),
            "{kind:?}, stride {stride}"
        );
    }
    // The same descriptor also read by a vector load later: the record
    // count has a host value once that use decides the view, so the
    // scalar load's dwords lower too.
    let mut words = program(dword2);
    words.splice(
        6..6,
        [
            0xE030_0000,
            0x8002_0100, // buffer_load_dword v1, off, s[8:11], 0
        ],
    );
    assert!(names(&words).contains(&"buffer_load_dword"));
    let (layout, used) = buffer_table(ResourceType::Srv, 0);
    assert!(lower_with(&words, layout, used).is_ok());
}

#[test]
fn scalar_memory_that_is_no_descriptor_names_what_it_reads() {
    // Each value stored to LDS, so that the program needs it.
    let program = |op: [u32; 2], sgpr: u32| {
        vec![
            op[0],
            op[1],
            0x7E02_0200 | sgpr, // v_mov_b32 v1, s<sgpr>
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ]
    };
    let cases = [
        ([0xF490_0180, 0], "s_memtime", "a timestamp read as data"),
        (
            [0xF494_0180, 0],
            "s_memrealtime",
            "a timestamp read as data",
        ),
        (
            [0xF4A8_0180, 0],
            "s_get_waveid_in_workgroup",
            "a wave's index in its group read as data",
        ),
    ];
    for (op, name, want) in cases {
        let words = program(op, 6);
        assert_eq!(names(&words)[0], name);
        assert_eq!(
            refusal(lower_with(&words, vec![], vec![])),
            Some(want),
            "{name}"
        );
    }
    // A load through a root view's address (s[0:1]) reads that
    // buffer's memory (`root_view_smem` lowers it); one from a table
    // with no use reads a descriptor.
    let root_view = program([0xF400_0180, 0xFA00_0000], 6); // s_load_dword s6, s[0:1], null
    assert_eq!(names(&root_view)[0], "s_load_dword");
    lower_with(&root_view, vec![root(RootType::Descriptor, 0, 0)], vec![])
        .expect("a root view's memory read by scalar loads lowers");
    // A table load with no resource use reads the bound descriptor
    // words (ray tracing reads them for GPU addresses).
    let mut table = vec![0xBE84_0300, 0xBE85_0380]; // s_mov_b32 s4, s0; s_mov_b32 s5, 0
    table.extend(program([0xF400_0182, 0xFA00_0000], 6)); // s_load_dword s6, s[4:5], null
    assert_eq!(names(&table)[2], "s_load_dword");
    lower_with(&table, vec![root(RootType::Table, 0, 0)], vec![])
        .expect("a descriptor read only as data gets the bound words");
}

#[test]
fn descriptor_dwords_read_as_data_name_what_they_hold() {
    // An image descriptor that a sample uses and a buffer descriptor
    // that a load uses, one dword of each stored as a number. Dwords
    // the translation rebuilds from the host view translate.
    let image = |sgpr: u32| {
        vec![
            0xBE82_0300, // s_mov_b32 s2, s0
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF40C_0301,
            0xFA00_0000, // s_load_dwordx8 s[12:19], s[2:3], null
            0xBE84_0301, // s_mov_b32 s4, s1
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0x7E04_0280, // v_mov_b32 v2, 0
            0x7E06_0280, // v_mov_b32 v3, 0
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:2D
            0x7E02_0200 | sgpr, // v_mov_b32 v1, s<sgpr>
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ]
    };
    assert_eq!(names(&image(12))[8], "image_sample_lz");
    for (dword, want) in [
        (0, Some("an image descriptor's address read as data")),
        (2, None),
        (
            5,
            Some("an image descriptor's array pitch or mip limits read as data"),
        ),
        (
            6,
            Some("an image descriptor's compression fields read as data"),
        ),
        (
            7,
            Some("an image descriptor's metadata address read as data"),
        ),
    ] {
        let (layout, used) = texture_tables();
        let got = refusal(lower_with(&image(12 + dword), layout, used));
        assert_eq!(got, want, "image dword {dword}");
    }
    let buffer = |sgpr: u32| {
        vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xE030_0000,
            0x8002_0100,        // buffer_load_dword v1, off, s[8:11], 0
            0x7E02_0200 | sgpr, // v_mov_b32 v1, s<sgpr>
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ]
    };
    assert_eq!(names(&buffer(8))[3], "buffer_load_dword");
    for (dword, want) in [
        (0, Some("a buffer descriptor's address read as data")),
        (
            1,
            Some("a buffer descriptor's stride or high address read as data"),
        ),
        (2, None),
        (
            3,
            Some("a buffer descriptor's format or swizzle read as data"),
        ),
    ] {
        let (layout, used) = buffer_table(ResourceType::Srv, 0);
        let got = refusal(lower_with(&buffer(8 + dword), layout, used));
        assert_eq!(got, want, "buffer dword {dword}");
    }
}

#[test]
fn buffer_formats_by_number() {
    // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
    // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
    assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
    assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
    assert_eq!(
        buffer_format(50),
        Some((&[10, 10, 10, 2][..], Numeric::Unorm))
    );
    assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
    assert_eq!(
        buffer_format(77),
        Some((&[32, 32, 32, 32][..], Numeric::Float))
    );
    // Invalid, past the table.
    assert_eq!(buffer_format(0), None);
    assert_eq!(buffer_format(78), None);
}

#[test]
fn packed_buffer_formats_by_number() {
    // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
    // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
    // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
    assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
    assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
    assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
    assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
    assert_eq!(
        buffer_format(44),
        Some((&[2, 10, 10, 10][..], Numeric::Unorm))
    );
    assert_eq!(
        buffer_format(45),
        Some((&[2, 10, 10, 10][..], Numeric::Snorm))
    );
    assert_eq!(
        buffer_format(46),
        Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
    );
    assert_eq!(
        buffer_format(47),
        Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
    );
    assert_eq!(
        buffer_format(48),
        Some((&[2, 10, 10, 10][..], Numeric::Uint))
    );
    assert_eq!(
        buffer_format(49),
        Some((&[2, 10, 10, 10][..], Numeric::Sint))
    );
}

/// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
/// `legacyF16ToF32` widens it.
fn half(h: u32) -> f32 {
    let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
    let v = match e {
        0 => m as f32 * 2f32.powi(-24),
        31 if m == 0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
    };
    if s == 1 { -v } else { v }
}

#[test]
fn small_floats_widen_as_halves() {
    // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
    // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
    let f11 = |x: u32| half(x << half_shift(11));
    let f10 = |x: u32| half(x << half_shift(10));
    assert_eq!(f11(0), 0.0);
    assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
    assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
    assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
    assert_eq!(f11(0x3C0), 1.0);
    assert_eq!(f11(0x3E0), 1.5);
    assert_eq!(f11(0x7BF), 65024.0); // largest
    assert_eq!(f11(0x7C0), f32::INFINITY);
    assert!(f11(0x7C1).is_nan());
    assert_eq!(f10(0x001), 2f32.powi(-19));
    assert_eq!(f10(0x020), 2f32.powi(-14));
    assert_eq!(f10(0x1E0), 1.0);
    assert_eq!(f10(0x3DF), 64512.0);
    assert_eq!(f10(0x3E0), f32::INFINITY);
    assert!(f10(0x3FF).is_nan());
    // A 16-bit float is a half already, sign included.
    assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
}

#[test]
fn constant_64_bit_compares_lower_without_int64() {
    for (vopc, branch) in [
        (0x7DC0_0080u32, 0xBF86_0001u32), // v_cmp_f_u64 vcc, ...; s_cbranch_vccnz
        (0x7D4E_0080, 0xBF86_0001),       // v_cmp_t_i64 vcc, ...; s_cbranch_vccnz
        (0x7DE0_0080, 0xBF89_0001),       // v_cmpx_f_u64 exec, ...; s_cbranch_execnz
        (0x7D6E_0080, 0xBF89_0001),       // v_cmpx_t_i64 exec, ...; s_cbranch_execnz
    ] {
        let words = [
            0x7E00_0280u32, // v_mov_b32 v0, 0
            0x7E02_0280,    // v_mov_b32 v1, 0
            vopc,
            branch,
            0xBE80_0380, // s_mov_b32 s0, 0
            S_ENDPGM,
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, 2).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let emitted = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            crumbs: None,
        })
        .unwrap();
        let sfi0 = crate::dxbc::Container::parse(&emitted.dxil)
            .unwrap()
            .part(crate::dxbc::FourCC::SFI0)
            .unwrap()
            .data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_eq!(features & 0x8000, 0, "unexpected INT64 feature bit");
    }
}

#[test]
fn permlane16_lowers() {
    let words = [
        // v_permlanex16_b32 v2, v0, s0, s1 (VOP3 opcode 0x378)
        0xD778_0002,
        (1 << 18) | 0x100,
        S_ENDPGM,
    ];
    lowers(&words, 0).expect("v_permlanex16_b32 should lower");
}
