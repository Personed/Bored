// Tests for where a program's constant table starts.

use crate::{ir, resource};

const S_ENDPGM: u32 = 0xBF81_0000;

/// A program's constant table starts after its instructions, unless a
/// load at a fixed offset reads below that.
#[test]
fn the_code_table_starts_after_the_instructions() {
    // s_nop 0, s_endpgm, then 1,000 dwords of constants.
    let mut words = vec![0xBF80_0000, S_ENDPGM];
    words.extend(0..1000u32);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, false, 256).unwrap();
    let at = |offset: u32, dynamic: bool| resource::Address {
        param: resource::CODE,
        offset,
        dynamic: dynamic.then_some(ir::Value(0)),
    };
    let (computed, fixed_after, fixed_inside) = (at(0, true), at(12, false), at(4, false));
    assert_eq!(code_table_start(&p, &[&computed], 0), 8);
    assert_eq!(code_table_start(&p, &[&computed, &fixed_after], 0), 8);
    assert_eq!(code_table_start(&p, &[&computed, &fixed_inside], 0), 0);
    assert_eq!(code_table_start(&p, &[&fixed_after], 12), 12);
}
