/// Where a program's constant table starts in its code: where its
/// instructions end, unless a load at a fixed offset reads below that (then
/// at `lowest`, the lowest offset any load or descriptor names). The
/// program's constants follow its instructions, and a load at a computed
/// offset reads them, not instruction bytes; a table from the first
/// computed load's offset 0 was the whole program, tens of KiB, which
/// Vulkan drivers can keep in every thread's local memory.
fn code_table_start(
    p: &crate::ir::Program,
    addresses: &[&crate::resource::Address],
    lowest: u32,
) -> u32 {
    let code_end = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .map(|i| i.machine.offset + 4 * i.machine.words.len())
        .max()
        .unwrap_or(0) as u32;
    let fixed_below = addresses
        .iter()
        .any(|a| a.dynamic.is_none() && a.offset < code_end);
    if fixed_below {
        lowest
    } else {
        lowest.max(code_end & !3)
    }
}
