//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_PRESERVE_PLACEHOLDER, MEM_PRIVATE, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, PERFORMANCE_INFORMATION,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

/// Bytes of title memory committed: physical pages allocated and not yet
/// freed, plus `MEM_COMMIT` virtual allocations. This is what
/// `XMemGetWorkingSetStatistics` reports as the title's use: counting the
/// whole process instead had the layer's own memory (copy-back buffers,
/// host heaps, the card's driver allocations) in the figure — 27 GB in
/// run 60's gameplay against a limit of 13.5 GB — and a title budgeting
/// its streaming from it believed it was far over its limit. Heap blocks
/// (`XMemAlloc`) stay uncounted: the title hooks them into arenas of its
/// own, which are physical pages or virtual allocations already here.
static COMMITTED: AtomicUsize = AtomicUsize::new(0);

/// Sizes of committed virtual allocations by address, so `XMemFreeDefault`
/// (the 32-bit path frees whole reservations) can uncount them.
static COMMITTED_VIRTUAL: LazyLock<Mutex<HashMap<usize, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;
/// Coalesces adjacent placeholders into one (`MEM_COALESCE_PLACEHOLDERS`,
/// not in windows-sys' memory module).
const MEM_COALESCE_PLACEHOLDERS: u32 = 0x1;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (
/// the title's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (the title's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                &mut parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    if !at.is_null()
        && placeholder
        && size > 0
        && let Some(aligned) = size.checked_next_multiple_of(PAGE)
    {
        record_placeholder_reservation(at as usize, aligned);
    }
    if !at.is_null() && !placeholder && kind & MEM_COMMIT != 0 {
        let previous = COMMITTED_VIRTUAL
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(at as usize, size);
        // A recommit of the same range replaces its charge rather than
        // adding to it.
        if let Some(old) = previous {
            let _ = COMMITTED.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |c| {
                Some(c.saturating_sub(old))
            });
        }
        COMMITTED.fetch_add(size, Ordering::Relaxed);
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (the title's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (the title does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (the title uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            if let Some(size) = COMMITTED_VIRTUAL
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&(address as usize))
            {
                COMMITTED.fetch_sub(size, Ordering::Relaxed);
            }
            VirtualFree(address, 0, MEM_RELEASE);
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;

struct Section(HANDLE);

// SAFETY: a section handle may be used from any thread.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Drop for Section {
    fn drop(&mut self) {
        victory_gpu::title_memory_section_gone(self.0 as usize);
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.0) };
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

impl Slice {
    /// The key in `Physical::shown` of the page `k` pages after this one.
    fn key(&self, k: usize) -> (usize, usize) {
        (Arc::as_ptr(&self.section) as usize, self.index + k)
    }
}

#[derive(Default)]
struct Physical {
    next_page: usize,
    pages: HashMap<usize, Slice>,
    /// Original placeholder reservations (`base -> len`) from `XMemVirtualAlloc`.
    /// Coalescing never crosses a reservation boundary.
    placeholders: BTreeMap<usize, usize>,
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
    /// Where each page shows: by its section and its index there, the
    /// address of that page in every view that shows it now ([`shown_at`];
    /// as upstream's c75fdba keeps it). The title maps pages where the
    /// card reads them and has them filled afterwards by DirectStorage,
    /// which writes through views of our own.
    shown: HashMap<(usize, usize), Vec<usize>>,
}

static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(|| {
    Mutex::new(Physical {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        ..Default::default()
    })
});

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    let Some(size) = n.checked_mul(PAGE).filter(|&s| s > 0) else {
        return 0;
    };
    // A pagefile-backed section of `size` bytes, its pages committed when
    // first mapped or written (`commit`): the title asks for 6.1
    // GiB at once, which a busy host often cannot commit (error 1450).
    // SAFETY: plain section creation.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            // Under Wine a committed section is a sparse file, free until
            // touched, and committing a reserved one page by page costs a
            // wineserver request each that scans every range committed so
            // far: commit it whole there.
            PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
            (size as u64 >> 32) as u32,
            size as u32,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        let error = std::io::Error::last_os_error();
        tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    }
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    // For the GPU side's in-place census: only a committed section can be
    // opened as a host heap.
    victory_gpu::title_memory_section(handle as usize, size as u64, on_wine());
    COMMITTED.fetch_add(size, Ordering::Relaxed);
    let section = Arc::new(Section(handle));
    let mut physical = physical();
    let first = physical.next_page;
    physical.next_page += n * FRAMES;
    for index in 0..n {
        let slice = Slice {
            section: section.clone(),
            index,
        };
        let number = first + index * FRAMES;
        physical.pages.insert(number, slice);
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(number) };
    }
    1
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: the title keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut physical = physical();
    let mut freed = 0;
    for i in 0..n {
        // SAFETY: the title passes `n` page numbers.
        if physical.pages.remove(&unsafe { *pages.add(i) }).is_some() {
            freed += PAGE;
        }
    }
    // Saturates: a double free must not wrap the title's figure around.
    let _ = COMMITTED.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |c| {
        Some(c.saturating_sub(freed))
    });
    1
}

/// A view of our own on consecutive slices of one section, committed and
/// writable, unmapped when dropped.
struct View {
    at: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

impl View {
    /// Maps `n` slices from `slice` on.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe {
            MapViewOfFile(
                slice.section.0,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if at.Value.is_null() {
            return None;
        }
        let view = View { at, len };
        commit(at.Value as usize, len).then_some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the view holds `len` committed read-write bytes, ours
        // alone while `self` lives (other views of the slices only alias
        // them, as the console's do).
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(self.at) };
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let physical = physical();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Whether every page under `[offset, offset + len)` is known (a write that
/// fails with them known is a host failure).
pub(crate) fn pages_known(pages: &[usize], offset: usize, len: usize) -> bool {
    runs(pages, offset, len).is_some()
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// One view per run of consecutive pages, not one per page: each view
/// costs a map, a commit and an unmap (which flushes every core's TLB).
/// False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let Some(mut view) = View::map(slice, *n) else {
            return false;
        };
        let k = (n * PAGE - within).min(data.len() - done);
        view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
        (within, done) = (0, done + k);
    }
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    view: View,
    within: usize,
    len: usize,
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        &mut self.view.bytes()[within..within + len]
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    Some(PageWindow {
        view: View::map(slice, *n)?,
        within: offset % PAGE,
        len,
    })
}

/// The title addresses that show bytes `[offset, offset + len)` of `pages`
/// now, byte for byte, as runs (address, bytes) in address order: a write
/// to the pages through a view of our own ([`page_window`],
/// [`write_physical`]) lands there too, and nothing that watches those
/// addresses sees it (the GPU side's guards fault only on a write at the
/// guarded address itself). Empty if no view shows them. Byte for byte,
/// not by whole pages: the title packs several small textures into one
/// page and fills each with a request of its own.
pub(crate) fn shown_at(pages: &[usize], offset: usize, len: usize) -> Vec<(usize, usize)> {
    let Some(end) = offset.checked_add(len) else {
        return Vec::new();
    };
    let first = offset / PAGE;
    let Some(touched) = pages.get(first..end.div_ceil(PAGE)) else {
        return Vec::new();
    };
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    {
        let physical = physical();
        for (i, page) in touched.iter().enumerate() {
            let Some(addresses) = physical
                .pages
                .get(page)
                .and_then(|slice| physical.shown.get(&slice.key(0)))
            else {
                continue;
            };
            // The bytes of this page that the write covers.
            let lo = if i == 0 { offset % PAGE } else { 0 };
            let hi = (end - (first + i) * PAGE).min(PAGE);
            pieces.extend(addresses.iter().map(|&at| (at + lo, hi - lo)));
        }
    }
    pieces.sort_unstable();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (at, bytes) in pieces {
        match runs.last_mut() {
            Some((start, n)) if *start + *n == at => *n += bytes,
            _ => runs.push((at, bytes)),
        }
    }
    runs
}

/// Bytes `[offset, offset + len)` of `pages` have been written through a
/// view of our own: the GPU side is told at every title address that
/// shows them (`victory_gpu::title_memory_written`; what it does with it
/// is `VICTORY_PAGE_WRITES`' to say, and it counts either way). Call when
/// the bytes are there, and without the page table's lock ([`shown_at`]
/// takes and leaves it).
pub(crate) fn pages_written(pages: &[usize], offset: usize, len: usize) {
    for (at, bytes) in shown_at(pages, offset, len) {
        victory_gpu::title_memory_written(at, bytes, true);
    }
}

fn record_placeholder_reservation(base: usize, len: usize) {
    let Some(end) = base.checked_add(len) else {
        return;
    };
    let mut physical = physical();
    let first = physical
        .placeholders
        .range(..=base)
        .next_back()
        .filter(|(at, size)| *at + *size > base)
        .map(|(at, _)| *at);
    let stale: Vec<usize> = first
        .into_iter()
        .chain(
            physical
                .placeholders
                .range(base.saturating_add(1)..end)
                .map(|(at, _)| *at),
        )
        .collect();
    for at in stale {
        physical.placeholders.remove(&at);
    }
    physical.placeholders.insert(base, len);
}

fn reservation_bounds(physical: &Physical, addr: usize) -> Option<(usize, usize)> {
    physical
        .placeholders
        .range(..=addr)
        .next_back()
        .filter(|&(base, len)| addr - base < *len)
        .map(|(&base, &len)| (base, base + len))
}

fn same_reservation(physical: &Physical, a: usize, b: usize) -> bool {
    matches!(
        (reservation_bounds(physical, a), reservation_bounds(physical, b)),
        (Some(ra), Some(rb)) if ra == rb
    )
}

fn page_slice_at(physical: &Physical, addr: usize) -> Option<Slice> {
    physical
        .views
        .range(..=addr)
        .next_back()
        .filter(|(base, (size, _))| addr - *base < *size)
        .map(|(base, (_, s))| Slice {
            index: s.index + (addr - base) / PAGE,
            section: s.section.clone(),
        })
}

fn has_any_view(physical: &Physical, base: usize, len: usize) -> bool {
    let Some(end) = base.checked_add(len) else {
        return false;
    };
    if len == 0 {
        return false;
    }
    if physical
        .views
        .range(..=base)
        .next_back()
        .is_some_and(|(at, (size, _))| *at + *size > base)
    {
        return true;
    }
    physical
        .views
        .range(base.saturating_add(1)..end)
        .next()
        .is_some()
}

fn overlapping_views(physical: &Physical, base: usize, len: usize) -> Vec<(usize, usize, Slice)> {
    let Some(end) = base.checked_add(len) else {
        return Vec::new();
    };
    if len == 0 {
        return Vec::new();
    }
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + *size > base)
        .map(|(at, (size, s))| (*at, *size, s));
    first
        .into_iter()
        .chain(
            physical
                .views
                .range(base.saturating_add(1)..end)
                .map(|(at, (size, s))| (*at, *size, s)),
        )
        .filter_map(|(at, size, s)| {
            let clip_start = at.max(base);
            let clip_end = (at + size).min(end);
            (clip_end > clip_start).then(|| {
                (
                    clip_start,
                    clip_end - clip_start,
                    Slice {
                        index: s.index + (clip_start - at) / PAGE,
                        section: s.section.clone(),
                    },
                )
            })
        })
        .collect()
}

fn map_physical_pages_locked(
    physical: &mut Physical,
    base: usize,
    n: usize,
    pages: *const usize,
) -> bool {
    if n == 0 {
        return true;
    }
    let Some(len) = n.checked_mul(PAGE) else {
        return false;
    };
    let page_slice = (!pages.is_null()).then(|| {
        // SAFETY: the title passes `n` page numbers.
        unsafe { std::slice::from_raw_parts(pages, n) }
    });
    if let Some(ps) = page_slice {
        for &page in ps {
            if !physical.pages.contains_key(&page) {
                tracing::warn!(page, "XMemMapPhysicalPages: unknown page");
                return false;
            }
        }
    }
    if has_any_view(physical, base, len) {
        unmap(physical, base, len);
    }
    let Some(pages) = page_slice else {
        return true;
    };
    let mut i = 0;
    while i < n {
        let first = physical.pages[&pages[i]].clone();
        // One view per run of consecutive slices of one section.
        let run = 1 + pages[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(page, k)| {
                let s = &physical.pages[page];
                Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k
            })
            .count();
        if !map(physical, base + i * PAGE, run * PAGE, first) {
            return false;
        }
        i += run;
    }
    true
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    if n == 0 {
        return address;
    }
    let base = address as usize;
    let mut physical = physical();
    // Until the pages are mapped again: `unmap` takes whole views away and
    // maps what stays of them again, and a mapped view is a reservation
    // until it is committed. A read of title memory in place must not fall
    // into that gap (`victory_gpu::title_memory_remapping`).
    let _changing = victory_gpu::title_memory_remapping();
    if map_physical_pages_locked(&mut physical, base, n, pages) {
        address
    } else {
        std::ptr::null_mut()
    }
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (the title passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    if n == 0 {
        return 1;
    }
    // SAFETY: the title passes `n` addresses and, unless null, `n` page numbers.
    let addrs = unsafe { std::slice::from_raw_parts(addresses, n) };
    let page_slice = (!pages.is_null()).then(|| {
        // SAFETY: `pages` is non-null and holds `n` page numbers.
        unsafe { std::slice::from_raw_parts(pages, n) }
    });
    let mut physical = physical();
    // Hold the remapping guard once across the whole batch so readers never
    // meet an unmapped gap and lock overhead stays O(1) per scatter call.
    let _changing = victory_gpu::title_memory_remapping();
    let mut i = 0;
    while i < n {
        let base = addrs[i] as usize;
        if page_slice.is_none_or(|ps| ps[i] == 0) {
            let mut run = 1;
            while i + run < n
                && base.checked_add(run * PAGE) == Some(addrs[i + run] as usize)
                && same_reservation(&physical, base, addrs[i + run] as usize)
                && page_slice.is_none_or(|ps| ps[i + run] == 0)
            {
                run += 1;
            }
            if has_any_view(&physical, base, run * PAGE) {
                unmap(&mut physical, base, run * PAGE);
            }
            i += run;
            continue;
        }
        let ps = page_slice.expect("checked non-null");
        let Some(first) = physical.pages.get(&ps[i]).cloned() else {
            tracing::warn!(page = ps[i], "XMemMapPhysicalPagesScatter: unknown page");
            return 0;
        };
        // Only coalesce contiguous addresses within the same reservation whose
        // physical pages are consecutive slices of one section; otherwise a
        // failure on entry `i` would disturb later independent entries.
        let mut run = 1;
        while i + run < n
            && base.checked_add(run * PAGE) == Some(addrs[i + run] as usize)
            && same_reservation(&physical, base, addrs[i + run] as usize)
            && ps[i + run] != 0
            && physical.pages.get(&ps[i + run]).is_some_and(|s| {
                Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + run
            })
        {
            run += 1;
        }
        let len = run * PAGE;
        let saved = if run > 1 && has_any_view(&physical, base, len) {
            overlapping_views(&physical, base, len)
        } else {
            Vec::new()
        };
        if !saved.is_empty() || has_any_view(&physical, base, len) {
            unmap(&mut physical, base, len);
        }
        if !map(&mut physical, base, len, first) {
            if !saved.is_empty() {
                if has_any_view(&physical, base, len) {
                    unmap(&mut physical, base, len);
                }
                for (at, view_len, slice) in saved {
                    let _ = map(&mut physical, at, view_len, slice);
                }
            }
            return 0;
        }
        i += run;
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`):
/// the title maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Coalesces consecutive source slices into multi-page views within
/// one reservation, skips unchanged mappings, and returns false without
/// unmapping `to` if any source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    if len == 0 {
        return true;
    }
    let (to, from, len) = (to as usize, from as usize, len as usize);
    let pages = len.div_ceil(PAGE);
    let Some(mapped_len) = pages.checked_mul(PAGE) else {
        return false;
    };
    if to.checked_add(mapped_len).is_none() {
        return false;
    }
    let mut physical = physical();
    // As in `XMemMapPhysicalPages`: no read in place meets the gap.
    let _changing = victory_gpu::title_memory_remapping();
    if from == 0 {
        if has_any_view(&physical, to, mapped_len) {
            unmap(&mut physical, to, mapped_len);
        }
        return true;
    }
    if from.checked_add(mapped_len).is_none() {
        return false;
    }
    let mut runs: Vec<(usize, usize, Slice)> = Vec::with_capacity(1);
    let mut unchanged = true;
    for k in 0..pages {
        let Some(src) = page_slice_at(&physical, from + k * PAGE) else {
            return false;
        };
        let dst_addr = to + k * PAGE;
        if unchanged {
            unchanged = page_slice_at(&physical, dst_addr).is_some_and(|dst| {
                Arc::ptr_eq(&dst.section, &src.section) && dst.index == src.index
            });
        }
        match runs.last_mut() {
            Some((run_start, run_pages, first))
                if Arc::ptr_eq(&first.section, &src.section)
                    && src.index == first.index + *run_pages
                    && same_reservation(&physical, *run_start, dst_addr) =>
            {
                *run_pages += 1;
            }
            _ => runs.push((dst_addr, 1, src)),
        }
    }
    if unchanged {
        return true;
    }
    if has_any_view(&physical, to, mapped_len) {
        unmap(&mut physical, to, mapped_len);
    }
    for (dst, run_pages, first) in runs {
        if !map(&mut physical, dst, run_pages * PAGE, first) {
            return false;
        }
    }
    true
}

fn query_region(addr: usize) -> Option<MEMORY_BASIC_INFORMATION> {
    // SAFETY: plain data; zero is a valid initial value.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: `info` points to `size_of::<MEMORY_BASIC_INFORMATION>()` writable bytes.
    let queried = unsafe {
        VirtualQueryEx(
            GetCurrentProcess(),
            addr as *const c_void,
            (&raw mut info).cast(),
            size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };
    (queried != 0).then_some(info)
}

/// Prepares `[base, base + len)` (or its maximal contiguous reserved prefix in
/// `bounds`) as a single placeholder ready for
/// `MapViewOfFile3(..., MEM_REPLACE_PLACEHOLDER)`. When reservation provenance
/// (`bounds`) is unknown, never coalesces across separate placeholders.
fn prepare_placeholder(base: usize, len: usize, bounds: Option<(usize, usize)>) -> usize {
    let Some(end) = base.checked_add(len) else {
        return 0;
    };
    if len == 0 {
        return 0;
    }
    let Some(first) = query_region(base) else {
        return 0;
    };
    if first.State != MEM_RESERVE || first.Type != MEM_PRIVATE || first.RegionSize == 0 {
        return 0;
    }
    // `VirtualQueryEx` sets `BaseAddress` to the queried page and
    // `AllocationBase` to the start of the placeholder itself.
    let first_base = first.AllocationBase as usize;
    let Some(first_end) = (first.BaseAddress as usize).checked_add(first.RegionSize) else {
        return 0;
    };
    if first_base > base || first_end <= base {
        return 0;
    }
    if first_base <= base && first_end >= end {
        if first_base < base || first_end > end {
            // SAFETY: splits the single placeholder around `[base, end)`.
            unsafe {
                VirtualFree(
                    base as *mut c_void,
                    len,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
            }
        }
        return len;
    }
    let Some((res_base, res_end)) = bounds else {
        // Without reservation provenance, split and map only the first placeholder
        // so separate adjacent reservations are never merged on Wine or Windows.
        let avail = first_end.min(end);
        if first_base < base || first_end > avail {
            // SAFETY: splits `[first_base, first_end)` around `[base, avail)`.
            unsafe {
                VirtualFree(
                    base as *mut c_void,
                    avail - base,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
            }
        }
        return avail - base;
    };
    let limit = end.min(res_end);
    let mut cur = first_end;
    let mut last_base = first_base;
    let mut last_end = first_end;
    while cur < limit {
        let Some(mbi) = query_region(cur) else {
            break;
        };
        if mbi.State != MEM_RESERVE || mbi.Type != MEM_PRIVATE || mbi.RegionSize == 0 {
            break;
        }
        let region_start = mbi.AllocationBase as usize;
        let Some(next_end) = (mbi.BaseAddress as usize).checked_add(mbi.RegionSize) else {
            break;
        };
        if region_start < res_base || next_end <= cur {
            break;
        }
        last_base = region_start;
        last_end = next_end;
        cur = next_end;
    }
    let avail = cur.min(limit);
    if avail <= first_end {
        if first_base < base || first_end > avail {
            // SAFETY: splits `[first_base, first_end)` around `[base, avail)`.
            unsafe {
                VirtualFree(
                    base as *mut c_void,
                    avail - base,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
            }
        }
        return avail - base;
    }
    if first_base < base {
        // SAFETY: splits `[first_base, first_end)` at `base` so `[base, first_end)`
        // begins on an exact placeholder boundary.
        unsafe {
            VirtualFree(
                base as *mut c_void,
                first_end - base,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
        }
    }
    if last_end > avail && last_base < avail {
        // SAFETY: splits `[last_base, last_end)` at `avail` so `[last_base, avail)`
        // ends on an exact placeholder boundary.
        unsafe {
            VirtualFree(
                last_base as *mut c_void,
                avail - last_base,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
        }
    }
    // SAFETY: `[base, avail)` spans adjacent placeholders aligned to exact
    // placeholder boundaries within the same reservation `res_base..res_end`.
    let coalesced = unsafe {
        VirtualFree(
            base as *mut c_void,
            avail - base,
            MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS,
        )
    };
    if coalesced != 0 {
        avail - base
    } else {
        first_end - base
    }
}

/// Maps `slice` and the slices after it over `[base, base + len)`,
/// coalescing fragmented placeholders in `[base, base + len)` when needed.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    let mut done = 0;
    while done < len {
        let bounds = reservation_bounds(physical, base + done);
        let prepared = prepare_placeholder(base + done, len - done, bounds);
        let (chunk, is_prepared) = if prepared >= PAGE && prepared.is_multiple_of(PAGE) {
            (prepared, true)
        } else {
            (len - done, false)
        };
        let chunk_slice = Slice {
            index: slice.index + done / PAGE,
            section: slice.section.clone(),
        };
        if !map_one(physical, base + done, chunk, chunk_slice, is_prepared) {
            return false;
        }
        done += chunk;
    }
    true
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder.
fn map_one(
    physical: &mut Physical,
    base: usize,
    len: usize,
    slice: Slice,
    prepared: bool,
) -> bool {
    if !prepared {
        // SAFETY: splits the placeholder around the range (failing harmlessly
        // when the range is the whole placeholder); acts only on placeholders.
        unsafe {
            VirtualFree(
                base as *mut c_void,
                len,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
        }
    }
    // SAFETY: maps the section onto the prepared placeholder range.
    let view = unsafe {
        MapViewOfFile3(
            slice.section.0,
            GetCurrentProcess(),
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value.is_null() || view.Value as usize != base {
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    if !commit(base, len) {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        return false;
    }
    // New contents, read-write: the GPU side's copies no longer guard it.
    victory_gpu::title_memory_remapped(base, len);
    // What the range shows now, for the GPU side's in-place census.
    victory_gpu::title_memory_view(
        base,
        len,
        slice.section.0 as usize,
        (slice.index * PAGE) as u64,
    );
    for k in 0..len.div_ceil(PAGE) {
        physical
            .shown
            .entry(slice.key(k))
            .or_default()
            .push(base + k * PAGE);
    }
    physical.views.insert(base, (len, slice));
    true
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        victory_loader::log_crash(&commit_limit_message(len, &error, queried.then_some(&info)));
    } else {
        tracing::warn!(len, %error, "committing physical pages failed");
    }
    false
}

fn commit_limit_message(
    len: usize,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of physical pages: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return;
    }
    let end = base + len;
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    for at in hit {
        let Some((size, slice)) = physical.views.remove(&at) else {
            continue;
        };
        for k in 0..size.div_ceil(PAGE) {
            let key = slice.key(k);
            if let Some(addresses) = physical.shown.get_mut(&key) {
                addresses.retain(|&a| a != at + k * PAGE);
                if addresses.is_empty() {
                    physical.shown.remove(&key);
                }
            }
        }
        victory_gpu::title_memory_remapped(at, size);
        // SAFETY: a view we mapped; it becomes a placeholder again.
        unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        if at < base {
            map(physical, at, base - at, slice.clone());
        }
        if at + size > end {
            let index = slice.index + (end - at) / PAGE;
            map(physical, end, at + size - end, Slice { index, ..slice });
        }
    }
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: COMMITTED.load(Ordering::Relaxed),
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    // What the title is told, logged at the first asking and at every
    // 1,024th: "used" is the title's own committed memory (its physical
    // pages and virtual allocations), not the process's — the layer's
    // copy-back buffers, host heaps and driver allocations are not the
    // title's to budget against. Before this count, the figure was the
    // whole process's (27 GB in run 60's gameplay against a limit of
    // 13.5 GB) and a title budgeting its streaming from it believed it
    // was far over its limit.
    static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let asked = ASKED.fetch_add(1, Ordering::Relaxed);
    if asked.is_multiple_of(1024) {
        tracing::info!(
            asked = asked + 1,
            which = _which,
            limit_mb = stats.title_limit >> 20,
            used_mb = stats.title_used >> 20,
            "the title asks how much memory it uses"
        );
    }
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        // A committed allocation of the title's own, so `title_used` is
        // non-zero no matter which other tests ran.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            MIB,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS,
            PAGE_READWRITE,
        );
        assert!(!p.is_null());
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used >= MIB);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, &error, Some(&info));
        assert!(report.contains("4194304 bytes"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, &error, None);
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    /// A thread that reads a mapped page in place, as the GPU side reads a
    /// texture, never meets the moment in which the runtime has taken the
    /// page's view away to map it again: mapping the same pages at the same
    /// address unmaps the view and maps a new one, and without the lock the
    /// reader faults on a page "reserved, never committed" (the crashes of
    /// runs 71 and 74).
    #[test]
    fn a_read_in_place_never_meets_the_gap_of_a_remapping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u8>().write(1) };
        let first = a as usize;
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let reader = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut reads = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let _reading = victory_gpu::title_memory_reading();
                    // SAFETY: the page is mapped whenever no change of
                    // mapping is under way, and none is while `_reading`
                    // is held.
                    assert_eq!(unsafe { (first as *const u8).read_volatile() }, 1);
                    reads += 1;
                }
                reads
            })
        };
        for _ in 0..2000 {
            // The same pages at the same address: the view goes and comes
            // back, with the same bytes.
            assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        }
        stop.store(true, Ordering::Relaxed);
        assert!(reader.join().expect("the reader never faulted") > 0);
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through a view of our own.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = physical().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// Pages say where the title has them mapped, byte for byte (after
    /// fork PR #571's test; its runs are whole pages).
    #[test]
    fn pages_say_where_they_are_mapped() {
        let pages = allocate(3);
        let reserve = |n: usize| {
            let at = XMemVirtualAlloc(
                std::ptr::null_mut(),
                n * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            );
            assert!(!at.is_null());
            at
        };
        // `b` keeps a page unmapped at either end, so that no run at `b`
        // ever touches one at `a`, wherever the host puts the two.
        let (a, b) = (reserve(3), reserve(4));
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty(), "mapped nowhere");
        // All three at `a`; the last two at `b`, from its second page on.
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        let second = b.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(second, 2, pages[1..].as_ptr()), second);
        let (a, second) = (a as usize, second as usize);
        let sorted = |mut runs: Vec<(usize, usize)>| {
            runs.sort_unstable();
            runs
        };
        assert_eq!(shown_at(&pages, 0, PAGE), vec![(a, PAGE)]);
        // A request for part of a page names those bytes alone.
        assert_eq!(shown_at(&pages, 0x4000, 0x3000), vec![(a + 0x4000, 0x3000)]);
        // Across two pages, shown at two addresses each: one run at each.
        assert_eq!(
            shown_at(&pages, PAGE + 1, PAGE),
            sorted(vec![(a + PAGE + 1, PAGE), (second + 1, PAGE)])
        );
        // A page unmapped in the middle of a view: the rest still shows.
        let middle = (a + PAGE) as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(middle, 1, std::ptr::null()), middle);
        assert_eq!(
            shown_at(&pages, 0, 3 * PAGE),
            sorted(vec![(a, PAGE), (a + 2 * PAGE, PAGE), (second, 2 * PAGE)])
        );
        // Too few pages for the bytes: nothing (the write fails too).
        assert!(shown_at(&pages, 0, 3 * PAGE + 1).is_empty());
        for (at, n) in [(a, 3), (second, 2)] {
            let at = at as *mut c_void;
            assert_eq!(XMemMapPhysicalPages(at, n, std::ptr::null()), at);
        }
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty());
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn map_alias_coalesces_runs_skips_noop_and_fails_atomically() {
        let (a, b) = (allocate(3), allocate(1));
        let reserve = |n: usize| {
            let p = XMemVirtualAlloc(
                std::ptr::null_mut(),
                n * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            );
            assert!(!p.is_null());
            p
        };
        let (src, src2, dst, unmapped_src) = (reserve(3), reserve(3), reserve(3), reserve(3));
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);

        assert_eq!(XMemMapPhysicalPages(src, 3, a.as_ptr()), src);
        // SAFETY: `src` has 3 mapped read-write pages.
        unsafe {
            at(src, 0).cast::<u8>().write(11);
            at(src, 1).cast::<u8>().write(22);
            at(src, 2).cast::<u8>().write(33);
        }

        // Coalesces 3 consecutive source slices into one view at `dst`.
        assert!(map_alias(dst as u64, src as u64, (3 * PAGE) as u64));
        assert_eq!(
            physical().views.get(&(dst as usize)).map(|v| v.0),
            Some(3 * PAGE)
        );
        // SAFETY: `dst` aliases `src`.
        unsafe {
            assert_eq!(at(dst, 0).cast::<u8>().read(), 11);
            assert_eq!(at(dst, 1).cast::<u8>().read(), 22);
            assert_eq!(at(dst, 2).cast::<u8>().read(), 33);
        }

        // Repeating the identical alias is a no-op and keeps the coalesced view.
        assert!(map_alias(dst as u64, src as u64, (3 * PAGE) as u64));
        assert_eq!(
            physical().views.get(&(dst as usize)).map(|v| v.0),
            Some(3 * PAGE)
        );

        // An unmapped source fails without unmapping or clobbering `dst`.
        assert!(!map_alias(
            dst as u64,
            unmapped_src as u64,
            (3 * PAGE) as u64
        ));
        assert_eq!(
            physical().views.get(&(dst as usize)).map(|v| v.0),
            Some(3 * PAGE)
        );
        // SAFETY: `dst` still shows the prior pages.
        unsafe {
            assert_eq!(at(dst, 0).cast::<u8>().read(), 11);
            assert_eq!(at(dst, 2).cast::<u8>().read(), 33);
        }

        // A mixed-section source splits into two coalesced runs: 2 * PAGE + PAGE.
        let mixed = [a[0], a[1], b[0]];
        assert_eq!(XMemMapPhysicalPages(src2, 3, mixed.as_ptr()), src2);
        // SAFETY: `src2` has 3 mapped read-write pages.
        unsafe { at(src2, 2).cast::<u8>().write(44) };
        assert!(map_alias(dst as u64, src2 as u64, (3 * PAGE) as u64));
        assert_eq!(
            physical().views.get(&(dst as usize)).map(|v| v.0),
            Some(2 * PAGE)
        );
        assert_eq!(
            physical().views.get(&(at(dst, 2) as usize)).map(|v| v.0),
            Some(PAGE)
        );
        // SAFETY: `dst` now aliases `mixed`.
        unsafe {
            assert_eq!(at(dst, 0).cast::<u8>().read(), 11);
            assert_eq!(at(dst, 1).cast::<u8>().read(), 22);
            assert_eq!(at(dst, 2).cast::<u8>().read(), 44);
        }

        // `from == 0` unmaps cleanly and a repeat unmap stays a safe no-op.
        assert!(map_alias(dst as u64, 0, (3 * PAGE) as u64));
        assert_eq!(query(dst).host.State, MEM_RESERVE);
        assert!(map_alias(dst as u64, 0, (3 * PAGE) as u64));

        assert_eq!(XMemMapPhysicalPages(src, 3, std::ptr::null()), src);
        assert_eq!(XMemMapPhysicalPages(src2, 3, std::ptr::null()), src2);
        for p in [src, src2, dst, unmapped_src] {
            release_placeholders(p, 3 * PAGE);
        }
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    fn release_placeholders(base: *mut c_void, len: usize) {
        physical().placeholders.remove(&(base as usize));
        let mut cur = base as usize;
        let end = cur + len;
        while cur < end {
            let mbi = query_region(cur).expect("placeholder region exists");
            assert_eq!(mbi.State, MEM_RESERVE);
            let alloc = mbi.AllocationBase;
            let region_end = (mbi.BaseAddress as usize) + mbi.RegionSize;
            // SAFETY: `alloc` is an unmapped reserved placeholder region.
            assert_ne!(unsafe { VirtualFree(alloc, 0, MEM_RELEASE) }, 0);
            cur = region_end;
        }
    }

    #[test]
    fn remapping_fragmented_views_into_contiguous_run_coalesces_placeholders() {
        let (a, b) = (allocate(3), allocate(3));
        let reserve = |n: usize| {
            let p = XMemVirtualAlloc(
                std::ptr::null_mut(),
                n * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            );
            assert!(!p.is_null());
            p
        };
        let (va, vb) = (reserve(3), reserve(3));
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);

        // Fragment `va` and `vb` into three 1-page placeholders/views each.
        for k in 0..3 {
            assert_eq!(XMemMapPhysicalPages(at(va, k), 1, &a[k]), at(va, k));
            assert_eq!(XMemMapPhysicalPages(at(vb, k), 1, &b[k]), at(vb, k));
            assert_eq!(
                physical().views.get(&(at(va, k) as usize)).map(|v| v.0),
                Some(PAGE)
            );
        }
        // SAFETY: `va` has 3 mapped read-write pages.
        unsafe {
            at(va, 0).cast::<u8>().write(10);
            at(va, 1).cast::<u8>().write(20);
            at(va, 2).cast::<u8>().write(30);
        }

        // Remap `va` as one contiguous 3-page run: placeholders coalesce into one view.
        assert_eq!(XMemMapPhysicalPages(va, 3, a.as_ptr()), va);
        assert_eq!(
            physical().views.get(&(va as usize)).map(|v| v.0),
            Some(3 * PAGE)
        );
        // SAFETY: `va` still shows the three pages.
        unsafe {
            assert_eq!(at(va, 0).cast::<u8>().read(), 10);
            assert_eq!(at(va, 1).cast::<u8>().read(), 20);
            assert_eq!(at(va, 2).cast::<u8>().read(), 30);
        }

        // Alias `vb` (currently 3 fragmented views of `b`) to `va` (3 pages of `a`).
        assert!(map_alias(vb as u64, va as u64, (3 * PAGE) as u64));
        assert_eq!(
            physical().views.get(&(vb as usize)).map(|v| v.0),
            Some(3 * PAGE)
        );
        // SAFETY: `vb` now aliases `va`.
        unsafe {
            assert_eq!(at(vb, 0).cast::<u8>().read(), 10);
            assert_eq!(at(vb, 1).cast::<u8>().read(), 20);
            assert_eq!(at(vb, 2).cast::<u8>().read(), 30);
        }

        assert_eq!(XMemMapPhysicalPages(va, 3, std::ptr::null()), va);
        assert_eq!(XMemMapPhysicalPages(vb, 3, std::ptr::null()), vb);
        release_placeholders(va, 3 * PAGE);
        release_placeholders(vb, 3 * PAGE);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(3, b.as_ptr()), 1);
    }

    #[test]
    fn scatter_batches_contiguous_and_scattered_mappings() {
        let a = allocate(3);
        let v = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!v.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);

        assert_eq!(
            XMemMapPhysicalPagesScatter(std::ptr::null(), 1, a.as_ptr()),
            0
        );
        assert_eq!(XMemMapPhysicalPagesScatter([v].as_ptr(), 0, a.as_ptr()), 1);
        assert_eq!(query(v).host.State, MEM_RESERVE);

        // Contiguous addresses with consecutive slices coalesce into one 3-page view.
        let addrs = [v, at(v, 1), at(v, 2)];
        assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), 3, a.as_ptr()), 1);
        assert_eq!(
            physical().views.get(&(v as usize)).map(|entry| entry.0),
            Some(3 * PAGE)
        );
        // SAFETY: `v..v+3*PAGE` is mapped read-write.
        unsafe {
            at(v, 0).cast::<u8>().write(51);
            at(v, 1).cast::<u8>().write(52);
            at(v, 2).cast::<u8>().write(53);
        }

        // Mixed unmap (0) and map in a single scatter call.
        let mixed_addrs = [at(v, 1), at(v, 3)];
        let mixed_pages = [0usize, a[0]];
        assert_eq!(
            XMemMapPhysicalPagesScatter(mixed_addrs.as_ptr(), 2, mixed_pages.as_ptr()),
            1
        );
        assert_eq!(query(at(v, 1)).host.State, MEM_RESERVE);
        // SAFETY: `at(v, 0)`, `at(v, 2)`, and `at(v, 3)` are mapped read-write.
        unsafe {
            assert_eq!(at(v, 0).cast::<u8>().read(), 51);
            assert_eq!(at(v, 2).cast::<u8>().read(), 53);
            assert_eq!(at(v, 3).cast::<u8>().read(), 51);
        }

        // Null `pages` unmaps every entry in the scatter list.
        let all_addrs = [v, at(v, 1), at(v, 2), at(v, 3)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(all_addrs.as_ptr(), 4, std::ptr::null()),
            1
        );
        for addr in all_addrs {
            assert_eq!(query(addr).host.State, MEM_RESERVE);
        }

        // Unknown page fails cleanly.
        assert_eq!(
            XMemMapPhysicalPagesScatter([v].as_ptr(), 1, [0x10usize].as_ptr()),
            0
        );

        release_placeholders(v, 4 * PAGE);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
    }

    #[test]
    fn adjacent_reservations_stay_independent_across_scatter_and_alias() {
        let a = allocate(2);
        let src = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        let v = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!src.is_null() && !v.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(src, 2, a.as_ptr()), src);

        // Split `v` into two placeholders and split `placeholders` tracking to
        // model two independent adjacent 1-page reservations.
        // SAFETY: `v` is a 2-page placeholder created above.
        assert_ne!(
            unsafe { VirtualFree(v, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) },
            0
        );
        {
            let mut phys = physical();
            phys.placeholders.insert(v as usize, PAGE);
            phys.placeholders.insert(at(v, 1) as usize, PAGE);
        }

        let addrs = [v, at(v, 1)];
        assert_eq!(XMemMapPhysicalPagesScatter(addrs.as_ptr(), 2, a.as_ptr()), 1);
        assert_eq!(
            physical().views.get(&(v as usize)).map(|entry| entry.0),
            Some(PAGE)
        );
        assert_eq!(
            physical().views.get(&(at(v, 1) as usize)).map(|entry| entry.0),
            Some(PAGE)
        );

        assert_eq!(
            XMemMapPhysicalPagesScatter(addrs.as_ptr(), 2, std::ptr::null()),
            1
        );
        // Aliasing across independent reservations also preserves their boundary.
        assert!(map_alias(v as u64, src as u64, (2 * PAGE) as u64));
        assert_eq!(
            physical().views.get(&(v as usize)).map(|entry| entry.0),
            Some(PAGE)
        );
        assert_eq!(
            physical().views.get(&(at(v, 1) as usize)).map(|entry| entry.0),
            Some(PAGE)
        );

        assert!(map_alias(v as u64, 0, (2 * PAGE) as u64));
        {
            let mut phys = physical();
            phys.placeholders.remove(&(v as usize));
            phys.placeholders.remove(&(at(v, 1) as usize));
        }
        // Releasing the first reservation leaves the second intact.
        // SAFETY: `v` is an unmapped 1-page placeholder.
        assert_ne!(unsafe { VirtualFree(v, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(at(v, 1)).host.State, MEM_RESERVE);
        // SAFETY: `at(v, 1)` is still an unmapped 1-page placeholder.
        assert_ne!(unsafe { VirtualFree(at(v, 1), 0, MEM_RELEASE) }, 0);

        assert_eq!(XMemMapPhysicalPages(src, 2, std::ptr::null()), src);
        release_placeholders(src, 2 * PAGE);
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
    }

    #[test]
    fn scatter_failure_preserves_later_mappings() {
        let (a, b) = (allocate(1), allocate(1));
        let v = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!v.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);

        assert_eq!(XMemMapPhysicalPages(at(v, 1), 1, b.as_ptr()), at(v, 1));
        // SAFETY: `at(v, 1)` is mapped read-write.
        unsafe { at(v, 1).cast::<u8>().write(77) };

        // Unknown page at entry 0 leaves mapped entry 1 intact.
        let addrs = [v, at(v, 1)];
        let bad_pages = [0x10usize, a[0]];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addrs.as_ptr(), 2, bad_pages.as_ptr()),
            0
        );
        assert_eq!(query(at(v, 1)).host.State, MEM_COMMIT);
        // SAFETY: `at(v, 1)` was never unmapped.
        assert_eq!(unsafe { at(v, 1).cast::<u8>().read() }, 77);

        // Host map failure at entry 0 (null or non-placeholder target) with a valid
        // different-section entry 1 also leaves mapped entry 1 intact.
        // SAFETY: plain committed host memory (not a placeholder).
        let non_placeholder = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!non_placeholder.is_null());
        for bad in [std::ptr::null_mut(), non_placeholder] {
            let bad_addrs = [bad, at(v, 1)];
            let good_pages = [a[0], a[0]];
            assert_eq!(
                XMemMapPhysicalPagesScatter(bad_addrs.as_ptr(), 2, good_pages.as_ptr()),
                0
            );
            assert_eq!(query(at(v, 1)).host.State, MEM_COMMIT);
            // SAFETY: `at(v, 1)` still shows `b[0]`.
            assert_eq!(unsafe { at(v, 1).cast::<u8>().read() }, 77);
        }
        // SAFETY: committed test allocation created above.
        assert_ne!(unsafe { VirtualFree(non_placeholder, 0, MEM_RELEASE) }, 0);

        assert_eq!(XMemMapPhysicalPages(at(v, 1), 1, std::ptr::null()), at(v, 1));
        release_placeholders(v, 2 * PAGE);
        assert_eq!(XMemFreePhysicalPages(1, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }
}
