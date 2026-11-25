//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of each page at every
//! submission that samples it. Only the subresources whose memory holds a
//! changed page are detiled again (and, when watched, only the changed
//! pages read again).
//!
//! shortcut: a changed subresource is detiled whole, not just its blocks in
//! the pages written, and every subresource is still copied into the host
//! texture. Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

#[cfg(test)]
use victory_formats::texture::Error;
use victory_formats::texture::Layout;
use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    size: usize,
    data: *mut u8,
    footprints: Vec<UploadFootprint>,
    layout: Layout,
    /// Mips per array slice.
    mips: u32,
    /// Per subresource, the byte ranges of the memory it is detiled from.
    extents: Vec<Vec<Range<usize>>>,
    /// Top console mips omitted from the host copy.
    mip_drop: u32,
    memory: *const u8,
    last: Mutex<Last>,
}

struct UploadFootprint {
    placed: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    range: Range<usize>,
    depth_pitch: usize,
}

impl UploadFootprint {
    fn new(
        placed: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        rows: u32,
        row_size: u64,
        total: usize,
    ) -> Option<Self> {
        let pitch = placed.Footprint.RowPitch as usize;
        let row_size = usize::try_from(row_size).ok()?;
        let (len, depth_pitch) = footprint_size(pitch, rows, placed.Footprint.Depth, row_size)?;
        let start = usize::try_from(placed.Offset).ok()?;
        let end = start.checked_add(len)?;
        (end <= total).then_some(Self {
            placed,
            range: start..end,
            depth_pitch,
        })
    }
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after,
/// with room to read the memory into.
#[derive(Default)]
struct Last {
    hashes: Vec<blake3::Hash>,
    asked: u64,
    read: Vec<u8>,
}

/// Whether any of `changed` overlaps any of `extents` (both sorted).
fn overlaps(extents: &[Range<usize>], changed: &[Range<usize>]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < extents.len() && j < changed.len() {
        let (e, c) = (&extents[i], &changed[j]);
        if e.start < c.end && c.start < e.end {
            return true;
        }
        if e.end <= c.end {
            i += 1;
        } else {
            j += 1;
        }
    }
    false
}

/// The parts of `[0, len)` in which `a` and `b`, hashes of its `page`-byte
/// parts, differ; `None` when they do not hash the same parts.
fn changed_pages(
    a: &[blake3::Hash],
    b: &[blake3::Hash],
    page: usize,
    len: usize,
) -> Option<Vec<Range<usize>>> {
    (a.len() == b.len()).then(|| {
        (0..a.len())
            .filter(|&i| a[i] != b[i])
            .map(|i| i * page..((i + 1) * page).min(len))
            .collect()
    })
}

/// `[start, start + len)`'s byte ranges, relative to `start`, of the
/// `page`-byte pages numbered `pages` (in order).
fn page_ranges(
    pages: impl Iterator<Item = usize>,
    page: usize,
    start: usize,
    len: usize,
) -> Vec<Range<usize>> {
    pages
        .map(|p| {
            let end = ((p + 1) * page).min(start + len);
            (p * page).max(start) - start..end.saturating_sub(start)
        })
        .filter(|r| r.start < r.end)
        .collect()
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostWrite {
    /// Write-watched memory: the write-watch asking when the host copy happened.
    Watched {
        start: usize,
        end: usize,
        asked: u64,
    },
    /// Non-watched memory (physical-page view): content hash at the host copy.
    Hashed {
        memory: usize,
        size: usize,
        hash: Option<blake3::Hash>,
    },
    /// Layout unknown or unsupported: keep host contents until released.
    Fixed,
}

/// Textures the host GPU wrote by a copy or resolve: their contents stay on
/// the host until the title writes their memory again.
static HOST_WRITTEN: Mutex<Option<HashMap<usize, HostWrite>>> = Mutex::new(None);

fn host_write_state(memory: *const u8, size: usize) -> HostWrite {
    let start = memory as usize;
    if let Some((now, _)) = crate::watch::writes(start, start + size, false) {
        return HostWrite::Watched {
            start,
            end: start + size,
            asked: now,
        };
    }
    let hash = committed(memory, size).then(|| {
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(memory, size) };
        blake3::hash(tiled)
    });
    HostWrite::Hashed {
        memory: start,
        size,
        hash,
    }
}

/// Whether `resource`'s host copy is still newer than its title memory.
fn still_host_written(resource: usize) -> bool {
    let state = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|map| map.get(&resource).copied());
    let Some(state) = state else {
        return false;
    };
    let unchanged = match state {
        HostWrite::Fixed => true,
        HostWrite::Watched { start, end, asked } => crate::watch::writes(start, end, false)
            .is_none_or(|(_, written)| written.iter().all(|&(_, w)| w <= asked)),
        HostWrite::Hashed { memory, size, hash } => {
            if !committed(memory as *const u8, size) {
                true
            } else {
                // SAFETY: committed memory of the texture's layout size.
                let tiled = unsafe { std::slice::from_raw_parts(memory as *const u8, size) };
                hash == Some(blake3::hash(tiled))
            }
        }
    };
    if !unchanged {
        let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(map) = map.as_mut()
            && map.get(&resource) == Some(&state)
        {
            map.remove(&resource);
        }
    }
    unchanged
}

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    if crate::resource::host_resource(resource).is_some_and(|t| !title_written(t)) {
        return;
    }
    let state = crate::resource::texture_memory(resource)
        .and_then(|(memory, desc)| {
            let layout = crate::resource::texture_layout(&desc)?.planes[0]
                .layout
                .clone();
            (!layout.desc.swizzle.xor()).then(|| host_write_state(memory, layout.size as usize))
        })
        .unwrap_or(HostWrite::Fixed);
    let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    map.get_or_insert_with(HashMap::new)
        .insert(resource as usize, state);
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut host_written = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(host_written) = host_written.as_mut() {
        host_written.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    // SAFETY: a live resource.
    title_written_flags(unsafe { texture.GetDesc() }.Flags)
}

fn title_written_flags(flags: D3D12_RESOURCE_FLAGS) -> bool {
    let host_writable = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
        | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
        | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    (flags & host_writable) == D3D12_RESOURCE_FLAG_NONE
}

/// A fingerprint of the title memory `[at, at + len)`, never 0: of all of
/// it up to 48 KiB, else of 16 KiB at its start, its middle and its end
/// (a texture streams in whole mips; the small ones lie first). Read
/// through the kernel, so pages not there count as zeros.
pub(crate) fn fingerprint(at: u64, len: u64) -> u64 {
    const WINDOW: usize = 16 << 10;
    thread_local! {
        static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let len = len as usize;
    let mut hasher = blake3::Hasher::new();
    READ.with(|read| {
        let mut read = read.borrow_mut();
        read.resize(3 * WINDOW, 0);
        let mut take = |from: usize, bytes: usize| {
            read[..bytes].fill(0);
            crate::host::copy_mapped(
                read.as_mut_ptr(),
                (at as usize + from) as *const u8,
                bytes,
                false,
            );
            hasher.update(&read[..bytes]);
        };
        if len <= 3 * WINDOW {
            take(0, len);
        } else {
            take(0, WINDOW);
            take((len / 2) & !0xfff, WINDOW);
            take(len - WINDOW, WINDOW);
        }
    });
    let hash = hasher.finalize();
    let mut first = [0u8; 8];
    first.copy_from_slice(&hash.as_bytes()[..8]);
    u64::from_le_bytes(first) | 1
}

/// Where a staged title texture's memory is: address and bytes.
pub(crate) fn staged_memory(resource: This) -> Option<(u64, u64)> {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let staging = map.as_ref()?.get(&(resource as usize))?;
    Some((staging.memory as u64, staging.layout.size))
}

/// Frames after a whole upload of a texture at which its memory is looked
/// at again, each counted from the look before.
const RECHECK_FRAMES: [u64; 5] = [4, 12, 40, 120, 400];

/// Textures uploaded whole, looked at again while their memory may still
/// be streaming in with nothing to tell of it: resource -> (address,
/// bytes, fingerprint at the upload, looks that found it unchanged, frame
/// of the next look).
/// (Address, bytes, fingerprint at the upload, looks that found it
/// unchanged, frame of the next look) of one uploaded texture.
type Recheck = (u64, u64, u64, u8, u64);

static RECHECKS: Mutex<Option<HashMap<usize, Recheck>>> = Mutex::new(None);

/// Whether uploaded textures are looked at again after their upload
/// (`VICTORY_HEAP_RECHECK=0`: no).
fn heap_recheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_RECHECK").is_none_or(|v| v != "0"))
}

/// `resource` was uploaded whole: looked at again from frame `due` (the
/// title may fill the texture's memory only after it wrote the view, with
/// no descriptor write and nothing from the runtime to tell of it; the
/// host copy then kept what the memory held at the first look for the
/// rest of the run).
fn recheck_later(resource: This, due: u64) {
    if !heap_recheck() {
        return;
    }
    let Some((at, len)) = staged_memory(resource) else {
        return;
    };
    let print = fingerprint(at, len);
    let mut map = RECHECKS.lock().unwrap_or_else(|e| e.into_inner());
    map.get_or_insert_with(HashMap::new)
        .insert(resource as usize, (at, len, print, 0, due));
}

/// The memory of a texture looked at again at `frame` changed since its
/// upload: the next refresh re-reads and re-detiles it whole, and the
/// caller uploads it again. Unchanged: the look is rescheduled
/// (`RECHECK_FRAMES`), or the looks end.
pub(crate) fn recheck_due(resource: This, frame: u64) -> bool {
    if !heap_recheck() {
        return false;
    }
    let entry = {
        let mut map = RECHECKS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(map) = map.as_mut() else {
            return false;
        };
        let Some(entry) = map.get_mut(&(resource as usize)) else {
            return false;
        };
        if frame < entry.4 {
            return false;
        }
        *entry
    };
    let (at, len, print, looks, _) = entry;
    if fingerprint(at, len) == print {
        let mut map = RECHECKS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(map) = map.as_mut() {
            match RECHECK_FRAMES.get(usize::from(looks) + 1) {
                Some(wait) => {
                    if let Some(entry) = map.get_mut(&(resource as usize)) {
                        entry.3 = looks + 1;
                        entry.4 = frame + wait;
                    }
                }
                None => {
                    map.remove(&(resource as usize));
                }
            }
        }
        return false;
    }
    {
        let mut map = RECHECKS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(map) = map.as_mut() {
            map.remove(&(resource as usize));
        }
    }
    // The incremental refresh asks the write-watch which pages changed;
    // this write is one nothing told of. The next refresh re-reads the
    // memory whole instead.
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(staging) = map.as_ref().and_then(|map| map.get(&(resource as usize))) {
        staging.last.lock().unwrap_or_else(|e| e.into_inner()).asked = 0;
    }
    tracing::warn!(
        resource = format_args!("{:?}", resource as usize),
        address = format_args!("{at:#x}"),
        bytes = len,
        looks,
        "a texture changed after its upload with nothing to tell of it: uploaded again"
    );
    true
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    if !list.first_use(resource as usize) || still_host_written(resource as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.placed.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.placed.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.placed.Offset, footprint.range.len() as u64),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint.placed,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
    recheck_later(resource, crate::queue::presented() + RECHECK_FRAMES[0]);
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let Ok(size) = usize::try_from(self.layout.size) else {
            return 0;
        };
        let start = self.memory as usize;
        let Some(end) = start
            .checked_add(size)
            .filter(|_| size <= isize::MAX as usize)
        else {
            return 0;
        };
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some((now, written)) = crate::watch::writes(start, end, false) {
            let before = last.asked;
            if before != 0 && written.iter().all(|&(_, w)| w <= before) {
                return 0;
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            if before == 0 || last.read.len() != size {
                last.read.resize(size, 0);
                crate::host::copy_mapped(last.read.as_mut_ptr(), self.memory, size, false);
                self.detile(&last.read, None);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
                return size;
            }
            // The rest of the memory is as last read: read again only the
            // pages written since, and detile what they hold.
            let pages = written
                .iter()
                .filter(|&&(_, w)| w > before)
                .map(|&(p, _)| p);
            let changed = page_ranges(pages, crate::watch::PAGE, start, size);
            let mut read = 0;
            for r in &changed {
                // SAFETY: `r` lies inside both the read copy and the
                // texture's memory, each `size` bytes.
                let (to, from) = unsafe {
                    (
                        last.read.as_mut_ptr().add(r.start),
                        self.memory.add(r.start),
                    )
                };
                crate::host::copy_mapped(to, from, r.len(), false);
                read += r.len();
            }
            self.detile(&last.read, Some(&changed));
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, read as u64);
            return read;
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return 0;
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let page = crate::watch::PAGE;
        let hashes: Vec<blake3::Hash> = tiled.chunks(page).map(blake3::hash).collect();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hashes == hashes {
            return size;
        }
        let changed = changed_pages(&last.hashes, &hashes, page, size);
        self.detile(tiled, changed.as_deref());
        last.hashes = hashes;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer: the
    /// subresources whose memory overlaps `changed` (sorted byte ranges),
    /// or all of them.
    fn detile(&self, tiled: &[u8], changed: Option<&[Range<usize>]>) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, footprint) in self.footprints.iter().enumerate() {
            if changed.is_some_and(|c| !overlaps(&self.extents[i], c)) {
                continue;
            }
            let (host_mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let mip = host_mip + self.mip_drop;
            // SAFETY: the upload buffer is mapped for exactly this range.
            let linear = unsafe { std::slice::from_raw_parts_mut(self.data, self.size) };
            if let Err(e) = self.layout.detile(
                tiled,
                mip,
                slice,
                &mut linear[footprint.range.clone()],
                footprint.placed.Footprint.RowPitch as usize,
                footprint.depth_pitch,
            ) {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

#[cfg(test)]
fn detile_plane(
    layout: &Layout,
    mips: u32,
    footprints: &[UploadFootprint],
    tiled: &[u8],
    linear: &mut [u8],
) -> Result<(), (Error, u32, u32)> {
    for (i, footprint) in footprints.iter().enumerate() {
        let (mip, slice) = (i as u32 % mips, i as u32 / mips);
        let target =
            linear
                .get_mut(footprint.range.clone())
                .ok_or((Error::TooSmall, mip, slice))?;
        layout
            .detile(
                tiled,
                mip,
                slice,
                target,
                footprint.placed.Footprint.RowPitch as usize,
                footprint.depth_pitch,
            )
            .map_err(|e| (e, mip, slice))?;
    }
    Ok(())
}

/// Bytes read by a copy and the stride between depth slices.
fn footprint_size(pitch: usize, rows: u32, depth: u32, row_size: usize) -> Option<(usize, usize)> {
    if row_size == 0 || row_size > pitch {
        return None;
    }
    let depth_pitch = pitch.checked_mul(rows as usize)?;
    // D3D12 allocates the data bytes of the last row only.
    let preceding_rows = (rows as usize)
        .checked_mul(depth as usize)?
        .checked_sub(1)?;
    let len = pitch.checked_mul(preceding_rows)?.checked_add(row_size)?;
    Some((len, depth_pitch))
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return false;
    };
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        let Some(next) = (info.BaseAddress as usize).checked_add(info.RegionSize) else {
            return false;
        };
        p = next;
    }
    true
}

fn plane_subresources(desc: &D3D12_RESOURCE_DESC) -> u32 {
    let mips = u32::from(desc.MipLevels);
    if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        mips
    } else {
        mips * u32::from(desc.DepthOrArraySize)
    }
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = plane_subresources(&d);
    if count == 0 {
        return None;
    }
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut row_sizes = vec![0u64; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            Some(row_sizes.as_mut_ptr()),
            Some(&mut total),
        );
    }
    let size = usize::try_from(total)
        .ok()
        .filter(|&size| size <= isize::MAX as usize)?;
    let footprints = footprints
        .into_iter()
        .zip(rows)
        .zip(row_sizes)
        .map(|((placed, rows), row_size)| UploadFootprint::new(placed, rows, row_size, size))
        .collect::<Option<Vec<_>>>()?;
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let mips = u32::from(d.MipLevels).max(1);
    let (mip_drop, _) = crate::resource::host_mip_info(resource);
    let whole: Vec<_> = std::iter::once(0..layout.size as usize).collect();
    let extents = (0..count)
        .map(|i| match layout.extents(i % mips + mip_drop, i / mips) {
            Ok(e) => e
                .into_iter()
                .map(|r| r.start as usize..r.end as usize)
                .collect(),
            // Detiling it fails too; any change "touches" it.
            Err(_) => whole.clone(),
        })
        .collect();
    let s = Arc::new(Staging {
        buffer,
        size,
        data,
        footprints,
        layout,
        mips: u32::from(d.MipLevels),
        extents,
        mip_drop,
        memory,
        last: Mutex::default(),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

#[cfg(test)]
// Single ranges in brackets are meant: lists of one range.
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;
    use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_R8_UNORM,
        DXGI_FORMAT_R8G8B8A8_UNORM,
    };

    #[test]
    fn ranges_overlap_only_when_they_share_a_byte() {
        let extents = [10..20, 40..50];
        assert!(overlaps(&extents, &[15..16]));
        assert!(overlaps(&extents, &[0..5, 49..60]));
        assert!(overlaps(&extents, &[0..11]));
        // Touching ends share no byte.
        assert!(!overlaps(&extents, &[20..40]));
        assert!(!overlaps(&extents, &[0..10, 20..40, 50..90]));
        assert!(!overlaps(&extents, &[]));
        assert!(!overlaps(&[], &[0..100]));
        // A long change across several extents, and the reverse.
        assert!(overlaps(&[10..20, 40..50], &[0..100]));
        assert!(overlaps(&[0..100], &[10..20, 40..50]));
    }

    #[test]
    fn hashed_pages_that_differ_are_the_change() {
        let h = |b: u8| blake3::hash(&[b]);
        let (a, b) = ([h(1), h(2), h(3)], [h(1), h(9), h(9)]);
        // The last page is short: the memory ends inside it.
        assert_eq!(
            changed_pages(&a, &b, 64, 150),
            Some(vec![64..128, 128..150])
        );
        assert_eq!(changed_pages(&a, &a, 64, 150), Some(vec![]));
        // Nothing hashed before: no part known to be unchanged.
        assert_eq!(changed_pages(&[], &b, 64, 150), None);
    }

    #[test]
    fn written_pages_become_ranges_of_the_memory() {
        // Memory from 100 for 250 bytes, in 64-byte pages 1..=5.
        let r = page_ranges([1, 3, 5].into_iter(), 64, 100, 250);
        assert_eq!(r, vec![0..28, 92..156, 220..250]);
        // Pages outside the memory give nothing.
        assert!(page_ranges([0, 6].into_iter(), 64, 100, 250).is_empty());
    }

    #[test]
    fn host_written_clears_after_write_watched_memory_is_written() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let len = crate::watch::PAGE;
        // SAFETY: a fresh write-watched allocation, kept (not freed: a reused
        // address carries stale watch state into another test's first read).
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!ptr.is_null());
        let key = ptr as usize;
        // Drain watch state a reused address may carry from a freed
        // allocation, so the snapshot below sees a clean slate.
        let _ = crate::watch::writes(key, key + len, false);
        let state = host_write_state(ptr, len);
        assert!(matches!(state, HostWrite::Watched { .. }));
        {
            let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
            map.get_or_insert_with(HashMap::new).insert(key, state);
        }
        assert!(
            still_host_written(key),
            "untouched memory stays host-written"
        );
        // SAFETY: committed page of ours.
        unsafe { ptr.write(0x42) };
        assert!(
            !still_host_written(key),
            "writing title memory clears host-written state"
        );
        assert!(!still_host_written(key));
        forget(key);
    }

    #[test]
    fn host_written_clears_after_non_watched_memory_changes() {
        let mut buf = vec![0x11u8; 256];
        let key = buf.as_ptr() as usize;
        let state = host_write_state(buf.as_ptr(), buf.len());
        assert!(matches!(state, HostWrite::Hashed { hash: Some(_), .. }));
        {
            let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
            map.get_or_insert_with(HashMap::new).insert(key, state);
        }
        assert!(still_host_written(key), "unchanged bytes stay host-written");
        buf[0] = 0x22;
        assert!(
            !still_host_written(key),
            "changed title bytes clear host-written state"
        );
        assert!(!still_host_written(key));
        forget(key);
    }

    #[test]
    fn footprint_ends_after_the_last_rows_data() {
        // Two depth slices of three rows. The sixth row starts at 1280,
        // but only its 20 data bytes belong to the upload range.
        assert_eq!(footprint_size(256, 3, 2, 20), Some((1300, 768)));
    }

    #[test]
    fn footprint_rejects_invalid_geometry_and_out_of_bounds_placement() {
        assert_eq!(footprint_size(256, 0, 1, 20), None);
        assert_eq!(footprint_size(256, 1, 0, 20), None);
        assert_eq!(footprint_size(256, 1, 1, 0), None);
        assert_eq!(footprint_size(256, 1, 1, 257), None);
        assert_eq!(footprint_size(usize::MAX, 1, 2, 1), None);

        let placed = placed(512, [5, 3, 2]);
        let footprint = UploadFootprint::new(placed, 3, 20, 1812).unwrap();
        assert_eq!(footprint.range, 512..1812);
        assert!(UploadFootprint::new(placed, 3, 20, 1811).is_none());
        let overflowing = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: u64::MAX,
            ..placed
        };
        assert!(UploadFootprint::new(overflowing, 3, 20, usize::MAX).is_none());
    }

    #[test]
    fn plane_zero_counts_each_array_mip_and_each_volume_mip_once() {
        let mut desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE1D,
            DepthOrArraySize: 6,
            MipLevels: 4,
            ..Default::default()
        };
        assert_eq!(plane_subresources(&desc), 24);
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        assert_eq!(plane_subresources(&desc), 24);
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        assert_eq!(plane_subresources(&desc), 4);
    }

    fn placed(offset: u64, [width, height, depth]: [u32; 3]) -> D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: offset,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Width: width,
                Height: height,
                Depth: depth,
                RowPitch: 256,
                ..Default::default()
            },
        }
    }

    // Guest mip and slice, host offset and dimensions, then rows and bytes per row.
    type Shape = ((u32, u32), u64, [u32; 3], u32, u64);

    fn assert_detiles(desc: TextureDesc, element_bytes: usize, shapes: &[Shape], total: usize) {
        let layout = &Texture::new(&desc).unwrap().planes[0].layout;
        let mut tiled = vec![0u8; layout.size as usize];
        let mut linear = vec![0xff; total];
        let mut expected = linear.clone();
        let mut footprints = Vec::new();
        for (i, &((mip, slice), offset, dims, rows, row_size)) in shapes.iter().enumerate() {
            footprints
                .push(UploadFootprint::new(placed(offset, dims), rows, row_size, total).unwrap());
            for z in 0..dims[2] {
                for y in 0..rows {
                    for x in 0..row_size as usize / element_bytes {
                        let guest = layout.offset(mip, x as u32, y, slice + z).unwrap() as usize;
                        let host =
                            offset as usize + (z * rows + y) as usize * 256 + x * element_bytes;
                        let value = (i * 37 + (z * 7 + y * 13) as usize + x * 3) as u8;
                        for b in 0..element_bytes {
                            tiled[guest + b] = value + b as u8;
                            expected[host + b] = value + b as u8;
                        }
                    }
                }
            }
        }
        detile_plane(layout, desc.mips, &footprints, &tiled, &mut linear).unwrap();
        assert_eq!(linear, expected);
    }

    #[test]
    fn array_upload_uses_mip_first_order_and_preserves_padding() {
        let desc = TextureDesc {
            dimension: Dimension::Tex2d,
            width: 7,
            height: 5,
            depth_or_array: 2,
            mips: 2,
            format: DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            samples: 1,
            swizzle: Swizzle::SW_4KB_S,
        };
        assert_detiles(
            desc,
            4,
            &[
                ((0, 0), 0, [7, 5, 1], 5, 28),
                ((1, 0), 1536, [3, 2, 1], 2, 12),
                ((0, 1), 2048, [7, 5, 1], 5, 28),
                ((1, 1), 3584, [3, 2, 1], 2, 12),
            ],
            3852,
        );
    }

    #[test]
    fn volume_upload_uses_pitched_depth_slices_and_shrinks_depth_by_mip() {
        let desc = TextureDesc {
            dimension: Dimension::Tex3d,
            width: 5,
            height: 3,
            depth_or_array: 3,
            mips: 2,
            format: DXGI_FORMAT_R8_UNORM.0 as u32,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        };
        assert_detiles(
            desc,
            1,
            &[
                ((0, 0), 0, [5, 3, 3], 3, 5),
                ((1, 0), 2560, [2, 1, 1], 1, 2),
            ],
            2562,
        );
    }

    #[test]
    fn block_compressed_upload_uses_whole_blocks_for_partial_rows_and_mips() {
        for (format, block_bytes) in [(DXGI_FORMAT_BC1_UNORM, 8), (DXGI_FORMAT_BC7_UNORM, 16)] {
            let desc = TextureDesc {
                dimension: Dimension::Tex2d,
                width: 7,
                height: 5,
                depth_or_array: 1,
                mips: 2,
                format: format.0 as u32,
                samples: 1,
                swizzle: Swizzle::SW_4KB_S,
            };
            assert_detiles(
                desc,
                block_bytes as usize,
                &[
                    ((0, 0), 0, [8, 8, 1], 2, block_bytes * 2),
                    ((1, 0), 512, [4, 4, 1], 1, block_bytes),
                ],
                512 + block_bytes as usize,
            );
        }
    }

    #[test]
    fn host_writable_flags_exclude_texture_uploads() {
        assert!(title_written_flags(D3D12_RESOURCE_FLAG_NONE));
        assert!(title_written_flags(
            D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE
        ));
        for flag in [
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ] {
            assert!(!title_written_flags(flag));
            assert!(!title_written_flags(
                flag | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE
            ));
        }
    }

    #[test]
    fn committed_rejects_an_address_range_that_wraps() {
        assert!(!committed(std::ptr::without_provenance(usize::MAX), 2));
        assert!(committed(std::ptr::without_provenance(usize::MAX), 0));
        assert!(!committed(std::ptr::null(), 1));
    }

    #[test]
    fn committed_checks_every_region_in_the_range() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };

        // SAFETY: reserve two pages, then commit them separately with
        // different protections so VirtualQuery reports distinct regions.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 8192, MEM_RESERVE, PAGE_READWRITE);
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base, 4096, MEM_COMMIT, PAGE_READWRITE), base);
            let bytes = base.cast::<u8>();
            assert!(committed(bytes, 4096));
            assert!(!committed(bytes.add(4095), 2));
            let second = bytes.add(4096).cast();
            assert_eq!(
                VirtualAlloc(second, 4096, MEM_COMMIT, PAGE_READONLY),
                second
            );
            assert!(committed(bytes.add(4095), 2));
            assert!(committed(bytes, 8192));
            assert_ne!(VirtualFree(base, 0, MEM_RELEASE), 0);
        }
    }
}
