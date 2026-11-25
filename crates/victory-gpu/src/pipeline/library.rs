//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder in [`SHARDS`] files, chosen by that name; the host refuses a
//!    library of another adapter or driver, and an entry whose description
//!    differs, so neither is ever used. A save serialises and writes only
//!    the shards that gained pipelines, because the host can only serialise
//!    a library whole and one blob of the world's pipelines is over 600 MB.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//!
//! Stores and serialising are serialised. Loads run outside that lock on a
//! reference of the library's own: vkd3d-proton keeps no compiled code in its
//! library, so there a load is a whole compile and holding the lock would make
//! the workers wait for one another.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Vec<u8>,
    /// Pipelines stored since the last save.
    added: u64,
    /// Names already stored during this run (avoids duplicate `StorePipeline`
    /// calls when two worker threads compile the same pipeline concurrently).
    stored: HashSet<String>,
}

// SAFETY: D3D12 pipeline libraries are internally synchronised except for
// same-name loads, which `LOADS` serialises; stores and serialising are under
// the mutex that holds the library.
unsafe impl Send for Library {}

/// How many files the library is split into. A save serialises and writes
/// only the shards that gained pipelines; see the module comment.
const SHARDS: usize = 16;
const EARLY_FILE: &str = "early-pipelines.txt";
/// Loads of the same pipeline serialise (the host documents that one as not
/// thread-safe); loads of different pipelines may overlap.
static LOADS: LazyLock<Mutex<HashMap<String, Arc<Mutex<()>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

/// The file shard `shard` is kept in.
fn shard_file(shard: usize) -> String {
    format!("host-pipelines.{shard:02}.bin")
}

/// The shard a pipeline named `name` belongs to. The name is a hash, so this
/// is spread and stable across runs: a pipeline always lands in the same
/// file, which is what lets a save write only the files that changed.
fn shard_of(name: &str) -> usize {
    let mut h = 0u32;
    for b in name.bytes() {
        h = h.wrapping_mul(31).wrapping_add(u32::from(b));
    }
    h as usize % SHARDS
}

/// One save at a time: serialising and publishing must not interleave, or an
/// older complete snapshot could be published over a newer one.
static SAVING: Mutex<()> = Mutex::new(());

fn libraries() -> Option<&'static [Mutex<Library>; SHARDS]> {
    static LIBRARIES: OnceLock<Option<[Mutex<Library>; SHARDS]>> = OnceLock::new();
    LIBRARIES
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let read = |file| DIR.get().and_then(|d| std::fs::read(d.join(file)).ok());
            let mut shards = Vec::with_capacity(SHARDS);
            for shard in 0..SHARDS {
                let blob = read(shard_file(shard)).unwrap_or_default();
                shards.push(Mutex::new(open(&device, blob)?));
            }
            shards.try_into().ok()
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // An empty library is asked for with a null blob: an empty slice's
        // pointer is dangling, which vkd3d-proton (Proton) rejects.
        let data = if bytes.is_empty() {
            std::ptr::null()
        } else {
            bytes.as_ptr()
        };
        let mut library: Option<ID3D12PipelineLibrary> = None;
        // SAFETY: the bytes outlive the library (kept beside it); the out
        // pointer is ours and typed by the interface id passed with it.
        unsafe {
            (Interface::vtable(device).CreatePipelineLibrary)(
                Interface::as_raw(device),
                data.cast(),
                bytes.len(),
                &ID3D12PipelineLibrary::IID,
                (&raw mut library).cast(),
            )
            .ok()?;
        }
        library.ok_or_else(windows::core::Error::empty)?.cast()
    };
    if !blob.is_empty() {
        match make(&blob) {
            Ok(library) => {
                tracing::info!(bytes = blob.len(), "host pipeline library loaded");
                return Some(Library {
                    library,
                    _blob: blob,
                    added: 0,
                    stored: HashSet::new(),
                });
            }
            Err(e) => {
                tracing::info!(error = %e, "host pipeline library rejected; starting an empty one")
            }
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    let library = make(&[]).ok()?;
    Some(Library {
        library,
        _blob: Vec::new(),
        added: 0,
        stored: HashSet::new(),
    })
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root]
        .into_iter()
        .chain(code.iter().copied())
        .chain([state.as_bytes()])
    {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    let Some(libraries) = libraries() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    let shard = &libraries[shard_of(name)];
    // A load can be a whole compile under vkd3d-proton, whose library keeps
    // no compiled code; do not hold the store/serialise lock across it. The
    // host requires only same-name loads to serialise, so those get a lock of
    // their own and different pipelines still load in parallel.
    let handle = shard
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .library
        .clone();
    let same = LOADS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(name.to_owned())
        .or_default()
        .clone();
    let _same = same.lock().unwrap_or_else(|e| e.into_inner());
    if let Ok(pso) = load(&handle, PCWSTR(wide.as_ptr())) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the lock: compiles take up to seconds.
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = shard.lock().unwrap_or_else(|e| e.into_inner());
    if l.stored.insert(name.to_owned()) {
        // SAFETY: a live pipeline; a name already stored in an earlier run
        // is refused, which leaves the library as it was.
        if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
            l.added += 1;
        }
    }
    Ok(pso)
}

/// Writes the shards that gained pipelines since the last save, and the
/// early list if it changed, when `SAVE_EVERY` has passed (or `now`).
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let (Some(dir), Some(libraries)) = (DIR.get(), libraries()) else {
        return;
    };
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    // Serialising and publishing are both inside this lock, so two saves
    // cannot interleave and publish an older snapshot over a newer one.
    let _saving = SAVING.lock().unwrap_or_else(|e| e.into_inner());
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!(error = %e, "host pipeline library folder not made");
        return;
    }
    let used_changed = USED_CHANGED.load(Ordering::Relaxed);
    let used: Vec<String> = USED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .flatten()
        .cloned()
        .collect();
    let mut saved = 0u32;
    let mut total = 0usize;
    for (shard, library) in libraries.iter().enumerate() {
        let (blob, added) = {
            let l = library.lock().unwrap_or_else(|e| e.into_inner());
            if l.added == 0 {
                continue;
            }
            // SAFETY: a buffer of the size the library asks for.
            let bytes = unsafe {
                let mut bytes = vec![0u8; l.library.GetSerializedSize()];
                l.library.Serialize(&mut bytes).map(|()| bytes)
            };
            match bytes {
                Ok(bytes) => (bytes, l.added),
                Err(e) => {
                    tracing::warn!(error = %e, shard, "host pipeline library shard not serialised");
                    continue;
                }
            }
        };
        match write_atomically(&dir.join(shard_file(shard)), &blob) {
            Ok(()) => {
                // Cleared only once the bytes are published, by what was
                // serialised: a failed write leaves the shard dirty for the
                // next save, and a store during the write keeps it dirty.
                let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
                l.added = l.added.saturating_sub(added);
                saved += 1;
                total += blob.len();
            }
            Err(e) => tracing::warn!(error = %e, shard, "host pipeline library shard not saved"),
        }
    }
    if used_changed || saved > 0 {
        match write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()) {
            Ok(()) => USED_CHANGED.store(false, Ordering::Relaxed),
            Err(e) => tracing::warn!(error = %e, "early pipeline list not saved"),
        }
    }
    if saved > 0 {
        tracing::info!(
            shards = saved,
            bytes = total,
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipeline library saved"
        );
    }
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Title pipeline keys requested by draws or dispatches (this run and the last).
static USED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<HashSet<String>>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let set: HashSet<String> = listed
            .lines()
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect();
        tracing::info!(pipelines = set.len(), "title pipelines to start early");
        *used = Some(set);
    }
    used
}

/// Whether the title pipeline with `key` was requested in this or an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|s| s.contains(key))
}

/// A draw or dispatch requested the title pipeline with `key`.
pub(crate) fn note_used(key: &str) {
    let mut used = used();
    let set = used.get_or_insert_with(HashSet::new);
    if set.len() < EARLY_MAX && set.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    #[test]
    fn a_pipeline_lands_in_one_shard_and_the_shards_spread() {
        let first = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(shard_of(&first), shard_of(&first), "stable across runs");
        let mut seen = [0usize; SHARDS];
        for i in 0..2048u32 {
            seen[shard_of(&format!("{i:032x}"))] += 1;
        }
        assert!(seen.iter().all(|&n| n > 0), "every shard is used: {seen:?}");
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let library =
            open(&device, b"not a pipeline library".repeat(16)).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }
}
