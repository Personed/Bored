//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by the translator
//! that translated it (its source fingerprint and settings, see `build`), so
//! a changed translator never serves an old translation. Entries go through
//! `victory-pipeline`'s content-addressed cache, which writes them
//! atomically; an entry that does not read back is translated again, and a
//! cache that cannot be written is skipped (it never fails a translation).
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (entries of older builds
//! are ignored anyway).

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, MeshEmu, MeshMode, MeshOutput, StaticSampler, UserData};
use victory_shader::resource::RootIndexedView;
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// The build fingerprint in front of every cache input, from [`build`].
const BUILD_LEN: usize = 32;

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open(dir) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What identifies the translation, besides the container: the shader
/// crate's fingerprint (its source, every switch it reads, the host facts the
/// device gave it, the loop limit, barrier waits and clean-up passes) and
/// this file's encoding of the result. Not the whole executable any more: a
/// rebuild that leaves the translator alone keeps its translations (each new
/// build used to translate the title's thousands of shaders again, which kept
/// the first minutes of every test run short of pipelines).
/// `VICTORY_SHADER_CACHE_KEY=exe` restores the old key (the executable's hash).
/// Worked out per lookup, so a translation made before the device said its
/// wave size is never filed under the later key.
fn build() -> [u8; 32] {
    static WARNED: std::sync::Once = std::sync::Once::new();
    let limit = victory_shader::emit::loop_limit();
    if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
        WARNED.call_once(|| tracing::warn!(limit, "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"));
    }
    static EXE: OnceLock<Option<[u8; 32]>> = OnceLock::new();
    let exe = EXE.get_or_init(|| {
        if std::env::var_os("VICTORY_SHADER_CACHE_KEY").is_some_and(|v| v == "exe") {
            match std::env::current_exe().and_then(std::fs::read) {
                Ok(bytes) => Some(*blake3::hash(&bytes).as_bytes()),
                Err(e) => {
                    tracing::warn!(error = %e, "cannot read our executable; keying on the translator");
                    None
                }
            }
        } else {
            None
        }
    });
    let mut h = blake3::Hasher::new();
    h.update(b"victory shader cache key 2\0");
    if let Some(exe) = exe {
        h.update(exe);
    }
    h.update(&victory_shader::emit::translation_fingerprint());
    h.update(include_bytes!("shader_cache.rs"));
    *h.finalize().as_bytes()
}

/// The offline warm-up (`victory translate --warm`): `container` is
/// translated into the cache unless it is there already. `Ok(true)`: it
/// was translated now. Needs [`use_shader_cache`] and the same host facts
/// the run will have (`victory_shader::emit::translation_fingerprint`).
pub fn warm(container: &[u8]) -> Result<bool, String> {
    let cache = CACHE.get().ok_or("no shader cache")?;
    let ran = AtomicBool::new(false);
    let translate = |c: &[u8]| {
        ran.store(true, Ordering::Relaxed);
        victory_shader::emit::translate(c).map_err(|e| e.to_string())
    };
    cached(cache, &build(), MeshMode::Native, container, translate).map(|_| ran.load(Ordering::Relaxed))
}

/// The translation of a console shader container, from the cache when
/// there is one.
pub(crate) fn host_shader(container: &[u8], mesh: MeshMode) -> Result<HostShader, String> {
    let translate =
        |c: &[u8]| victory_shader::emit::translate_with(c, mesh).map_err(|e| e.to_string());
    match CACHE.get() {
        Some(cache) => cached(cache, &build(), mesh, container, translate),
        None => translate(container),
    }
}



/// The cache entry's converter: the build and the container in, the
/// encoded translation out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let shader = (self.translate)(&input[BUILD_LEN + 1..]).map_err(anyhow::Error::msg)?;
        Ok(encode(&shader))
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    mesh: MeshMode,
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let mesh_byte = match mesh {
        MeshMode::Native => 0u8,
        MeshMode::Emulate => 1u8,
    };
    let input = [&build[..], &[mesh_byte], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
    };
    for _ in 0..2 {
        let path = match cache.get_or_convert(&converter, &input) {
            Ok(path) => path,
            Err(e) => {
                // A cache that cannot write (a full or read-only disk) must
                // not cost the shader: the translation is the same work the
                // converter just did, returned uncached instead of dropping
                // every draw of the program for the rest of the run.
                tracing::debug!("shader cache write failed ({e:#}); the shader is not cached");
                MISSES.fetch_add(1, Ordering::Relaxed);
                return (converter.translate)(container);
            }
        };
        if let Some(shader) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            let (counter, what) = if converter.ran.load(Ordering::Relaxed) {
                (&MISSES, "translated and cached")
            } else {
                (&HITS, "from the cache")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return Ok(shader);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    (converter.translate)(container)
}

const MAGIC: &[u8; 4] = b"VSH1";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

/// The tag of [`RootType::Other`]: a tag cannot hold every `u32`, so its
/// raw value follows it as its own word.
const OTHER_ROOT: u32 = 6;

fn root_tag(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(_) => OTHER_ROOT,
    }
}

/// then the raw value for [`RootType::Other`].
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_tag(u.root)] {
            word(&mut out, v);
        }
        if let RootType::Other(raw) = u.root {
            word(&mut out, raw);
        }
        word(&mut out, u.dword);
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, s.root_indexed_views.len() as u32);
    for view in &s.root_indexed_views {
        for value in [
            u32::from(view.table_slot),
            u32::from(view.constant_slot),
            view.constant_dword,
            view.byte_offset,
            u32::from(view.shift),
        ] {
            word(&mut out, value);
        }
    }
    let encode_outputs = |out: &mut Vec<u8>, items: &[MeshOutput]| {
        word(out, items.len() as u32);
        for o in items {
            let name = o.name.as_bytes();
            word(out, name.len() as u32);
            out.extend(name);
            for v in [
                o.index,
                u32::from(o.kind),
                u32::from(o.comp),
                u32::from(o.interp),
                o.row.unwrap_or(u32::MAX),
                u32::from(o.col),
                u32::from(o.cols),
                o.dword_offset,
            ] {
                word(out, v);
            }
        }
    };
    match &s.mesh_emu {
        None => word(&mut out, 0),
        Some(m) => {
            word(&mut out, 1);
            for v in [
                m.threads,
                m.max_vertices,
                m.max_primitives,
                m.vertex_dwords,
                m.prim_dwords,
                u32::from(m.reads_group_id),
            ] {
                word(&mut out, v);
            }
            encode_outputs(&mut out, &m.vertex_outputs);
            encode_outputs(&mut out, &m.prim_outputs);
        }
    }
    out
}

fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
        fn outputs(&mut self) -> Option<Vec<MeshOutput>> {
            let n = self.word()? as usize;
            let mut out = Vec::with_capacity(n.min(256));
            for _ in 0..n {
                let name = String::from_utf8(self.list()?).ok()?;
                let index = self.word()?;
                let kind = u8::try_from(self.word()?).ok()?;
                let comp = u8::try_from(self.word()?).ok()?;
                let interp = u8::try_from(self.word()?).ok()?;
                let row = match self.word()? {
                    u32::MAX => None,
                    r => Some(r),
                };
                let col = u8::try_from(self.word()?).ok()?;
                let cols = u8::try_from(self.word()?).ok()?;
                let dword_offset = self.word()?;
                out.push(MeshOutput {
                    name,
                    index,
                    kind,
                    comp,
                    interp,
                    row,
                    col,
                    cols,
                    dword_offset,
                });
            }
            Some(out)
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        let index = r.word()?;
        let slot = u8::try_from(r.word()?).ok()?;
        let root = match r.word()? {
            0 => RootType::Constants,
            1 => RootType::Descriptor,
            2 => RootType::Table,
            3 => RootType::ImmediateSampler,
            4 => RootType::ResourceHeap,
            5 => RootType::SamplerHeap,
            OTHER_ROOT => RootType::Other(r.word()?),
            _ => return None,
        };
        user_data.push(UserData {
            index,
            slot,
            root,
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let flag = |w: u32| match w {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    };
    let n = r.word()? as usize;
    let mut root_indexed_views = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        root_indexed_views.push(RootIndexedView {
            table_slot: u8::try_from(r.word()?).ok()?,
            constant_slot: u8::try_from(r.word()?).ok()?,
            constant_dword: r.word()?,
            byte_offset: r.word()?,
            shift: u8::try_from(r.word()?).ok().filter(|&s| s < 32)?,
        });
    }
    let mesh_emu = match r.word()? {
        0 => None,
        1 => Some(MeshEmu {
            threads: r.word()?,
            max_vertices: r.word()?,
            max_primitives: r.word()?,
            vertex_dwords: r.word()?,
            prim_dwords: r.word()?,
            reads_group_id: flag(r.word()?)?,
            vertex_outputs: r.outputs()?,
            prim_outputs: r.outputs()?,
        }),
        _ => return None,
    };
    r.0.is_empty().then_some(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_indexed_views,
        mesh_emu,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![1],
            root_indexed_views: vec![RootIndexedView {
                table_slot: 1,
                constant_slot: 2,
                constant_dword: 3,
                byte_offset: 32,
                shift: 5,
            }],
            mesh_emu: None,
        }
    }

    fn mesh_emu_shader() -> HostShader {
        let mut s = shader();
        s.stage = HwStage::Ms;
        s.mesh_emu = Some(MeshEmu {
            threads: 64,
            max_vertices: 32,
            max_primitives: 16,
            vertex_dwords: 8,
            prim_dwords: 4,
            reads_group_id: true,
            vertex_outputs: vec![
                MeshOutput {
                    name: "SV_Position".into(),
                    index: 0,
                    kind: 3,
                    comp: 9,
                    interp: 4,
                    row: None,
                    col: 0,
                    cols: 4,
                    dword_offset: 0,
                },
                MeshOutput {
                    name: "COLOR".into(),
                    index: 0,
                    kind: 0,
                    comp: 9,
                    interp: 2,
                    row: Some(0),
                    col: 0,
                    cols: 4,
                    dword_offset: 4,
                },
            ],
            prim_outputs: vec![MeshOutput {
                name: "PRIMID".into(),
                index: 0,
                kind: 0,
                comp: 5,
                interp: 1,
                row: Some(1),
                col: 0,
                cols: 1,
                dword_offset: 0,
            }],
        });
        s
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            cached(&cache, &build, MeshMode::Native, container, translate),
            Ok(shader())
        );
        assert_eq!(
            cached(&cache, &build, MeshMode::Native, container, translate),
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another converter id caches separately.
        let emu_runs = AtomicU64::new(0);
        let emu_translate = |_: &[u8]| {
            emu_runs.fetch_add(1, Ordering::Relaxed);
            Ok(mesh_emu_shader())
        };
        assert_eq!(
            cached(&cache, &build, MeshMode::Emulate, container, emu_translate),
            Ok(mesh_emu_shader())
        );
        assert_eq!(emu_runs.load(Ordering::Relaxed), 1);

        // Another build translates again.
        assert_eq!(
            cached(&cache, &[8; 32], MeshMode::Native, container, translate),
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH3 broken").unwrap();
        }
        assert_eq!(
            cached(&cache, &build, MeshMode::Native, container, translate),
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(
            cached(&cache, &build, MeshMode::Native, container, translate),
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Failures are not stored.
        let fail = |_: &[u8]| Err::<HostShader, _>("unsupported".to_owned());
        assert!(cached(&cache, &build, MeshMode::Native, b"other", fail).is_err());
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    /// A shader with every field at an unusual value.
    fn rich_shader() -> HostShader {
        HostShader {
            stage: HwStage::Gs,
            dxil: (0u8..=255).collect(),
            user_data: vec![
                UserData {
                    index: 0,
                    slot: 0,
                    root: RootType::Constants,
                    dword: 1,
                },
                UserData {
                    index: 7,
                    slot: 255,
                    root: RootType::Other(0xdead_beef),
                    dword: u32::MAX,
                },
                UserData {
                    index: u32::MAX,
                    slot: 2,
                    root: RootType::SamplerHeap,
                    dword: 3,
                },
            ],
            samplers: vec![[1, 2, 3, 4], [0, 0, 0, 0]],
            static_samplers: vec![
                StaticSampler::from_words([0x400, 0xc0_0000, 0x18f0_0000, 0]),
                StaticSampler::from_words([0, 0, 0, 0]),
            ],
            root_cbvs: vec![0, 1, 255],
            root_srvs: vec![2],
            root_uavs: Vec::new(),
            root_indexed_views: Vec::new(),
            mesh_emu: None,
        }
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        for shader in [shader(), mesh_emu_shader()] {
            let bytes = encode(&shader);
            assert_eq!(decode(&bytes), Some(shader));
            assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
            assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        }
        assert_eq!(decode(b"VSH2"), None);
    }

    #[test]
    fn the_whole_shader_survives_the_round_trip() {
        assert_eq!(decode(&encode(&rich_shader())), Some(rich_shader()));
    }

    #[test]
    fn every_stage_round_trips() {
        let stages = [
            HwStage::Ls,
            HwStage::Hs,
            HwStage::Es,
            HwStage::Gs,
            HwStage::Vs,
            HwStage::Ps,
            HwStage::Ms,
            HwStage::Cs,
        ];
        for stage in stages {
            let mut s = shader();
            s.stage = stage;
            assert_eq!(decode(&encode(&s)), Some(s.clone()), "{stage:?}");
        }
    }

    #[test]
    fn every_root_type_round_trips() {
        let roots = [
            RootType::Constants,
            RootType::Descriptor,
            RootType::Table,
            RootType::ImmediateSampler,
            RootType::ResourceHeap,
            RootType::SamplerHeap,
            RootType::Other(0),
            RootType::Other(1),
            RootType::Other(6),
            RootType::Other(u32::MAX),
        ];
        for root in roots {
            let mut s = shader();
            s.user_data[0].root = root;
            assert_eq!(decode(&encode(&s)), Some(s.clone()), "{root:?}");
        }
    }

    #[test]
    fn every_truncation_is_not_a_shader() {
        let bytes = encode(&rich_shader());
        for n in 0..bytes.len() {
            assert_eq!(decode(&bytes[..n]), None, "{n} of {} bytes", bytes.len());
        }
        assert_eq!(decode(&bytes), Some(rich_shader()));
    }

    #[test]
    fn impossible_lengths_are_not_shaders() {
        let s = rich_shader();
        let bytes = encode(&s);

        // The DXIL length (after the magic and the stage).
        let mut b = bytes.clone();
        b[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode(&b), None, "an oversized DXIL");

        // The user-data count (after the DXIL).
        let at = 12 + s.dxil.len();
        let mut b = bytes.clone();
        b[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode(&b), None, "too many user-data entries");

        // The root-SRV list's length: the bytes it promises are not there.
        let mut at = 12 + s.dxil.len() + 4;
        for u in &s.user_data {
            at += 16 + usize::from(matches!(u.root, RootType::Other(_))) * 4;
        }
        at += 4 + s.samplers.len() * 16;
        at += 4 + s.static_samplers.len() * 40;
        at += 4 + s.root_cbvs.len();
        let mut b = bytes.clone();
        b[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode(&b), None, "a root SRV list that is not there");
    }

    #[test]
    fn unknown_root_tags_are_not_shaders() {
        let s = shader();
        let bytes = encode(&s);
        // The first user-data entry: count, index, slot, then the root tag.
        let at = 12 + s.dxil.len() + 4 + 4 + 4;
        for tag in [7u32, 1000, u32::MAX] {
            let mut b = bytes.clone();
            b[at..at + 4].copy_from_slice(&tag.to_le_bytes());
            assert_eq!(decode(&b), None, "{tag}");
        }
    }

    #[test]
    fn damaged_entries_never_panic() {
        let bytes = encode(&rich_shader());
        for at in 0..bytes.len() {
            for xor in [0x01u8, 0x80, 0xff] {
                let mut b = bytes.clone();
                b[at] ^= xor;
                let _ = decode(&b);
            }
        }
        // Entries that are not one of ours: arbitrary bytes behind the magic.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for len in 0..96 {
            let mut b = b"VSH1".to_vec();
            for _ in 0..len {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                b.push(state as u8);
            }
            let _ = decode(&b);
        }
    }

    #[test]
    fn each_container_and_build_keeps_its_own_entry() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let runs = AtomicU64::new(0);
        let translate = |c: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            let mut s = shader();
            s.dxil = c.to_vec();
            Ok(s)
        };
        let of = |c: &[u8]| {
            let mut s = shader();
            s.dxil = c.to_vec();
            s
        };
        let (a, b) = (b"first container".as_slice(), b"second container");
        assert_eq!(
            cached(&cache, &[7; 32], MeshMode::Native, a, translate),
            Ok(of(a))
        );
        assert_eq!(
            cached(&cache, &[7; 32], MeshMode::Native, b, translate),
            Ok(of(b))
        );
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        assert_eq!(walk(dir.path()).len(), 2, "one entry per container");

        assert_eq!(
            cached(&cache, &[7; 32], MeshMode::Native, a, translate),
            Ok(of(a))
        );
        assert_eq!(
            cached(&cache, &[7; 32], MeshMode::Native, b, translate),
            Ok(of(b))
        );
        assert_eq!(runs.load(Ordering::Relaxed), 2, "both came from the cache");

        // Another build's translation is a separate entry too.
        assert_eq!(
            cached(&cache, &[8; 32], MeshMode::Native, b, translate),
            Ok(of(b))
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(walk(dir.path()).len(), 3);
    }


}
