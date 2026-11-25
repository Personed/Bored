// Included by lower.rs: mesh emulation setup methods.

struct MeshEmuVals {
    uav: V,
    local_group: V,
    group_base: V,
    flat_lt_verts: Option<V>,
    flat_lt_prims: Option<V>,
    index_bytes: u32,
    vertex_stride_bytes: u32,
    prim_offset_bytes: u32,
    prim_stride_bytes: u32,
}

struct MeshEmuState {
    threads: u32,
    max_vertices: u32,
    max_primitives: u32,
    topology: u8,
    vertex_sigs: Vec<SigElement>,
    vertex_offsets: Vec<u32>,
    vertex_dwords: u32,
    prim_sigs: Vec<SigElement>,
    prim_offsets: Vec<u32>,
    prim_dwords: u32,
    reads_group_id: bool,
    vals: Option<MeshEmuVals>,
}

fn pack_offsets(sigs: &[SigElement]) -> (Vec<u32>, u32) {
    let mut offsets = Vec::with_capacity(sigs.len());
    let mut next = 0u32;
    for e in sigs {
        offsets.push(next);
        next += u32::from(e.cols);
    }
    (offsets, next)
}

impl Lower<'_> {
    fn mesh_emu_init(&mut self) -> Result<(), Why> {
        let (threads, raw_verts, raw_prims, topology, vertex_dwords, prim_dwords) = {
            let m = self.mesh_emu.as_ref().expect("mesh emu");
            (
                m.threads,
                m.max_vertices,
                m.max_primitives,
                m.topology,
                m.vertex_dwords,
                m.prim_dwords,
            )
        };
        if topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads == 0 || threads > 1024 || raw_verts > 256 || raw_prims > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        let max_vertices = raw_verts.min(threads);
        let max_primitives = raw_prims.min(threads);
        if max_vertices == 0 || max_primitives == 0 {
            return Err("an empty mesh shader output");
        }
        let index_bytes = max_primitives * 4;
        let vertex_stride_bytes = vertex_dwords * 4;
        let prim_offset_bytes = index_bytes + max_vertices * vertex_stride_bytes;
        let prim_stride_bytes = prim_dwords * 4;
        let group_bytes = prim_offset_bytes + max_primitives * prim_stride_bytes;

        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let uav = self.s.emu_uav_handle();
        let k0 = self.s.m.i32(0);
        let local_group = self.s.op(e, "groupId.i32", Attr::ReadNone, i32t, 94, &[k0]);
        let flat = self.mesh_thread();
        let group_stride = self.s.m.i32(group_bytes);
        let group_base = self.s.m.bin(e, Bin::Mul, local_group, group_stride);
        let flat_lt_verts = (threads > max_vertices).then(|| {
            let mv = self.s.m.i32(max_vertices);
            self.s.m.cmp(e, pred::IULT, flat, mv)
        });
        let flat_lt_prims = (threads > max_primitives).then(|| {
            let mp = self.s.m.i32(max_primitives);
            self.s.m.cmp(e, pred::IULT, flat, mp)
        });
        let m = self.mesh_emu.as_mut().expect("mesh emu");
        m.max_vertices = max_vertices;
        m.max_primitives = max_primitives;
        m.vals = Some(MeshEmuVals {
            uav,
            local_group,
            group_base,
            flat_lt_verts,
            flat_lt_prims,
            index_bytes,
            vertex_stride_bytes,
            prim_offset_bytes,
            prim_stride_bytes,
        });
        Ok(())
    }

    /// Initializes every primitive index word in the group slice as culled
    /// (`bit 31 = 1`) so unexported slots never draw stale data.
    fn mesh_emu_prologue(&mut self) {
        let (uav, group_base, flat_lt_prims) = {
            let v = self
                .mesh_emu
                .as_ref()
                .expect("mesh emu")
                .vals
                .as_ref()
                .expect("mesh emu init");
            (v.uav, v.group_base, v.flat_lt_prims)
        };
        let flat = self.mesh_thread();
        let at = self.begin_guarded(flat_lt_prims);
        let c = self.cur;
        let k2 = self.s.m.i32(2);
        let byte_off = self.s.m.bin(c, Bin::Shl, flat, k2);
        let addr = self.s.m.bin(c, Bin::Add, group_base, byte_off);
        let undef = self.s.m.undef(self.s.t.i32);
        let culled = self.s.m.i32(0x8000_0000);
        let mask1 = self.s.m.i8(1);
        let align4 = self.s.m.i32(4);
        let void = self.s.t.void;
        self.s.op(
            c,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            void,
            140,
            &[uav, addr, undef, culled, undef, undef, undef, mask1, align4],
        );
        self.end_guarded(at, &[]);
    }
}
