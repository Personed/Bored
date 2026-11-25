fn mesh_emu_state(ms: &crate::native::MeshIo, x: &Parts) -> MeshEmuState {
    let (vertex_offsets, vertex_dwords) = pack_offsets(&x.outputs);
    let (prim_offsets, prim_dwords) = pack_offsets(&x.prims);
    MeshEmuState {
        threads: ms.threads,
        max_vertices: ms.max_vertices,
        max_primitives: ms.max_primitives,
        topology: ms.topology,
        vertex_sigs: x.outputs.clone(),
        vertex_offsets,
        vertex_dwords,
        prim_sigs: x.prims.clone(),
        prim_offsets,
        prim_dwords,
        reads_group_id: false,
        vals: None,
    }
}

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


impl MeshEmuState {
    fn finish(self) -> super::MeshEmu {
        let to_outputs = |sigs: Vec<SigElement>, offsets: Vec<u32>| {
            sigs.into_iter()
                .zip(offsets)
                .map(|(s, dword_offset)| super::MeshOutput {
                    name: s.name,
                    index: s.index,
                    kind: s.kind,
                    comp: s.comp,
                    interp: s.interp,
                    row: s.row,
                    col: s.col,
                    cols: s.cols,
                    dword_offset,
                })
                .collect()
        };
        super::MeshEmu {
            threads: self.threads,
            max_vertices: self.max_vertices,
            max_primitives: self.max_primitives,
            vertex_dwords: self.vertex_dwords,
            prim_dwords: self.prim_dwords,
            reads_group_id: self.reads_group_id,
            vertex_outputs: to_outputs(self.vertex_sigs, self.vertex_offsets),
            prim_outputs: to_outputs(self.prim_sigs, self.prim_offsets),
        }
    }
}

impl<'a> Lower<'a> {
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

    fn combine_guards(&mut self, a: Option<V>, b: Option<V>) -> Option<V> {
        match (a, b) {
            (Some(x), Some(y)) => Some(self.s.m.bin(self.cur, Bin::And, x, y)),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        }
    }

    /// Emulated mesh: `exp prim` writes the packed primitive index word
    /// (preserving bit 31 as the cull flag) to `u0, space 2`.
    fn emit_indices_emu(&mut self, e: &Export, guard: Option<V>) -> Result<(), Why> {
        let v = e.values[0].ok_or("a primitive export without its indices")?;
        let x = self.i32_of(v)?;
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
        let guard = self.combine_guards(guard, flat_lt_prims);
        let flat = self.mesh_thread();
        let at = self.begin_guarded(guard);
        let c = self.cur;
        let k2 = self.s.m.i32(2);
        let byte_off = self.s.m.bin(c, Bin::Shl, flat, k2);
        let addr = self.s.m.bin(c, Bin::Add, group_base, byte_off);
        let undef = self.s.m.undef(self.s.t.i32);
        let mask1 = self.s.m.i8(1);
        let align4 = self.s.m.i32(4);
        let void = self.s.t.void;
        self.s.op(
            c,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            void,
            140,
            &[uav, addr, undef, x, undef, undef, undef, mask1, align4],
        );
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Emulated mesh: stores vertex (`primitive == false`) or per-primitive
    /// (`primitive == true`) exported scalars into `u0, space 2`, coalescing
    /// contiguous dword slots up to 4 dwords per store.
    fn store_mesh_emu_outputs(
        &mut self,
        guard: Option<V>,
        primitive: bool,
        stores: &[(usize, u8, Value, Option<bool>)],
    ) -> Result<(), Why> {
        if stores.is_empty() {
            return Ok(());
        }
        let mut items: Vec<(u32, V, Option<bool>)> = Vec::with_capacity(stores.len());
        for &(id, col, value, half) in stores {
            let x = self.i32_of(value)?;
            let slot = {
                let m = self.mesh_emu.as_mut().expect("mesh emu");
                let (sigs, offsets) = if primitive {
                    (&mut m.prim_sigs, &m.prim_offsets)
                } else {
                    (&mut m.vertex_sigs, &m.vertex_offsets)
                };
                sigs[id].usage |= 1 << col;
                offsets[id] + u32::from(col)
            };
            items.push((slot, x, half));
        }
        items.sort_by_key(|&(slot, _, _)| slot);
        let mut deduped: Vec<(u32, V, Option<bool>)> = Vec::with_capacity(items.len());
        for item in items {
            if let Some(last) = deduped.last_mut()
                && last.0 == item.0
            {
                *last = item;
            } else {
                deduped.push(item);
            }
        }
        let (uav, group_base, bound_guard, base_offset, stride_bytes) = {
            let v = self
                .mesh_emu
                .as_ref()
                .expect("mesh emu")
                .vals
                .as_ref()
                .expect("mesh emu init");
            if primitive {
                (
                    v.uav,
                    v.group_base,
                    v.flat_lt_prims,
                    v.prim_offset_bytes,
                    v.prim_stride_bytes,
                )
            } else {
                (
                    v.uav,
                    v.group_base,
                    v.flat_lt_verts,
                    v.index_bytes,
                    v.vertex_stride_bytes,
                )
            }
        };
        let guard = self.combine_guards(guard, bound_guard);
        let flat = self.mesh_thread();
        let at = self.begin_guarded(guard);
        let c = self.cur;
        let mut converted = Vec::with_capacity(deduped.len());
        for (slot, mut x, half) in deduped {
            if let Some(high) = half {
                if high {
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                let f32t = self.s.t.f32;
                let f = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                x = self.as_int(f);
            }
            converted.push((slot, x));
        }
        let stride = self.s.m.i32(stride_bytes);
        let rel = self.s.m.bin(c, Bin::Mul, flat, stride);
        let off = if base_offset == 0 {
            rel
        } else {
            let base_k = self.s.m.i32(base_offset);
            self.s.m.bin(c, Bin::Add, base_k, rel)
        };
        let rec_base = self.s.m.bin(c, Bin::Add, group_base, off);
        let undef = self.s.m.undef(self.s.t.i32);
        let align4 = self.s.m.i32(4);
        let void = self.s.t.void;
        let mut idx = 0;
        while idx < converted.len() {
            let start_slot = converted[idx].0;
            let mut len = 1usize;
            while len < 4
                && idx + len < converted.len()
                && converted[idx + len].0 == start_slot + len as u32
            {
                len += 1;
            }
            let addr = if start_slot == 0 {
                rec_base
            } else {
                let disp = self.s.m.i32(start_slot * 4);
                self.s.m.bin(c, Bin::Add, rec_base, disp)
            };
            let v0 = converted[idx].1;
            let v1 = if len > 1 { converted[idx + 1].1 } else { undef };
            let v2 = if len > 2 { converted[idx + 2].1 } else { undef };
            let v3 = if len > 3 { converted[idx + 3].1 } else { undef };
            let mask = self.s.m.i8((1u8 << len) - 1);
            self.s.op(
                c,
                "rawBufferStore.i32",
                Attr::NoUnwind,
                void,
                140,
                &[uav, addr, undef, v0, v1, v2, v3, mask, align4],
            );
            idx += len;
        }
        self.end_guarded(at, &[]);
        Ok(())
    }
    fn mesh_group_emu(&mut self) -> Val {
        let m = self.mesh_emu.as_mut().expect("mesh emu");
        m.reads_group_id = true;
        let local_group = m.vals.as_ref().expect("mesh emu init").local_group;
        let cbv = self.s.emu_cbv_handle();
        let srv = self.s.emu_srv_handle();
        let c = self.cur;
        let i32t = self.s.t.i32;
        let k0 = self.s.m.i32(0);
        let base_ret = self.s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[cbv, k0],
        );
        let base = self.s.m.extract(c, base_ret, 0);
        let flag = self.s.m.i32(super::INDIRECT_SLOT_MAP_FLAG);
        let is_map = self.s.m.cmp(c, pred::EQ, base, flag);
        let map_bb = self.s.m.new_block();
        let add_bb = self.s.m.new_block();
        let join_bb = self.s.m.new_block();
        self.s.m.terminate(c, Term::CondBr(is_map, map_bb, add_bb));

        let k2 = self.s.m.i32(2);
        let off = self.s.m.bin(map_bb, Bin::Shl, local_group, k2);
        let undef = self.s.m.undef(i32t);
        let mask1 = self.s.m.i8(1);
        let align4 = self.s.m.i32(4);
        let r = self.s.op(
            map_bb,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[srv, off, undef, mask1, align4],
        );
        let f = self.s.m.extract(map_bb, r, 0);
        let g_map = self.s.m.cast(map_bb, Cast::Bitcast, f, i32t);
        self.s.m.terminate(map_bb, Term::Br(join_bb));

        let g_add = self.s.m.bin(add_bb, Bin::Add, local_group, base);
        self.s.m.terminate(add_bb, Term::Br(join_bb));

        let g = self.s.m.phi(join_bb, i32t);
        self.s.m.set_phi(g, vec![(g_map, map_bb), (g_add, add_bb)]);
        let n = self.s.m.i32(self.x.mesh_stride);
        let res = self.s.m.bin(join_bb, Bin::Mul, g, n);
        self.cur = join_bb;
        Val::I32(res)
    }

}
