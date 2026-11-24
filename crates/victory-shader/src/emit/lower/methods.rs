impl<'a> Lower<'a> {
    fn exit(&mut self, b: usize) -> Result<(), Stop> {
        let offset = self.x.p.blocks[b]
            .insts
            .last()
            .map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        self.cur_reads.clear();
        self.cur_old_deps.clear();
        self.cur_edge = None;
        let term = match self.x.st.exits[b] {
            LaneExit::Jump(t) if self.counters.contains_key(&t) && self.rpo[t] <= self.rpo[b] => {
                let ok = self.count_iteration(t, pred::IULT);
                let abort = self.abort_block();
                Term::CondBr(ok, self.block[t].expect("reachable"), abort)
            }
            LaneExit::Jump(t) => {
                self.enter_loop(b, t, self.cur);
                Term::Br(self.block[t].expect("reachable"))
            }
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // A negated condition branches the other way round.
                let (cond, taken, not_taken) = match self.x.st.bools.node(when) {
                    _ if self.x.st.wide.contains(&b) => {
                        let c = self
                            .wave_wide(b, when)
                            .map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        let a = self
                            .atom(atom, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (a, not_taken, taken)
                    }
                    _ => {
                        let c = self
                            .boolean(when, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                };
                let reads = std::mem::take(&mut self.cur_reads);
                if !reads.is_empty() {
                    self.sinks.push(SinkRead {
                        block: b,
                        guard: None,
                        offset,
                        deps: reads,
                    });
                }
                // A back edge is taken only below the iteration limit.
                let back =
                    |l: &Self, to: usize| l.counters.contains_key(&to) && l.rpo[to] <= l.rpo[b];
                // (No `xor` of a compare: a loop exit testing one hung the
                // AMD driver.)
                let cond = if back(self, taken) {
                    let ok = self.count_iteration(taken, pred::IULT);
                    self.s.m.bin(self.cur, Bin::And, cond, ok)
                } else if back(self, not_taken) {
                    let stop = self.count_iteration(not_taken, pred::IUGE);
                    self.s.m.bin(self.cur, Bin::Or, cond, stop)
                } else {
                    cond
                };
                // Critical edges get a block of their own, so loops have
                // dedicated exits and preheaders as DXC emits them.
                let (t, f) = (self.edge(b, taken), self.edge(b, not_taken));
                for (to, via) in [(taken, t), (not_taken, f)] {
                    if !back(self, to) {
                        // From an edge block, or from here.
                        let from = if Some(via) == self.block[to] {
                            self.cur
                        } else {
                            via
                        };
                        self.enter_loop(b, to, from);
                    }
                }
                Term::CondBr(cond, t, f)
            }
            LaneExit::End => {
                if self.s.pixel && !self.exported[b] && !self.has_done_export(b) {
                    // A lane leaving without its final export was killed.
                    let t = self.s.m.bool(true);
                    self.s
                        .op(self.cur, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
                }
                Term::Br(self.ret_block())
            }
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        };
        let _ = Exit::End;
        self.s.m.terminate(self.cur, term);
        Ok(())
    }

    /// A forward edge into loop header `to` (from DXIL block `via`) starts
    /// its count at 0.
    fn enter_loop(&mut self, _from: usize, to: usize, via: usize) {
        if let Some((_, incoming)) = self.counters.get_mut(&to) {
            let z = self.s.m.i32(0);
            incoming.push((z, via));
        }
    }

    /// A back edge to header `to` (a back edge keeps its block, no
    /// critical-edge block): the count goes up; the result compares it with
    /// [`LOOP_LIMIT`] by `p` (below: go on; at or above: stop).
    fn count_iteration(&mut self, to: usize, p: u8) -> V {
        let c = self.cur;
        let (phi, _) = self.counters[&to];
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(c, Bin::Add, phi, one);
        // Per stage unless the switch names one: a loop that never meets its
        // exit (a count read from a buffer the host has not got yet, say)
        // runs to the limit in every lane. A million iterations a pixel
        // hung the GPU for seconds a draw until Windows reset it
        // (PS_Textured_Eight, 2026-11-24, run 12).
        // shortcut: a pixel loop past 128 iterations, a vertex or mesh loop
        // past 256 or a compute loop past 4,096 is cut short. (They were
        // 1,024, 4,096 and 65,536: the opening cutscene then lost the card
        // 14-25 frames in, in a minimap or shadow mesh draw; with every
        // loop cut at 64 it ran 90 frames. Raise them again once shaders
        // stop reading data that is not there yet.)
        let limit = match super::loop_limit() {
            LOOP_LIMIT if self.s.pixel => 1 << 7,
            LOOP_LIMIT if self.s.mesh.is_some() || self.s.threads.is_none() => 1 << 8,
            LOOP_LIMIT => 1 << 12,
            named => named,
        };
        let limit = self.s.m.i32(limit);
        // The breadcrumb may end the block (a store of one lane), so the
        // back edge leaves from where it ends.
        self.crumb(to, next);
        let c = self.cur;
        self.counters
            .get_mut(&to)
            .expect("header")
            .1
            .push((next, c));
        self.s.m.cmp(c, p, next, limit)
    }

    /// Breadcrumbs (see [`super::breadcrumbs`]): the count of the loop at
    /// header `to` goes to its word of the shader's slot, as the largest
    /// any lane reached: the wave's largest, stored by its first active lane
    /// (one atomic a wave, not one a lane). The buffer's UAV is the view
    /// heap's entry at the index in the first user-data dword past the
    /// console's.
    #[allow(dead_code)]
    fn crumb(&mut self, to: usize, count: V) {
        let Some(slot) = self.x.crumbs else { return };
        let next = self.crumb_loops.len() as u32;
        let n = *self.crumb_loops.entry(to).or_insert(next);
        if n >= super::BREADCRUMB_LOOPS {
            return;
        }
        let first = self
            .x
            .native
            .layout
            .iter()
            .map(|r| u32::from(r.user_data) + r.sgprs)
            .max()
            .unwrap_or(0);
        let index = self.user_dword(UserData {
            index: first,
            slot: 0,
            root: super::BREADCRUMB_ROOT,
            dword: 0,
        });
        let c = self.cur;
        let (i1, i32t) = (self.s.t.i1, self.s.t.i32);
        self.s.flags |= flags::WAVE_OPS;
        // The largest count of the lanes that take the back edge (unsigned max).
        let (k, u) = (self.s.m.i8(3), self.s.m.i8(1));
        let count = self.s.op(
            c,
            "waveActiveOp.i32",
            Attr::NoUnwind,
            i32t,
            119,
            &[count, k, u],
        );
        let first = self
            .s
            .op(c, "waveIsFirstLane", Attr::NoUnwind, i1, 110, &[]);
        let guarded = self.begin_guarded(Some(first));
        let c = self.cur;
        // A raw buffer UAV (`atomicMax`, operation 7, at the word's byte offset).
        let h = self.s.heap_handle(c, index, false, false, (11 | 0x1000, 0));
        let at = self.s.m.i32((slot * super::BREADCRUMB_LOOPS + n) * 4);
        let (op, unused) = (self.s.m.i32(7), self.s.m.undef(i32t));
        self.s.op(
            c,
            "atomicBinOp.i32",
            Attr::NoUnwind,
            i32t,
            78,
            &[h, op, at, unused, unused, count],
        );
        self.end_guarded(guarded, &[]);
    }

    /// The block a lane stopped by [`LOOP_LIMIT`] leaves through.
    fn abort_block(&mut self) -> usize {
        if let Some(a) = self.abort {
            return a;
        }
        let a = self.s.m.new_block();
        if self.s.pixel {
            let t = self.s.m.bool(true);
            self.s
                .op(a, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
        }
        let r = self.ret_block();
        self.s.m.terminate(a, Term::Br(r));
        self.abort = Some(a);
        a
    }

    /// The function's single returning block (see `ret`).
    fn ret_block(&mut self) -> usize {
        if let Some(r) = self.ret {
            return r;
        }
        let r = self.s.m.new_block();
        self.s.m.terminate(r, Term::Ret);
        self.ret = Some(r);
        r
    }

    /// The DXIL block a branch from IR block `from` to `to` goes to.
    fn edge(&mut self, from: usize, to: usize) -> usize {
        let target = self.block[to].expect("reachable");
        let preds = self.x.p.blocks[to]
            .preds
            .iter()
            .filter(|&&q| self.block[q].is_some())
            .count();
        // Back edges stay as they are: the exiting block is the latch.
        if preds < 2 || self.rpo[to] <= self.rpo[from] {
            return target;
        }
        let e = self.s.m.new_block();
        self.s.m.terminate(e, Term::Br(target));
        self.edges.insert((from, to), e);
        e
    }

    fn fill_phis(&mut self) -> Result<(), Stop> {
        let p = self.x.p;
        let data_phis = std::mem::take(&mut self.data_phis);
        let mut has_missing = false;
        // Lane-mask joins first (what an edge brings may ask for more joins,
        // of masks and of numbers): each edge's mask as the structure
        // tracked it, a constant or a scalar-built one included.
        let st = self.x.st;
        while let Some((v, b, i)) = self.pending_masks.pop() {
            let phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let args = st
                .phi_masks
                .get(&phi.value)
                .ok_or_else(|| unsupported(offset, "a lane-mask join of an untracked mask"))?;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let x = self
                    .boolean(args[n], qb)
                    .map_err(|w| unsupported(offset, w))?;
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        // A join that only another join reads is read wherever that one is. Left
        // unmarked, a value without a DXIL form coming into it became zero without a
        // word (the looked-up tint of the material resolve programs, whose size
        // computation had no value, was zero in every lane that way).
        // shortcut: only with `VICTORY_STRICT_JOINS` set (the offline pack
        // listing sets it to count them): 40 programs that translate today
        // stop by name with it (vehicle paint pixel shaders at "the first
        // half of an interpolation", foliage mesh shaders); until those are
        // lowered they keep drawing with that zero.
        static STRICT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let strict = *STRICT.get_or_init(|| std::env::var_os("VICTORY_STRICT_JOINS").is_some());
        let mut work: Vec<Value> = if strict {
            self.read_phis.iter().copied().collect()
        } else {
            Vec::new()
        };
        while let Some(v) = work.pop() {
            let Some(&(b, i)) = self.phi_at.get(&v) else {
                continue;
            };
            for &a in &p.blocks[b].phis[i].args {
                if self.phi_at.contains_key(&a) && self.read_phis.insert(a) {
                    work.push(a);
                }
            }
        }
        for &(v, b, i) in &data_phis {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            // A join of numbers with dead registers (`dead`): the paths that bring those
            // never read the result, so they bring zero. Not for a join that is itself a
            // lane mask (a ballot gives that one as data).
            let st = self.x.st;
            let numbers = !bit
                && !st.masks.contains_key(&phi.value)
                && !st.highs.contains_key(&phi.value)
                && phi
                    .args
                    .iter()
                    .any(|a| matches!(self.vals.get(a), Some(Val::I32(_))));
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                if self.read_phis.contains(&phi.value) && !self.vals.contains_key(&arg) {
                    let _ = self.lazy_barycentric(arg);
                    // A root parameter nothing read before the join (user
                    // data is loaded on first use, in the entry block).
                    if !self.vals.contains_key(&arg)
                        && let Some(&(param, dword)) = self.params.get(&arg)
                    {
                        let val = self.user(param, dword);
                        self.vals.insert(arg, val);
                    }
                }
                let x = match (self.vals.get(&arg).copied(), bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                    // A lane mask's dword reaching a join that also takes
                    // data: the registers are reused once the mask is dead.
                    // shortcut: undefined on that edge, assuming the program
                    // reads the joined value only on paths from a data edge.
                    (Some(Val::Missing(_)), false)
                        if self.x.st.masks.contains_key(&arg)
                            || self.x.st.highs.contains_key(&arg) =>
                    {
                        self.s.m.undef(self.s.t.i32)
                    }
                    (Some(Val::Missing(_)) | None, false)
                        if numbers && self.dead(arg, &mut Vec::new()) =>
                    {
                        has_missing = true;
                        self.s.m.i32(0)
                    }
                    // A lane mask joined with numbers (a traversal that keeps its lanes' masks on
                    // a stack with its node numbers): the mask comes as the number it is, the
                    // ballot of its lanes where it is written.
                    (Some(Val::Missing(_)) | None, false)
                        if self.read_phis.contains(&phi.value)
                            && self.mask_data == MaskData::Console
                            && (st.masks.contains_key(&arg) || st.highs.contains_key(&arg)) =>
                    {
                        match self.mask_dword(arg) {
                            Ok(x) => x,
                            Err(why) => return Err(Stop::Retry(phi.value, why)),
                        }
                    }
                    // SCC of a mask operation ("some lane is left") joined with a scalar test:
                    // the compiler branches around code that sets SCC and tests SCC after it,
                    // the skip bringing "no lane" along. As data it is "some lane of the wave
                    // has the bit" - right only if the wave took that skip together, as the
                    // console does: a lane that skipped alone would bring "some lane" where the
                    // console has the skipped code's SCC. So the branch must be one the wave
                    // takes together; else its region is named for `emit`'s retry to widen it.
                    (Some(Val::Missing(_)) | None, true)
                        if self.read_phis.contains(&phi.value)
                            && self.mask_data == MaskData::Console
                            && st.scc_some.contains_key(&arg) =>
                    {
                        let per_lane = !st.wide.contains(&q)
                            && matches!(st.exits[q], LaneExit::Branch { when, .. }
                                if !st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(_))));
                        if per_lane {
                            for &other in p.blocks[b].preds.iter().filter(|&&o| o != q) {
                                need(other);
                            }
                            return Err(Stop::Retry(phi.value, LANE_MASK));
                        }
                        match self.scc_some(arg) {
                            Ok(x) => x,
                            Err(why) => return Err(Stop::Retry(phi.value, why)),
                        }
                    }
                    (Some(Val::Bit(_)), false) if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, "an SCC joined with SGPR values"));
                    }
                    (Some(Val::Missing(why)), _) if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, why));
                    }
                    _ => {
                        has_missing = true;
                        if bit {
                            self.s.m.bool(false)
                        } else {
                            self.s.m.i32(0)
                        }
                    }
                };
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        for (phi, incoming) in std::mem::take(&mut self.counters).into_values() {
            self.s.m.set_phi(phi, incoming);
        }
        while let Some((v, b, i)) = self.pending_masks.pop() {
            let phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let m =
                    *self.x.st.masks.get(&phi.args[n]).ok_or_else(|| {
                        unsupported(offset, "a lane-mask join of an untracked mask")
                    })?;
                self.cur_reads.clear();
                self.cur_edge = None;
                let x = self.boolean(m, qb).map_err(|w| unsupported(offset, w))?;
                let reads = std::mem::take(&mut self.cur_reads);
                if !reads.is_empty() {
                    let deps = reads
                        .into_iter()
                        .map(|dep| match dep {
                            ReadDep::Value(value) => ReadDep::Edge {
                                from: q,
                                to: b,
                                value,
                            },
                            other => other,
                        })
                        .collect();
                    self.sinks.push(SinkRead {
                        block: q,
                        guard: None,
                        offset,
                        deps,
                    });
                }
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        if (!has_missing && self.direct_missing.is_empty()) || self.sinks.is_empty() {
            return Ok(());
        }
        self.check_missing_reach(&data_phis)
    }

    /// Rejects the program if any missing or mismatched phi incoming edge (or
    /// direct missing value surviving via a partial/guarded overwrite) can
    /// reach an observable sink along a feasible path in [`Structured`].
    fn check_missing_reach(&self, data_phis: &[(V, usize, usize)]) -> Result<(), Stop> {
        const MAX_TAINT_ROUNDS: usize = 256;
        let p = self.x.p;
        let st = self.x.st;
        let mut bools = BoolSet::new(&st.bools);
        let mut taint: HashMap<Value, Vec<(Why, Bool)>> = HashMap::new();

        let site_cond =
            |bools: &mut BoolSet, block: usize, guard: Option<Bool>, invert: bool| -> Bool {
                match guard {
                    Some(g) => {
                        let g = if invert { bools.not(g) } else { g };
                        bools.and(st.reach[block], g)
                    }
                    None => st.reach[block],
                }
            };
        let dep_cond = |bools: &mut BoolSet, base: Bool, dep: ReadDep| -> (Value, Bool) {
            match dep {
                ReadDep::Value(u) => (u, base),
                ReadDep::Edge { from, to, value: u } => {
                    let ec = bools.edge(st, from, to);
                    (u, bools.and(base, ec))
                }
            }
        };
        let merge = |taint: &mut HashMap<Value, Vec<(Why, Bool)>>,
                     dst: Value,
                     why: Why,
                     cond: Bool,
                     bools: &mut BoolSet|
         -> bool {
            if cond == Bool::FALSE {
                return false;
            }
            let list = taint.entry(dst).or_default();
            if let Some((_, existing)) = list.iter_mut().find(|(w, _)| *w == why) {
                let next = bools.or(*existing, cond);
                if next != *existing {
                    *existing = next;
                    return true;
                }
                false
            } else {
                list.push((why, cond));
                true
            }
        };

        for &(dst, why, block, guard, invert) in &self.direct_missing {
            let c = site_cond(&mut bools, block, guard, invert);
            merge(&mut taint, dst, why, c, &mut bools);
        }

        for &(_, b, i) in data_phis {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                if self.tail[q].is_none() {
                    continue;
                }
                let arg = phi.args[n];
                let why = match (self.vals.get(&arg).copied(), bit) {
                    (Some(Val::I32(_)), false) | (Some(Val::Bit(_)), true) => continue,
                    (Some(Val::Bit(_)), false) => "an SCC joined with SGPR values",
                    (Some(Val::Missing(why)), _) => why,
                    _ => "a join of mismatched values",
                };
                let mut c = bools.edge(st, q, b);
                if self.rpo[b] <= self.rpo[q] {
                    c = bools.forget(c, self.rpo[b]);
                }
                merge(&mut taint, phi.value, why, c, &mut bools);
            }
        }

        let propagate_step =
            |taint: &mut HashMap<Value, Vec<(Why, Bool)>>, bools: &mut BoolSet, precise: bool| {
                let mut changed = false;
                for &(_, b, i) in data_phis {
                    let phi = &p.blocks[b].phis[i];
                    for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                        if self.tail[q].is_none() {
                            continue;
                        }
                        let Some(list) = taint.get(&phi.args[n]).cloned() else {
                            continue;
                        };
                        let ec = if precise {
                            bools.edge(st, q, b)
                        } else {
                            Bool::TRUE
                        };
                        for (why, c) in list {
                            let tc = if precise {
                                let mut tc = bools.and(c, ec);
                                if self.rpo[b] <= self.rpo[q] {
                                    tc = bools.forget(tc, self.rpo[b]);
                                }
                                tc
                            } else {
                                Bool::TRUE
                            };
                            changed |= merge(taint, phi.value, why, tc, bools);
                        }
                    }
                }
                for &w in &self.deps_order {
                    for info in &self.deps[&w] {
                        let use_c = if precise {
                            site_cond(bools, info.block, info.guard, info.invert_guard)
                        } else {
                            Bool::TRUE
                        };
                        for &dep in &info.deps {
                            let (val, dc) = if precise {
                                dep_cond(bools, use_c, dep)
                            } else {
                                let u = match dep {
                                    ReadDep::Value(u) | ReadDep::Edge { value: u, .. } => u,
                                };
                                (u, Bool::TRUE)
                            };
                            let Some(list) = taint.get(&val).cloned() else {
                                continue;
                            };
                            for (why, c) in list {
                                if info.ignore_why == Some(why) {
                                    continue;
                                }
                                let tc = if precise {
                                    bools.and(c, dc)
                                } else {
                                    Bool::TRUE
                                };
                                changed |= merge(taint, w, why, tc, bools);
                            }
                        }
                    }
                }
                changed
            };

        let mut settled = false;
        for _ in 0..MAX_TAINT_ROUNDS {
            if !propagate_step(&mut taint, &mut bools, true) {
                settled = true;
                break;
            }
        }
        if !settled {
            // Conservative widening when cyclic propagation exceeds the round cap.
            for list in taint.values_mut() {
                for (_, c) in list.iter_mut() {
                    *c = Bool::TRUE;
                }
            }
            while propagate_step(&mut taint, &mut bools, false) {}
        }

        for sink in &self.sinks {
            let sink_c = site_cond(&mut bools, sink.block, sink.guard, false);
            for &dep in &sink.deps {
                let (val, dc) = dep_cond(&mut bools, sink_c, dep);
                if let Some(list) = taint.get(&val) {
                    for &(why, c) in list {
                        if bools.sat(c, dc) {
                            return Err(unsupported(sink.offset, why));
                        }
                    }
                }
            }
        }
        Ok(())
    }

}
