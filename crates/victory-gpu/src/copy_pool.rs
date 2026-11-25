//! A submission's refreshes, copied on several threads.
//!
//! Bringing host copies of title memory up to date is a memory copy, and
//! one thread copies far below what the memory can move. Refreshes inside
//! [`queued`] queue their copies, and [`run`] copies them on a few workers
//! with the submitting thread; it returns once all are done, so what
//! follows sees them all.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// A copy of title memory into host memory: destination, source, bytes.
pub(crate) type Piece = (usize, usize, usize);

thread_local! {
    static QUEUE: RefCell<Option<Vec<Piece>>> = const { RefCell::new(None) };
}

/// Runs `f`; the copies [`copy`] is asked for on this thread meanwhile are
/// not made but returned, for [`run`].
pub(crate) fn queued<R>(f: impl FnOnce() -> R) -> (R, Vec<Piece>) {
    let outer = QUEUE.with(|q| q.replace(Some(Vec::new())));
    let result = f();
    let pieces = QUEUE.with(|q| q.replace(outer)).unwrap_or_default();
    (result, pieces)
}

/// Copies `len` bytes of title memory at `src` into host memory at `dst`
/// ([`crate::host::copy_mapped`]), or queues the copy inside [`queued`].
/// The caller must not read the destination before [`run`] then.
pub(crate) fn copy(dst: *mut u8, src: *const u8, len: usize) {
    let piece = (dst as usize, src as usize, len);
    let queued = QUEUE.with(|q| q.borrow_mut().as_mut().map(|q| q.push(piece)).is_some());
    if !queued {
        crate::host::copy_mapped(dst, src, len, false);
    }
}

/// Copies are cut to this size, so that one big copy is shared too.
const PIECE: usize = 1 << 20;
/// Below this many bytes in all, one thread copies (waking the workers
/// costs tens of microseconds; 4 MiB take about 0.4 ms).
const PARALLEL_FROM: usize = 4 << 20;
/// Workers besides the submitting thread.
const WORKERS: usize = 3;

/// Makes the copies [`queued`] returned, on several threads where they are
/// many bytes; returns once all are made.
pub(crate) fn run(pieces: Vec<Piece>) {
    let total: usize = pieces.iter().map(|p| p.2).sum();
    let Some(pool) = (total >= PARALLEL_FROM).then(pool).flatten() else {
        for (dst, src, len) in pieces {
            crate::host::copy_mapped(dst as *mut u8, src as *const u8, len, false);
        }
        return;
    };
    let mut cut = Vec::with_capacity(pieces.len() + total / PIECE);
    for (dst, src, len) in pieces {
        let mut at = 0;
        while at < len {
            let n = (len - at).min(PIECE);
            cut.push((dst + at, src + at, n));
            at += n;
        }
    }
    pool.run(cut);
}

/// One [`run`]'s pieces, taken by index by whichever thread is free.
struct Job {
    pieces: Vec<Piece>,
    next: AtomicUsize,
    done: AtomicUsize,
}

impl Job {
    /// Copies pieces until none are left to take.
    fn work(&self) {
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            let Some(&(dst, src, len)) = self.pieces.get(i) else {
                return;
            };
            crate::host::copy_mapped(dst as *mut u8, src as *const u8, len, false);
            self.done.fetch_add(1, Ordering::Release);
        }
    }
}

struct Pool {
    /// The current job and its number (workers take each job once).
    job: Mutex<(u64, Option<Arc<Job>>)>,
    wake: Condvar,
    /// Signalled as workers finish their part of a job.
    finished: Condvar,
    /// One run at a time.
    running: Mutex<()>,
}

impl Pool {
    fn run(&self, pieces: Vec<Piece>) {
        let _one = self.running.lock().unwrap_or_else(|e| e.into_inner());
        let n = pieces.len();
        let job = Arc::new(Job {
            pieces,
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
        });
        {
            let mut slot = self.job.lock().unwrap_or_else(|e| e.into_inner());
            slot.0 += 1;
            slot.1 = Some(job.clone());
        }
        self.wake.notify_all();
        job.work();
        let mut slot = self.job.lock().unwrap_or_else(|e| e.into_inner());
        while job.done.load(Ordering::Acquire) < n {
            slot = self.finished.wait(slot).unwrap_or_else(|e| e.into_inner());
        }
        slot.1 = None;
    }

    fn worker(&self) {
        let mut seen = 0;
        loop {
            let job = {
                let mut slot = self.job.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if slot.0 != seen
                        && let Some(job) = &slot.1
                    {
                        seen = slot.0;
                        break job.clone();
                    }
                    slot = self.wake.wait(slot).unwrap_or_else(|e| e.into_inner());
                }
            };
            job.work();
            // Under the lock, so the waiting thread cannot miss it.
            let _slot = self.job.lock().unwrap_or_else(|e| e.into_inner());
            self.finished.notify_all();
        }
    }
}

/// The workers, started on first use (None where threads cannot start).
fn pool() -> Option<&'static Pool> {
    static POOL: OnceLock<Option<&'static Pool>> = OnceLock::new();
    *POOL.get_or_init(|| {
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            job: Mutex::new((0, None)),
            wake: Condvar::new(),
            finished: Condvar::new(),
            running: Mutex::new(()),
        }));
        let started = (0..WORKERS)
            .filter(|i| {
                std::thread::Builder::new()
                    .name(format!("victory refresh {i}"))
                    .spawn(move || pool.worker())
                    .is_ok()
            })
            .count();
        (started > 0).then_some(pool)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Queued copies are made by `run`, one big copy cut into pieces too,
    /// each byte where it belongs.
    #[test]
    fn queued_copies_are_made_together() {
        let len = 8 << 20;
        let src: Vec<u8> = (0..len).map(|i| (i * 7 + i / 4096) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr(), dst.as_mut_ptr());
        let ((), pieces) = queued(|| {
            for k in 0..64usize {
                let at = (k * 37 % 128) * (64 << 10);
                // SAFETY: inside both buffers.
                copy(unsafe { d.add(at) }, unsafe { s.add(at) }, 64 << 10);
            }
        });
        assert_eq!(pieces.len(), 64);
        assert!(dst.iter().all(|&b| b == 0));
        run(pieces);
        let mut want = vec![0u8; len];
        for k in 0..64usize {
            let at = (k * 37 % 128) * (64 << 10);
            want[at..at + (64 << 10)].copy_from_slice(&src[at..at + (64 << 10)]);
        }
        assert!(dst == want);
    }
}
