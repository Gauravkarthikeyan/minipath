# Design notes

This doc explains how minipath maps onto the Pathways design, what it leaves
out, and the choices that make it behave like an accelerator runtime even
though every "device" is a CPU process.

## The pieces, and where they live

| Pathways concept | minipath | File |
|---|---|---|
| Client (one Python process, off the hardware) | `Cluster` | `minipath/controller.py` |
| Virtual devices, resource manager | `Slice`, `ResourceManager` | `minipath/resources.py` |
| Central gang scheduler | `Cluster._dispatch` under `_sched_lock` | `minipath/controller.py` |
| Compiled SPMD block (a node in the dataflow graph) | `Program`, jitted on each worker | `minipath/controller.py`, `minipath/worker.py` |
| Futures, parallel asynchronous dispatch | `Buffer` / `Sharded`. `run()` returns before compute starts | `minipath/controller.py` |
| Worker per accelerator host | `Worker`, one OS process | `minipath/worker.py` |
| Device stream | the worker's single in-order executor thread | `minipath/worker.py` |
| Collectives over ICI | ring all-reduce over Unix sockets, worker to worker | `minipath/worker.py` |
| Cross-island (DCN) transfers | `transfer()` / `copy()`, worker to worker | both |
| Resilient / elastic training | epochs + abort, `begin_recovery`, `remap`, `shrink` | both |

## Rules that make the simulation honest

**One in-order stream per worker.** Every op a worker receives goes into a
FIFO and runs on one thread. An op whose input buffer hasn't arrived yet
blocks the stream, exactly as a kernel waiting on an event blocks a CUDA
stream. If the simulation let ops run out of order, the gang-scheduling
deadlock (E2) would never show up, and the pipeline bubble (E3) would be
hidden.

**The controller is not on the data path.** Tensors move worker to worker
(`send` ops push straight to the peer's socket). The controller only sees
bytes when you call `fetch`. In E4, restoring 10 MB of optimizer state onto
a spare never passes through the controller.

**Collectives are ops in the stream, not side channels.** `allreduce` is
queued like any other op and blocks the stream until every rank has
contributed. That is what makes ordering matter.

**One XLA thread per worker.** `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
intra_op_parallelism_threads=1`, so N workers on N cores behave like N
independent devices instead of fighting over a thread pool.

**Unix domain sockets, not localhost TCP.** The first version used TCP. A
synchronous request/response loop then hit the classic Nagle + delayed-ACK
stall, with 23 steps/s where it should have been about 2,000. Unix sockets
have no Nagle, so they remove the artifact. A real deployment would use TCP
with `TCP_NODELAY` and (on TPU) ICI for collectives.

## Futures and dispatch

`Cluster.run(prog, slice, *inputs)` does three things and returns:

1. Allocates output buffer ids (one per rank per output).
2. Builds one `run` message per rank, plus a `compile` message the first
   time this program goes to a given worker.
3. Hands the messages to the scheduler, which writes them to the workers'
   control sockets.

Nothing waits for the worker. Each returned `Sharded` holds the output ids,
so the next `run` can name them as inputs right away. This is Pathways'
*parallel asynchronous dispatch*. The controller can run ahead of the
hardware, so its per-op Python overhead (~50 µs per op here, see E1) overlaps with device
work instead of adding to it. E1 measures exactly that.

Buffers are freed by reference counting. When a `Buffer` object is
garbage-collected on the controller, its id goes onto a queue. The next
dispatch sends a `delete` into that worker's stream. Because the stream is
in order, the delete always runs after every op that was dispatched with
that buffer as an input.

## Gang scheduling

With `gang_scheduling=True`, `_dispatch` holds one lock while it writes an
op's messages to *all* members of its gang. So every op gets one global
position, and any two workers that share two gangs see them in the same
relative order. Without the lock, two client threads interleave freely.
E2 shows that this deadlocks in 100% of trials once each client has issued
a handful of all-reduces.

Pathways does the same thing per island with a centralized scheduler. The
paper notes it is what lets the system run "the same program" across
thousands of chips without per-program barriers.

## Failure handling

1. **Detect.** The controller has a reader thread per worker. EOF on that
   socket means the worker is gone. (A real system uses heartbeats. Here,
   detection takes ~40 ms.)
2. **Fence.** Bump the epoch and send `abort(epoch)` to every survivor
   *out of band*. The control thread handles it immediately rather than
   queueing it, so a worker stuck inside a collective waiting for the dead
   peer wakes up, fails the op, and skips every queued op from the old epoch.
   Dispatch raises `WorkerLost` until the client calls `begin_recovery()`.
3. **Re-plan.** The client decides. `rm.remap(vdev)` moves the lost
   virtual device onto a hot spare. If there is no spare, `rm.shrink(slice)`
   drops it (elastic). The training loop's `Slice` object never changes
   identity in the remap case.
4. **Restore.** Ask survivors which recent optimizer states they still
   hold (`held()`, which runs in-stream, after the abort drained). Pick the
   newest complete one, copy it to the workers that lack it, and
   re-dispatch from that step.

Data-parallel replicas are bit-identical: the ring all-reduce computes each
chunk once and then broadcasts it. Batches are a pure function of
`(step, rank)`. So a recovered run produces *exactly* the same losses as
a run with no failure. The test suite checks this
(`tests/test_workloads.py::test_recovery_is_bit_identical_to_clean_run`).

## What minipath does not do

* **No compiler.** Pathways sits on XLA with GSPMD sharding, so one
  "program" spans a whole slice of chips. Here a program is one jitted
  function per worker, and sharding is explicit in user code.
* **No multi-tenancy policy.** The resource manager hands out slices
  first-come, first-served. Pathways time-multiplexes clients on the same
  chips with priorities.
* **No real interconnect.** Everything is one machine, so the 1,200 GB/s
  vs. ~6 GB/s ICI/DCN gap that drives Pathways' placement choices does not
  exist here.
* **Detection by socket EOF.** A hung (not dead) worker would need
  heartbeats and timeouts.
* **The controller is a single point of failure**, as it is in Pathways.
  The client can be checkpointed, but minipath doesn't do it.
