# minipath

A single-controller runtime for distributed training, written from scratch
in about 1,100 lines of Python. It's my attempt to understand Google's
[Pathways](https://developers.googleblog.com/what-is-pathways-single-controller-training-for-large-models/)
by rebuilding the core ideas and measuring them on real workloads.

One Python process drives every worker. It dispatches compiled JAX programs
as futures, gang-schedules collectives, moves tensors worker-to-worker, and
when a worker dies, re-plans the job onto a spare without restarting it.

![A worker is SIGKILLed mid-training. The single controller remaps it to a spare and keeps going; SPMD restarts from its last checkpoint.](docs/figures/e4_fault_tolerance.png)

<sub>Data-parallel transformer training on Tiny Shakespeare, 4 replicas.
Left: a worker is SIGKILLed after step 130, with both runs aligned on the
kill. The single controller moves the lost virtual device to a hot spare
and restores state from a survivor's memory. SPMD restarts from its last
checkpoint. Right: the recovered run's losses match a failure-free run bit
for bit, until a second kill with no spare left forces an elastic shrink
from 4 to 3 replicas.</sub>

## Results

Every number below comes from a script in [`experiments/`](experiments/) on
a 4-core CPU machine. Raw output is in [`results/`](results/).

| Question | Result |
|---|---|
| **What does a single controller cost?** (E1) | On 4 workers, async dispatch runs trivial all-reduce chains at **474/s vs 588/s** for SPMD (on 2 workers: 1,756 vs 1,699). A controller that waits on every op (RPC style) gets 408/s. With real compute per op, async stays within ~10% of SPMD at every size tested. At 2 GFLOPs per op, it is 27.4 ms vs 27.5 ms. |
| **Why does it need a gang scheduler?** (E2) | Two clients sharing 4 workers deadlocked in **20 of 20 trials** without one, and **0 of 20** with one. |
| **Can one controller drive an MPMD pipeline?** (E3) | A 4-stage GPipe transformer matches single-device training to **3 × 10⁻⁷** in loss. The measured bubble matches the (P−1)/(M+P−1) bound: **43% measured vs 43% theory** at M=4. |
| **What does a failure cost?** (E4) | Single controller with a hot spare: new progress **2.8 s** after the kill, **2 steps** redone, and the losses are **bit-identical** to a run with no failure. SPMD with checkpoint/restart: **24.2 s** and **31 steps** redone. |
| **Does overlapping RL sampling and training help?** (E5) | Moving one line of the controller's Python makes rollouts for step k+1 overlap the update for step k: **1.48× faster** per iteration (62.7 → 42.5 ms), and CartPole is solved in about the same number of iterations (32–35 vs 30–34). |

## Why this exists

Most large-model training today is **SPMD, multi-controller**. Every host
runs the same Python program in lockstep, and the hosts coordinate only
through collectives (PyTorch DDP/FSDP under `torchrun`, multi-controller JAX).
That design is simple and fast, but three things get awkward:

1. **Failures.** One lost host stalls every peer inside an all-reduce. The
   standard fix is to kill the whole job and restart from the last
   checkpoint. The Gemini 2.5 report says training was interrupted
   ["multiple times per hour"](https://arxiv.org/abs/2507.06261).
2. **Different programs on different devices.** Pipeline stages run
   different code. RL post-training runs a sampler, a reward model and a
   learner. Under SPMD you encode all of that as rank-dependent `if`
   branches in one program.
3. **One view of the whole job.** No single process knows what is queued
   where.

Pathways flips the model. A single client builds the computation as a graph
of compiled SPMD blocks (nodes) joined by tensors (edges) placed on
*virtual* devices. A resource manager maps virtual devices to physical
chips, a central scheduler gives every op one global order, and dispatch is
asynchronous over futures, so the client stays off the critical path.

minipath implements each of those pieces small enough to read in an
afternoon, then checks the claims with experiments.

## How it works

```mermaid
flowchart LR
    subgraph C["Controller: 1 Python process"]
        U["training loop<br/>(plain Python)"] --> D["run / allreduce / transfer<br/>return futures immediately"]
        D --> S["gang scheduler<br/>one global op order"]
        R["resource manager<br/>virtual → physical devices<br/>spares, remap, shrink"] -.-> D
    end
    S -- "op messages" --> W0 & W1 & W2 & W3
    subgraph W["Workers: 1 process each"]
        W0["worker 0<br/>in-order stream<br/>buffer store"]
        W1["worker 1"]
        W2["worker 2"]
        W3["worker 3"]
    end
    W0 <-. "tensors, ring all-reduce<br/>(never through the controller)" .-> W1
    W1 <-.-> W2
    W2 <-.-> W3
```

* **Programs** are plain JAX functions. `cluster.compile(fn)` pickles one,
  and each worker `jax.jit`s it the first time it runs there.
* **`run`, `allreduce` and `transfer` return futures** (`Sharded`, one
  buffer id per rank) before any work happens. You chain futures the way you
  would chain arrays.
* **Every worker has one in-order stream**, like a CUDA stream or a TPU
  core. An op whose input hasn't arrived blocks the stream. That is what
  makes gang scheduling necessary, and it is how pipelines emerge without
  hand-written synchronization.
* **The controller is never on the data path.** Transfers and collectives
  go worker to worker. Transfers run beside the compute stream, the way a
  DMA engine does.
* **Failures fence by epoch.** On a worker loss, the controller bumps the
  epoch and sends an out-of-band abort, which unblocks survivors stuck in a
  collective. It then refuses dispatch until the client re-plans.

[`docs/DESIGN.md`](docs/DESIGN.md) maps each Pathways concept to the code
and lists what is deliberately left out.

### The API in one example

From [`examples/quickstart.py`](examples/quickstart.py), which trains the
char transformer data-parallel on 4 workers (`python examples/quickstart.py`,
about 2 minutes on 4 CPU cores):

```python
with Cluster(4) as cluster:
    dp = cluster.allocate(4)                                  # 4 virtual devices
    batch = cluster.compile(T.make_batcher(cfg), n_out=2)     # plain JAX functions
    lg = cluster.compile(progs["loss_and_grad"], n_out=2)
    adam = cluster.compile(progs["adam"], n_out=4)

    data = cluster.put(tokens, dp)            # replicated once; batches are sampled on-device
    p, m, v, t = (cluster.put(x, dp) for x in (params, zeros, zeros, np.float32(0)))

    for step in range(200):
        ranks = [(np.int32(step), np.int32(r)) for r in range(4)]
        x, y = cluster.run(batch, dp, data, consts=ranks)
        loss, g = cluster.run(lg, dp, p, x, y)
        g = cluster.allreduce(g)              # ring all-reduce, gang-scheduled
        p, m, v, t = cluster.run(adam, dp, p, m, v, g, t)
        if step % 25 == 0:
            print(f"step {step:3d}  loss {float(loss.fetch(0)):.3f}")   # the only blocking call
```

Every `run` / `allreduce` returns futures right away, so the loop runs
ahead of the workers and the only synchronization is `fetch`.

## Experiments

Each script takes `--quick` for a smoke-sized run. `make experiments` runs
them all (about 50 minutes, one at a time, since they are timing
benchmarks).

### E1. The cost of a single controller

[`experiments/e1_dispatch.py`](experiments/e1_dispatch.py). This is a
scaled-down version of the Pathways paper's dispatch micro-benchmark
(Barham et al. 2022, §5.1). The workload is a chain of trivial
gang-scheduled computations, each a program plus an all-reduce, and each one
consumes the previous one's output. It runs three ways: SPMD (every worker
runs the loop itself), a single controller that waits on each op, and a
single controller that dispatches ahead on futures.

![E1](docs/figures/e1_dispatch.png)

With trivial compute, the gap is pure overhead. On one worker, SPMD runs
50,509 computations/s because it sends no messages at all. The controller
pays about 120 µs of Python and IPC per computation (two ops). Once there is
a real collective, the all-reduce's own latency dominates, and async
dispatch lands at **103% of SPMD on 2 workers and 81% on 4**. A controller
that waits on every op is 1.2–5.5× slower than the async one, because every
op costs a full round trip.

The right panel adds real compute. Async tracks SPMD at every size: the
gap is 0–13%, and at n = 1024 (2 GFLOPs) it is 27.4 ms vs 27.5 ms. The
synchronous controller pays a round trip on every computation. That is
+50–65% on small ops, shrinking to about +9% at 2 GFLOPs.
This is the Pathways argument in one plot: dispatch overhead is a latency
cost, and futures hide it as long as there is work to overlap with.

### E2. Why a central gang scheduler

[`experiments/e2_gang.py`](experiments/e2_gang.py). Two clients share 4
workers and each issues a chain of all-reduces. Each worker runs ops in
arrival order. If worker 0 sees A's all-reduce first and worker 1 sees B's
first, each waits in a collective the other will never join. Deadlocks are
detected by a 2 s stall timeout, so they are measured, not assumed.

![E2](docs/figures/e2_gang.png)

Without a scheduler, all 20 trials deadlocked in *both* variants. That
includes the one where each client always sends to workers in the same rank
order, so the only nondeterminism left is how two Python threads
interleave. With that fixed order, the deadlock probability climbs from 15%
when each client issues a single all-reduce to 85% at 16. With the gang
scheduler, 0 of 20 trials deadlocked, and both clients' results were checked
for numerical correctness every time.

### E3. An MPMD pipeline from a Python loop

[`experiments/e3_pipeline.py`](experiments/e3_pipeline.py). A 4-layer,
0.8M-parameter transformer is cut into 4 stages on 4 workers. Stage 0
embeds, stage 3 fuses the head, loss and first backward step, and every
stage runs different programs. The controller's GPipe schedule is two
nested loops over futures. Stage-to-stage activations and gradients are
point-to-point transfers. Gradient clipping needs the norm over all stages,
so it is computed with one all-reduce across the four stage workers, even
though each runs a different program.

![E3](docs/figures/e3_pipeline.png)

Training is the same math, to float32 rounding: after 150 steps both curves
read 2.2495, with a max per-step difference of 3 × 10⁻⁷. The timeline is a
real trace from the workers. Forward ops ripple down the stages, the last
stage fuses loss and backward, and gradients ripple back up. Nothing in the
controller code says "wait for stage 2". The schedule falls out of futures
plus in-order streams.

| micro-batches M | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| bubble, theory (P−1)/(M+P−1) | 75% | 60% | 43% | 27% | 16% |
| bubble, measured | 76% | 61% | 43% | 34% | 32% |
| speedup vs. 1 worker (232 ms/step) | 1.57× | 2.02× | 1.96× | 1.77× | 1.52× |

Up to M = 4 the measured bubble is the textbook number. Past that, the
micro-batches get tiny (2 sequences at M = 16), and fixed per-op costs
(dispatch, a transfer, a jitted call) stop being negligible. The best
end-to-end speedup is ~2× on 4 workers. Backward costs ~3× forward here
(it recomputes the forward), and the stages aren't perfectly balanced.

### E4. Killing a worker mid-training

[`experiments/e4_fault_tolerance.py`](experiments/e4_fault_tolerance.py).
Data-parallel training of the same transformer on Tiny Shakespeare: 4
replicas, global batch 128 × 64 characters, 400 steps. A worker is
`SIGKILL`ed after step 130.

* **SPMD baseline.** This is the usual DDP shape: 4 processes run the same
  loop, and rank 0 checkpoints every 50 steps. When a rank dies its peers
  stall in the all-reduce, so the launcher tears down the job, starts 4
  fresh processes and resumes from the last checkpoint. Detection is
  instant here, which is generous to SPMD.
* **Single controller.** It sees the failure, aborts in-flight work, and
  remaps the lost virtual device onto a hot spare. It then asks survivors
  which recent optimizer states they still hold in memory, copies the newest
  one to the spare worker-to-worker, and re-dispatches from there. A second
  kill after step 280 finds no spare, so the job shrinks to 3 replicas and
  keeps going.

| | single controller | SPMD + checkpoints |
|---|---|---|
| failure detected | 48 ms | 31 ms |
| re-plan (remap + choose restore point) | 8 ms | n/a |
| time until a *new* step completes | **2.8 s** | **24.2 s** |
| steps executed twice (whole run) | **2** | **31** |
| end-to-end vs. its own clean run | see note | +40 s (+18%) |
| second kill, no spares left | shrinks to 3 replicas, new steps after 0.86 s | n/a |

The single controller loses almost nothing. It resumes from the newest
optimizer state still in a survivor's memory, which is two steps behind
because the controller runs two steps ahead. The 2.8 s is the
spare compiling its programs for the first time, plus re-running those two
steps. SPMD's relaunch was fast here: 0.9 s for 4 fresh processes on a
machine with warm caches. Its real cost is the 31 steps since the step-100
checkpoint. With a checkpoint every N
steps, the expected redo is N/2.

Recovery is exact. Batches depend only on (step, rank), and the ring
all-reduce gives every rank bit-identical results. So the recovered run's
losses equal the clean run's **bit for bit** for all 280 steps before the
elastic shrink. After the shrink, the global batch is 96 instead of 128 and
the curves diverge slightly (held-out loss 1.762 vs 1.747).

Note: the single-controller run with failures finished *faster* than its
clean run (224 s vs 235 s). That's because the last 120 steps used 3
replicas and a smaller global batch, so the end-to-end numbers aren't
comparable. The clean runs show the price of a controller here: 235 s vs
222 s for SPMD (6%), with 4 workers and the controller sharing 4 cores.

A sample from the clean run's final model (0.8M params, 400 steps, 4 minutes on CPU):

```
ROMEO:
Where, welcome is dozen'd the hearth only
That Lorden Which the most make but the grom two thee;
And thy march, forbusir'd from Police,
The pence being the king, but his charge,
```

### E5. RL with overlapping sampler and learner

[`experiments/e5_rl_overlap.py`](experiments/e5_rl_overlap.py). This is the
RL post-training shape from the blog: a sampler program (2 workers
running 512 CartPole environments, written in JAX) and a learner program
(2 data-parallel workers, REINFORCE with a per-timestep baseline), with
weights flowing back after each update. The two schedules differ in where
one line sits. *sync* samples with θ<sub>k</sub> and then trains. *async*
dispatches the rollout for step k+1 before the update for step k, so the
two overlap and the policy lags by one step.

![E5](docs/figures/e5_rl_overlap.png)

| | sync | async (lag 1) |
|---|---|---|
| ms / iteration (mean of 3 seeds) | 62.7 | **42.5** |
| env steps / s | 1.63 M | **2.41 M** |
| sampler / learner busy | 37% / 42% | 69% / 83% |
| iterations to mean return ≥ 195 | 30, 33, 34 | 32, 34, 35 |
| wall-clock to mean return ≥ 195 | 1.72–1.93 s | **1.25–1.27 s** |

Overlap buys 1.48× throughput. The one-step policy lag costs about one
iteration of sample efficiency. A synchronous schedule leaves each slice
idle while the other works, and the async one fills those gaps. The first
version of this experiment showed almost no gain, because of a runtime bug
(see "What I learned").

## What I learned building it

* **The single controller costs latency, not throughput.** Its per-op
  Python cost (~50 µs here) is real. A synchronous controller pays it on
  every step, plus a round trip. An asynchronous one pays it in parallel with
  device work, so it disappears once ops are bigger than the dispatch cost.
  Futures are the whole trick.
* **In-order device streams are what make ordering a correctness issue.**
  The gang scheduler isn't an optimization. Without it, the shared-device
  case deadlocks. I didn't believe how fast until E2.
* **Transfers must not sit in the compute stream.** My first version queued
  `send` ops in the source worker's stream. In E5, the batch for step k then
  waited behind the rollout for step k+1, which was itself waiting for new
  weights, and the "async" schedule ran at sync speed. Moving transfers
  beside the stream, the way DMA engines sit beside compute, took the async
  iteration from 61 ms to 47 ms in a smoke run, against 66 ms for sync.
* **Determinism turns recovery into a correctness test.** Batches are a
  function of (step, rank) and the ring all-reduce is bit-identical on every
  rank. So "did recovery work?" has an exact answer: the losses match a
  clean run to the last bit.
* **CI finds the races your machine hides.** The first CI run failed a
  test that passed every time locally. Two client threads could each decide
  a program still needed to be shipped to a worker, and the second one's
  `run` could then reach the worker before the first one's `compile`. On 4
  cores it never happened. On the 2-core CI runner it did. The fix sends the
  compile under the same per-connection lock as the first `run`.
* **Localhost TCP lies to you.** The first sync benchmark ran at 23 steps/s
  because of Nagle's algorithm plus delayed ACKs. Unix sockets fixed it
  (~2,000 steps/s).

## Limitations

This is a teaching-scale reconstruction, not a production system. Read the
numbers with these in mind:

* **Everything runs on one 4-core machine.** "Workers" are processes and
  "interconnect" is Unix sockets at ~330 MB/s. There is no 100× gap between
  ICI and DCN bandwidth as on TPU pods, so none of the placement problems
  that gap creates show up here.
* **The controller shares those 4 cores with the workers.** On Pathways it
  runs on its own CPU host. E1 part B uses 3 workers for this reason.
* **There is no compiler-level sharding.** A Pathways node is a GSPMD
  program spanning many chips. Here it is one `jax.jit` per worker, and
  sharding is explicit in user code.
* **The SPMD baseline in E4 gets instant failure detection and restarts on
  the same machine**, which makes it look better than a real job (NCCL
  timeouts, rescheduling, image pulls). On the other side, torchrun-style
  elastic agents can also use hot spares. The gap here comes mostly from
  re-doing the steps since the last checkpoint.
* **The controller is a single point of failure**, as it is in Pathways.
* **E5's utilization numbers are inflated** when all four workers and the
  controller compete for four cores, because ops simply take longer.

## Where this sits among real systems

| System | Model | Notes |
|---|---|---|
| PyTorch DDP / FSDP, multi-controller JAX | multi-controller SPMD | Every process runs the same program. Failures mean restart from a checkpoint (torchrun elastic can re-rendezvous). |
| [Pathways](https://arxiv.org/abs/2203.12533) (Google) | single controller over compiled SPMD blocks, gang-scheduled per island | Reports ~100% accelerator utilization for SPMD on 2048 TPUs, and SPMD-level throughput for 16-stage pipelines. Generally available on Google Cloud as of 2026. |
| [Ray](https://www.ray.io/) | single driver over tasks and actors | The Pathways blog puts it well: Ray's job is orchestration, Pathways' job is execution. The training *inside* a Ray worker group is still SPMD. |
| [PyTorch Monarch](https://pytorch.org/blog/introducing-pytorch-monarch/) (Meta) | single controller over process/actor meshes, Rust backend | The closest open-source analogue in the PyTorch world. It fails fast by default, like an uncaught exception in a local script. |
| [verl / HybridFlow](https://arxiv.org/abs/2409.19256) (ByteDance) | hybrid: single controller *between* models, multi-controller *within* each | Built for RLHF. Reports 1.53–20.57× throughput over prior RLHF systems. E5 is a tiny version of this split. |
| **minipath** | single controller over per-worker jitted programs | ~1,100 lines, for reading and measuring. |

## Reproduce

```bash
git clone https://github.com/Gauravkarthikeyan/rl-poster && cd rl-poster
pip install -e ".[dev]"          # jax, numpy, cloudpickle, matplotlib, pytest
pytest -q                        # 20 tests, ~40 s
python examples/quickstart.py    # train on Tiny Shakespeare with 4 workers, ~2 min
make quick                       # every experiment at smoke size, ~8 min
make experiments                 # full runs + figures, ~50 min
```

Tiny Shakespeare (1.1 MB) downloads on first use and is checked against a
SHA-256.

## Layout

```
minipath/
  controller.py      Cluster: dispatch, futures, gang scheduler, failure fencing
  worker.py          worker process: in-order stream, buffer store, ring all-reduce
  resources.py       virtual devices, slices, spares, remap, elastic shrink
  workloads/
    transformer.py   char-level transformer in plain JAX, splittable into stages
    shakespeare.py   dataset download + checksum
    cartpole.py      CartPole dynamics + REINFORCE in JAX
examples/            quickstart.py: the API on a real workload
experiments/         E1–E5 and plots.py (every figure is generated from results/*.json)
tests/               runtime semantics + workload correctness (pipeline == 1 device,
                     recovery == clean run, physics == closed form)
docs/DESIGN.md       how each Pathways idea maps to the code, and what's left out
```

## References

* Barham et al., [*Pathways: Asynchronous Distributed Dataflow for ML*](https://arxiv.org/abs/2203.12533), MLSys 2022.
* Google Developers Blog, [*What is Pathways? Single-controller training for large models*](https://developers.googleblog.com/what-is-pathways-single-controller-training-for-large-models/), 2026.
* Huang et al., [*GPipe*](https://arxiv.org/abs/1811.06965), NeurIPS 2019.
* Sheng et al., [*HybridFlow: A Flexible and Efficient RLHF Framework*](https://arxiv.org/abs/2409.19256), EuroSys 2025.
* [Introducing PyTorch Monarch](https://pytorch.org/blog/introducing-pytorch-monarch/), PyTorch blog, 2025.
* Gemini Team, [*Gemini 2.5*](https://arxiv.org/abs/2507.06261) technical report, 2025.
* Barto, Sutton & Anderson, *Neuronlike adaptive elements that can solve difficult learning control problems*, 1983 (CartPole).
* Karpathy, [char-rnn](https://github.com/karpathy/char-rnn) (Tiny Shakespeare).

## License

MIT
