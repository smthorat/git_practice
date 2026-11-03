# LSF on Pegasus (University of Miami) — Practical Guide

**Written for:** account `sxt1558` (group `hphr`, project `hphr`)
**Cluster inspected:** `pegasus`, IBM Spectrum LSF **10.1.0.12** (Suite for HPC 10.2.0.12)
**Inspection date:** 2026-11-03, from login node `login1` (RHEL 8.8)
**Scope:** read-only inspection. No jobs were submitted or cancelled; nothing was modified.

### How to read this guide

| Mark | Meaning |
|---|---|
| ✅ **VERIFIED** | Read directly from this cluster (`bqueues -l`, `lshosts`, `lsf.conf`, `lsb.queues`, `lsb.params`, `df`, `module avail`). |
| 📌 **SNAPSHOT** | A live number from 2026-11-03. It changes constantly — re-run the command. |
| ⚠️ **UNVERIFIED** | Could not be confirmed with read-only commands. Ask HPC support: <https://hpc.idsc.miami.edu> |
| 📖 **GENERAL LSF** | Standard LSF behaviour, not a Pegasus-specific policy. |

---

## ⚡ Read this first — three things that will bite you

### 1. `-P hphr` is mandatory. Every job. No exceptions.

✅ **VERIFIED** — `lsf.conf` sets `LSB_ESUB_METHOD="project"`, and the esub script
(`$LSF_SERVERDIR/etc/esub.project`) **rejects any job submitted without a valid project**.
The automatic fallback to a default project was deliberately disabled.

```bash
bsub -P hphr -q normal ...     # ✅ works
bsub -q normal ...             # ❌ rejected: "Error: no project specified."
```

✅ **VERIFIED** — your account is in exactly one project. From `/opt/ibm/lsfsuite/lsf/conf/lsf.projects`:

```
hphr nxs1664 err92 sxt1558
```

Check your own projects at any time:

```bash
egrep "\bsxt1558\b" /opt/ibm/lsfsuite/lsf/conf/lsf.projects | awk '{print $1}'
```

### 2. You do **not** have access to the `sccc_*_premium` queues

You asked about `sccc_premium`, `sccc_bigmem_premium` and `sccc_gpu_premium`.
✅ **VERIFIED** — all three exist, but your account is not authorised for any of them.
`bqueues -u sxt1558` does not list them, and the `USERS` lines in `lsb.queues` confirm it:

| Queue | `USERS` line in `lsb.queues` | You (`sxt1558`, group `hphr`)? |
|---|---|---|
| `sccc_premium` | `hpc sccc_hpc sccc_iavarone sccc_ceccarelli h.gallegos krd` | ❌ **No access** |
| `sccc_bigmem_premium` | `hpc sccc_hpc sccc_iavarone sccc_ceccarelli h.gallegos` | ❌ **No access** |
| `sccc_gpu_premium` | `hpc sccc_hpc sccc_iavarone h.gallegos sxw1590 dak216` | ❌ **No access** |
| `sccc` | `... hphr ...` | ✅ **Yes** |
| `sccc_bigmem` | `... hphr ...` | ✅ **Yes** |
| `sccc_gpu` | `... hphr ...` | ✅ **Yes** |

**The practical consequence:** you use the **non-premium** `sccc`, `sccc_bigmem` and `sccc_gpu`
queues. These are **preemptable by the premium queues** (see [§2.5](#25-preemption--important-for-sccc-queues)).
This guide documents the premium queues' hardware for reference, but all examples use queues you can actually submit to.

### 3. Memory is in **MB**, and LSF will **not** kill you for exceeding it

✅ **VERIFIED** — `lsf.conf` contains:

```
LSF_UNIT_FOR_LIMITS=MB          # -M and rusage[mem=] are MEGABYTES
LSF_ENFORCE_MEMLIMIT=N          # (comment in file: "20250516, Pedro/John, Disabled MEMLIMIT")
LSB_RESOURCE_ENFORCE="cpu gpu"  # cgroups enforce CPU and GPU — NOT memory
```

So `-M 16000` means **16000 MB ≈ 15.6 GB**. Confirmed on a real job of yours: `-M 24000` was
reported by `bjobs -l` as `MEMLIMIT 23.4 G`, and `-M 48000` as `46.8 G`.

Because `LSF_ENFORCE_MEMLIMIT=N`, LSF does **not** kill a job that exceeds `-M`. What kills you
instead is the **node running out of physical RAM** — the Linux OOM killer, which is messier and
can take out other people's jobs. Request memory honestly.

---

## 1. Understanding LSF

### 1.1 What is LSF and why do we need it?

Pegasus has ✅ **302 hosts** (`lsclusters`) shared by hundreds of users. If everyone just ran
programs wherever they liked, people would collide: two 100 GB jobs on a 64 GB node would both die.

LSF (Load Sharing Facility) is the **batch scheduler**. You describe what you need
(cores, RAM, time), LSF finds a node that has it, runs your job there, and keeps everyone else off
those resources until you're done. You get a guaranteed allocation; the cluster gets high utilisation.

### 1.2 How LSF manages resources

LSF thinks in **slots** (≈ one CPU core).

- ✅ **VERIFIED** — `lsb.params` sets `PARALLEL_SCHED_BY_SLOT=y`, so `-n 6` means **6 slots/cores**, not 6 nodes.
- ✅ **VERIFIED** — `LSF_PROCESS_TRACKING=Y`, `LSF_LINUX_CGROUP_ACCT=Y`, `LSB_RESOURCE_ENFORCE="cpu gpu"`.
  Your job is placed in a **Linux cgroup** pinned to the cores you requested. Spawning 64 threads
  in a `-n 6` job does not give you 64 cores — the threads time-share your 6, and everything runs slower.
- Memory is **reserved** by the scheduler (`-R "rusage[mem=...]"`) so it won't double-book a node,
  but **not enforced** at runtime (see the warning above).
- ✅ **VERIFIED** — all general queues set `RES_REQ = affinity[core(1)] ...`, i.e. one physical core per slot.

### 1.3 Login node vs compute node

| | Login node (`login1`) | Compute nodes (`n*`, `usr*`, `gpu*`, `dtn1`) |
|---|---|---|
| How you get there | `ssh` | only via LSF (`bsub`) |
| Count | ✅ 1 (+ `dtn1` for transfers) | ✅ 302 total hosts |
| Hardware | ✅ 16 cores, 188 GB RAM | varies — see [§2](#2-queues-and-resources) |
| Shared with | **everyone logged in, right now** | only jobs LSF placed there |
| Use it for | editing files, `bsub`, `bjobs`, small `ls`/`grep`, `git` | all real computation |
| Do **not** use it for | STAR, bwa, samtools sort, R/Python analysis, `gzip` of big FASTQs | — |

A `STAR` run on `login1` takes 16 cores and tens of GB away from every other person trying to
type a command. Admins kill such processes, and repeat offenders lose access.

### 1.4 Job lifecycle

```
bsub  ──▶  PEND  ──▶  RUN  ──▶  DONE  (exit 0)
                        │
                        └────▶  EXIT  (non-zero exit, OOM, killed, runlimit hit)
```

| State | Meaning |
|---|---|
| `PEND` | Queued. Waiting for resources, a dependency, or fairshare priority. |
| `RUN` | Executing on a compute node. |
| `DONE` | Finished with exit code 0. |
| `EXIT` | Finished with a non-zero exit code, or was killed. |
| `SSUSP` | Suspended by the system — on Pegasus usually **preemption** by a premium queue. |
| `USUSP` | Suspended by you or an admin (`bstop`). |

✅ **VERIFIED** — scheduling uses **FAIRSHARE** on every queue you can use
(`bqueues -l normal` → `SCHEDULING POLICIES: FAIRSHARE`), with
`CPU_TIME_FACTOR=0.7`, `RUN_TIME_FACTOR=0.7`, `RUN_JOB_FACTOR=3.0` (`bparams -a`).
**Translation:** the more you've run recently, the lower your priority. Having 400 jobs queued does
not get you served faster — it pushes your own priority down. Your current dynamic priority:

```bash
bqueues -l normal | grep -A50 SHARE_INFO_FOR | grep sxt1558
```

> ⚠️ **Important for checking finished jobs:** `bparams -a` shows `CLEAN_PERIOD = 3600`.
> A finished job vanishes from `bjobs` after **1 hour**. After that use `bhist -l <jobid>`.

### 1.5 Interactive vs batch jobs

| | Batch (normal case) | Interactive |
|---|---|---|
| Command | `bsub ... script.sh` | `bsub -Is ... /bin/bash` |
| Terminal | returns immediately | your shell moves onto a compute node |
| Output | goes to `-o` / `-e` files | goes to your screen |
| Survives disconnect | ✅ yes | ❌ no — dies with your SSH session |
| Use for | real analysis, long runs, pipelines | debugging, testing a command, compiling, exploring data |

⚠️ **Note on the `interactive` queue:** ✅ **VERIFIED** it is `Closed:Inact` for new jobs
(`bqueues`) and `PROCLIMIT = 1`. Use `-q normal` for interactive sessions instead —
`lsb.queues` shows `normal` has `INTERACTIVE = YES`.

```bash
# Interactive shell: 4 cores, 16 GB, 2 hours, on the normal queue
bsub -P hphr -q normal -n 4 -R "span[hosts=1] rusage[mem=16000]" -W 2:00 -Is /bin/bash
```

---

## 2. Queues and resources

### 2.1 Queues your account can submit to

✅ **VERIFIED** — `bqueues -u sxt1558`:

| Queue | Priority | Status 📌 | Max cores (`PROCLIMIT`) | Queue `RUNLIMIT` | Notes |
|---|---|---|---|---|---|
| **`normal`** | 30 | Open:Active | **16** | **none** | Default queue. Single node. |
| **`bigmem`** | 30 | Open:Active | none | **14400 min = 240 h (10 d)** | `INTERACTIVE = NO` |
| **`sccc`** | 10 | Open:Active | none | none | 64-core / 503 GB nodes. **Preemptable.** |
| **`sccc_bigmem`** | 10 | Open:Active | none | none | 1 node, **3.9 TB RAM**. **Preemptable.** |
| **`sccc_gpu`** | 10 | Open:Active | none | none | 1 node, 2 GPUs, 3.9 TB. **Preemptable.** |
| **`parallel`** | 30 | Open:Active | **512** | none | Multi-node MPI, `span[ptile=16]`. `INTERACTIVE = NO` |
| **`transfers`** | 30 | Open:Active | **4** | none | Data-transfer node `dtn1` only. |
| **`gpu_h100`** | 30 | Open:Active | none | none | H100 nodes `gpu1`, `gpu2`. |
| `gpu_arg` | 30 | Open:Active | none | none | — |
| `gpu_titan` | 30 | Open:Active | none | none | — |
| `jupyter` | 30 | Open:Active | none | none | For JupyterHub sessions. |
| `interactive` | 30 | **Closed:Active** | 1 | none | Closed to new jobs. |
| `general`, `sccc_dev`, `gpu_2080` | — | **Closed:Inact** | — | — | Not usable. |

✅ **VERIFIED** — `lsb.params` has **no `DEFAULT_RUNLIMIT`**. Queues other than `bigmem` have
**no run limit at all**: a job without `-W` can run indefinitely. Always set `-W` anyway —
it lets the scheduler backfill you into gaps, so you often start sooner.

✅ **VERIFIED** — `ABS_RUNLIMIT=Y`. `-W` is **wall-clock** time, not CPU-speed-normalised time.
`-W 6:00` = 6 real hours.

### 2.2 Hardware per queue

✅ **VERIFIED** via `lshosts` and `bmgroup -w`. `maxmem` is the node's **total** RAM —
the OS needs some, so usable is a little less.

| Queue | Hosts | Cores/node | RAM/node | Largest single job that fits |
|---|---|---|---|---|
| `normal` | ✅ 82 nodes (`n024`…`n335`) | **16** | ✅ **31.3 GB or 62.8 GB** (mixed) | 16 cores, ~60 GB (`PROCLIMIT`=16) |
| `bigmem` | ✅ 19 nodes (`n051`,`n128`,`n129`,`n131`,`n135`,`n136`,`n248`,`n502`–`n510`,`n517`,`n518`,`n262`) + `gpu1` | **16** | ✅ **251.8 GB** | 16 cores, ~250 GB |
| `sccc` | ✅ `n523 n524 n525 n531 n532 n534` | **64** | ✅ **502.9 GB** | 64 cores, ~500 GB |
| `sccc_bigmem` | ✅ `n521` (one node) | **64** | ✅ **3.9 TB** | 64 cores, ~3.9 TB |
| `sccc_gpu` | ✅ `n522` (one node) | **64** | ✅ **3.9 TB** + **2 GPUs** | 64 cores, 2 GPUs |
| `parallel` | ✅ 52 nodes | **16** | ✅ 62.8 GB (typical) | 512 cores across 32 nodes |
| `transfers` | ✅ `dtn1` | 28 | ✅ 219.1 GB | 4 cores (`PROCLIMIT`=4) |
| `gpu_h100` | ✅ `gpu1`, `gpu2` | ⚠️ not reporting | ⚠️ not reporting | ⚠️ ask support |
| *(ref.)* `sccc_premium` | `n521`–`n538` (16 nodes) | 64 | 502.9 GB | ❌ no access |
| *(ref.)* `sccc_bigmem_premium` | `n521`, `n522` | 64 | 3.9 TB | ❌ no access |

⚠️ **UNVERIFIED** — `lshosts` reports `ncpus`/`maxmem` as `-` for ~40 hosts whose LIM is down or
not reporting (`UNKNOWN_AUTO_DETECT`), including `gpu1`, `gpu2`, `n524`, `n531`, `n532`, `n534`,
`n504`, `n506`, `n509`, `n517`, `n518`, `n262`. Their specs could not be read. ✅ `lsload -gpu n522`
does confirm **2 GPUs** on `n522`; the model (the queue description says *"Nvidia A100"*)
could not be read from the CLI.

### 2.3 Maximum you can actually request

| Resource | Hard ceiling | Where it comes from |
|---|---|---|
| Cores, `normal` | **16** | ✅ `PROCLIMIT = 16` in `lsb.queues` |
| Cores, `parallel` | **512** | ✅ `PROCLIMIT = 512` |
| Cores, `transfers` | **4** | ✅ `PROCLIMIT = 4` |
| Cores, `bigmem` / `sccc*` | node core count (16 / 64) | ✅ no `PROCLIMIT`; all set `span[hosts=1]` |
| RAM | the node's physical RAM | ✅ no `MEMLIMIT` configured on any queue |
| Runtime | `bigmem` 240 h; others unlimited | ✅ `RUNLIMIT = 14400` on `bigmem` only |
| Job array size | **1000** | ✅ `MAX_JOB_ARRAY_SIZE = 1000` (`bparams -a`) |
| Jobs queued | effectively unlimited | ✅ `MAX_PEND_JOBS = 2147483647`; `blimits` shows no user limits |
| Job ID range | wraps at 999999 | ✅ `MAX_JOBID = 999999` |

✅ **VERIFIED** — `blimits` and `blimits -w` both return *"No resource usage found"*:
there are **no per-user slot or memory limits** configured for you. Fairshare, not a hard cap,
is what throttles you.

⚠️ Asking for more than a node physically has is not rejected at submit time — the job just
**pends forever** with `LSF_STRICT_RESREQ=Y` (✅ verified in `lsf.conf`), because no host can
ever satisfy it. `bjobs -p` will say so.

### 2.4 Which queue to use

| Your job needs | Queue | Why |
|---|---|---|
| ≤16 cores, ≤~30 GB, hours | **`normal`** | Most nodes, shortest wait, never preempted |
| ≤16 cores, 60–250 GB | **`bigmem`** | Only non-sccc queue with 251 GB nodes |
| Run longer than ~10 days | **`normal`** | `bigmem` caps at 240 h |
| 17–64 cores on one node | **`sccc`** | 64-core nodes; accept preemption risk |
| 250 GB – 3.9 TB RAM | **`sccc_bigmem`** | Only place with 3.9 TB (single node `n521`) |
| A GPU | **`sccc_gpu`** or `gpu_h100` | `n522` has 2 GPUs |
| >16 cores across several nodes (MPI) | **`parallel`** | `span[ptile=16]`, up to 512 cores |
| Download / upload many GB | **`transfers`** | `dtn1` has the fast external network |
| **RNA-seq (STAR, salmon, featureCounts)** | **`bigmem`** | STAR human index needs ~32–40 GB; 16 cores is plenty |
| **Metabolomics / R (DESeq2, limma, XCMS)** | **`normal`**, or `bigmem` if >30 GB | Usually single-threaded and modest |
| **Nextflow driver process** | **`normal`**, 1–2 cores | It only submits and waits |

### 2.5 Preemption — important for `sccc*` queues

✅ **VERIFIED** — from `lsb.queues`:

```
sccc_premium:         PREEMPTION = PREEMPTIVE[sccc],        PREEMPT_DELAY = 60
sccc_bigmem:          PREEMPTION = PREEMPTABLE[sccc_bigmem_premium]
sccc_bigmem_premium:  PREEMPTION = PREEMPTIVE[sccc_bigmem], PREEMPT_DELAY = 60
sccc_gpu:             PREEMPTION = PREEMPTABLE[sccc_gpu_premium]
sccc_gpu_premium:     PREEMPTION = PREEMPTIVE[sccc_gpu],    PREEMPT_DELAY = 60
```

A premium-queue user (which you are not) can **push your running `sccc*` job aside** after 60 seconds
to take its slots. 📖 **GENERAL LSF** — the default preemption action is to **suspend** the job
(`SSUSP`); it resumes when resources free up. No `TERMINATE_WHEN` is configured on these queues,
so your job should not be killed outright — but a suspended job's wall-clock keeps ticking, and
anything with a network timeout or a lease may break. ✅ `bparams -a` shows
`PREEMPTION_WAIT_TIME = 300`, `NO_PREEMPT_RUN_TIME = -1` (no grace period based on elapsed time).

**Rule of thumb:** `normal`, `bigmem` and `parallel` are never preempted. Use `sccc*` when you
genuinely need >16 cores or >250 GB, and checkpoint long runs if you can.

### 2.6 Checking current availability

```bash
# Are hosts free in a queue? (ok = accepting, closed = full/closed, unavail = down)
bhosts normal       | awk 'NR>1{print $2}' | sort | uniq -c
bhosts bigmem
bhosts sccc

# Free slots in the normal queue right now
bhosts normal | awk 'NR>1 && $2=="ok" {free+=$4-$5; tot+=$4} END{print free" free of "tot}'

# Free memory and load per node (mem column = FREE memory, not total)
lsload n061 n190
lsload bigmem

# How busy is each queue? (NJOBS / PEND / RUN)
bqueues -u sxt1558

# Per-node detail: slots, reserved memory, status reason
bhosts -l n521

# GPUs
lsload -gpu n522
```

📌 **SNAPSHOT 2026-11-03 ~11:20:**

| Queue | Hosts `ok` | `closed` | `unavail` | Free slots | Queue backlog (PEND/RUN) |
|---|---|---|---|---|---|
| `normal` | 51 | 27 | 6 | **434 of 816** | 2940 / 818 |
| `bigmem` | **0** | 12 | 6 (+1 unreach) | 0 | 262 / 96 |
| `sccc` | 2 (`n523`,`n525`) | 0 | 4 | ~126 of 128 | 10 / 2 |
| `sccc_bigmem` | 0 (`n521` = `closed_Full`) | 1 | 0 | 0 | 2 / 32 |
| `sccc_gpu` | 1 (`n522`, 16/64 used) | 0 | 0 | 48 | 0 / 0 |

Note the asymmetry: `normal` had 434 free slots while `bigmem` was completely full and `sccc` was
almost completely idle. **Always check before you pick a queue** — it changes the wait from hours
to minutes. On this snapshot, a 64-core / 400 GB job would have started immediately on `sccc`.

---

## 3. Storage and filesystem

### 3.1 What exists

✅ **VERIFIED** via `df -hT` and `ls -ld`:

| Path | Filesystem | Size | Used 📌 | Avail 📌 | Persistent? | Shared with |
|---|---|---|---|---|---|---|
| **`$HOME`** = `/nethome/sxt1558` | `ems2-gpfs2` (GPFS) | **250 GB** | 17 GB (7%) | 234 GB | ✅ Yes, backed up ⚠️ | you only |
| **`/scratch/hphr`** | `ems2-gpfs2` | **2.0 TB** | 1.2 TB (59%) | 858 GB | ⚠️ see §3.3 | group `hphr` |
| **`/scratch/projects/sxt1558`** | `ems2-gpfs2` | shares the 801 TB `/scratch` pool | 74 GB (yours) | 165 TB pool-wide | ⚠️ see §3.3 | you (world-readable parent) |
| `/scratch` (pool) | `ems2-gpfs2` | 801 TB | 636 TB (80%) | 165 TB | ⚠️ | everyone |
| `/projectnb` | `pix1-mmfs1` (GPFS) | 2.1 PB | 1.9 PB (**92%**) | 173 TB | ✅ | project groups |
| `/sccc` | `pix1-mmfs3` (GPFS) | 2.4 PB | 2.4 PB (**100% FULL**) | 20 TB | ✅ | SCCC groups |
| `/hihg` | `ess2-gpfs1` (GPFS) | 4.2 PB | 3.5 PB (85%) | 664 TB | ✅ | HIHG groups |
| `/tmp` | **local disk on each node** | 460 GB on `login1` | 78 GB (17%) | 382 GB | ❌ node-local, wiped | node-local |
| `$TMPDIR` | — | ✅ **not set** | — | — | — | — |

✅ **VERIFIED** — `/scratch/hphr` is `drwxrws---+ root:hphr`, i.e. **setgid**: files you create
there automatically belong to group `hphr` so your labmates can read them. Your existing layout:

```
/scratch/hphr/rna_analysis/
├── apptainer_cache/   # container images
├── raw_data/          # FASTQ
├── references/        # genome + annotation
├── results/
├── squire/            # pipeline outputs
└── work/              # Nextflow work dir (253 subdirs)
```

### 3.2 Where to put what

| Data | Put it in | Why |
|---|---|---|
| Scripts, `nextflow.config`, notebooks, small tables | **`$HOME`** (`~/squire_analysis`) | Small, versioned, the only place likely backed up |
| Conda environments | **`$HOME/.conda/envs`** | ✅ where the `miniforge3` module points `CONDA_ENVS_PATH` |
| **FASTQ, BAM, CRAM** | **`/scratch/hphr/rna_analysis/raw_data`** | Big, regenerable, fast GPFS, shared with the group |
| **Reference genomes, STAR/salmon indexes** | **`/scratch/hphr/rna_analysis/references`** | Big, built once, group-shared. Or use the `igenomes` module (§6.7) |
| Apptainer `.sif` images | **`/scratch/hphr/rna_analysis/apptainer_cache`** | 1–3 GB each; would eat your 250 GB home |
| **Nextflow `work/`** | **`/scratch/hphr/rna_analysis/work`** | Huge and throwaway. **Never** in `$HOME` |
| Final results (counts, DE tables, plots, reports) | `/scratch/hphr/.../results` **and copy the small ones to `$HOME`** | Scratch may not be backed up |
| Job logs | `$HOME/squire_analysis/logs` | Small, you want them when debugging |
| Per-task temp inside a job | **`/tmp` on the compute node** (§3.5) | Local disk is faster than GPFS and doesn't use your quota |

### 3.3 Quotas and retention — what I could **not** verify

⚠️ **UNVERIFIED — quota values.** `mmlsquota` exists (`/usr/lpp/mmfs/bin/mmlsquota`) but returns:

```
No quota enabled file system found.
mmlsquota: tslsquota failed. Error code 22.
```

GPFS quotas are not readable from your account. No other quota tool exists
(`lfs`, `gpfsquota`, `checkquota` are all absent).

**What I can say:** `df` reports `/nethome/sxt1558` as **250 GB** and `/scratch/hphr` as **2.0 TB**,
while the underlying `ems2-gpfs2` pool is 801 TB. These are **independent GPFS filesets** — their
`df` size is the administrative allocation, which in practice behaves as your limit. Treat **250 GB
home** and **2 TB group scratch** as your working budgets, but confirm the official quota with HPC
support rather than relying on `df`.

⚠️ **UNVERIFIED — scratch purge policy.** I found no retention policy, purge script, or README
anywhere under `/scratch` readable from your account. **I cannot confirm whether or when
`/scratch` files are automatically deleted.** Most HPC sites purge scratch after 30–90 days of no
access, and `/scratch/hphr` is an allocated fileset which *may* be exempt — but I will not guess.

**Until you confirm with <https://hpc.idsc.miami.edu>, assume scratch is not guaranteed:**
keep scripts and final small results in `$HOME`, and keep the recipe to regenerate anything
that lives only on scratch.

### 3.4 Commands to check space

```bash
# Space on the filesystems you use
df -h $HOME /scratch/hphr /scratch/projects/$USER

# What is using your home (biggest first)
du -sh $HOME/* 2>/dev/null | sort -rh | head -20

# What is using the group scratch
du -sh /scratch/hphr/rna_analysis/* | sort -rh

# Total for one tree
du -sh /scratch/projects/$USER        # → 74 GB on 2026-11-03

# Find the 20 biggest files in a tree
find /scratch/hphr/rna_analysis -type f -printf '%s\t%p\n' 2>/dev/null \
  | sort -rn | head -20 | awk '{printf "%.1f GB\t%s\n", $1/1073741824, $2}'

# Find files older than 60 days (purge candidates)
find /scratch/hphr/rna_analysis -type f -atime +60 -printf '%s\n' \
  | awk '{s+=$1} END{printf "%.1f GB not accessed in 60 days\n", s/1073741824}'

# Quotas (currently returns an error on this cluster — try anyway, it may be fixed)
mmlsquota
```

### 3.5 Node-local temp inside a job

`$TMPDIR` is ✅ **not set** on this cluster, so you must make your own. Local disk is much faster
than GPFS for many small random reads/writes (sorting BAMs, SQLite, Nextflow staging) and doesn't
count against your scratch allocation.

```bash
#!/bin/bash
# Make a private temp dir on the compute node's local disk, and clean it up no matter what
export TMPDIR=/tmp/$USER.$LSB_JOBID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT          # runs on success, failure, and on bkill

samtools sort -@ "$LSB_DJOB_NUMPROC" -T "$TMPDIR/sort" -o out.bam in.bam
```

⚠️ `/tmp` is **node-local**: 460 GB on `login1`, but ⚠️ **UNVERIFIED** on compute nodes. Check
free local space before relying on it, e.g. `lsload -l n128 | head`, or `df -h /tmp` inside a job.
The `trap` line matters — without it, your leftovers fill the node's disk and break other people's jobs.

✅ **VERIFIED** — LSF's own spool directory is `JOB_SPOOL_DIR=/scratch/lsf_tmp` (`bparams -a`).
That is LSF's, not yours; don't write there.

### 3.6 Best practices for large bioinformatics data

1. **Never copy a reference genome per sample.** Build the STAR/salmon index once in
   `/scratch/hphr/rna_analysis/references` and point every job at it.
2. **Keep FASTQs compressed.** `.fastq.gz` is ~4× smaller, and every modern tool reads it directly.
   Decompressing "to make it faster" wastes space and is usually slower.
3. **Delete intermediate BAMs** once counts are extracted. Unsorted + sorted + deduped BAMs for one
   RNA-seq sample is easily 15 GB; × 24 samples = 360 GB.
4. **Use CRAM instead of BAM** for anything you keep long-term (~40% smaller, needs the reference).
5. **Download on the `transfers` queue**, not the login node — `dtn1` has the fast external link.
6. **One directory per project, with a fixed layout** (see §8.4). Future-you will not remember
   what `test2_final_v3/` was.
7. **Clean `work/` after a Nextflow run finishes successfully.** It is often larger than the results.
   `nextflow clean -f -before <run_name>` removes it safely.
8. **Check free space before a big run.** A 24-sample RNA-seq run can produce several hundred GB;
   with 858 GB free on `/scratch/hphr`, two of those at once will fill it.

---

## 4. Submitting jobs

### 4.1 `bsub` basics

Two forms. The second is what you want almost always.

```bash
# Inline command
bsub -P hphr -q normal -J test -o test.%J.out -e test.%J.err "hostname; sleep 30"

# Run an existing script (preferred — reproducible, re-runnable)
bsub -P hphr -q normal -J test -o test.%J.out -e test.%J.err ./myscript.sh arg1 arg2
```

```
Job <627895> is submitted to queue <normal>.
         ↑ this number is your job ID — note it, you need it for everything else
```

Capture the ID in a script (this is the trick your `scripts/submit_prod.sh` already uses):

```bash
JID=$(bsub -P hphr -q normal -o o.%J ./step1.sh | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
echo "submitted $JID"
```

> 📖 Note: `bsub ./script.sh` passes the script as a command. LSF reads `#BSUB` directives
> **only** when the script is fed on **stdin**: `bsub < script.sh`. If you want `#BSUB` lines to
> take effect, use `bsub < script.sh`. If you use `bsub script.sh args`, put the options on the
> `bsub` command line instead. Both styles are shown below.

### 4.2 The options you need

| Option | Meaning | Example | Verified detail |
|---|---|---|---|
| `-P` | **Project — MANDATORY** | `-P hphr` | ✅ enforced by `esub.project` |
| `-q` | Queue | `-q bigmem` | ✅ default is `normal` |
| `-n` | **Slots (cores)**, not nodes | `-n 8` | ✅ `PARALLEL_SCHED_BY_SLOT=y` |
| `-R "span[hosts=1]"` | Keep all slots on one node | `-R "span[hosts=1]"` | ✅ already the queue default on all queues you use, but be explicit |
| `-R "rusage[mem=N]"` | **Reserve** N **MB** for the whole job | `-R "rusage[mem=48000]"` | ✅ MB; ✅ `RESOURCE_RESERVE_PER_TASK=N` → **per job, not per core** |
| `-M N` | Memory limit in **MB** (advisory here) | `-M 48000` | ✅ `LSF_ENFORCE_MEMLIMIT=N` → not enforced |
| `-W hh:mm` | Wall-clock run limit | `-W 24:00` | ✅ `ABS_RUNLIMIT=Y` → real time |
| `-J` | Job name | `-J star_sample1` | — |
| `-o` / `-e` | stdout / stderr file | `-o logs/x.%J.out` | `%J`=job ID, `%I`=array index |
| `-Is` | Interactive shell | `-Is /bin/bash` | needs `INTERACTIVE=YES` queue |
| `-w` | Dependency | `-w "done(627318)"` | ✅ `JOB_DEP_LAST_SUB=1` |
| `-g` | Job group (throttling) | `-g /sxt1558/sq_count` | — |
| `-J "n[1-24]"` | Job array | `-J "s[1-24]"` | ✅ max 1000 |
| `-gpu` | Request GPUs | `-gpu "num=1"` | ✅ `LSB_GPU_NEW_SYNTAX=extend` |
| `-u` / `-N` | Email on completion | `-N -u me@example.com` | ✅ mail via `hpc@ccs.miami.edu` |

#### ⚠️ The memory syntax that actually matters

✅ **VERIFIED** — `bparams -a` shows `RESOURCE_RESERVE_PER_TASK = N`. Therefore:

```bash
-n 8 -R "rusage[mem=32000]"    # reserves 32000 MB TOTAL for the 8-core job
                               # NOT 8 × 32000 MB
```

This is the single most-misunderstood LSF setting. On a cluster with
`RESOURCE_RESERVE_PER_TASK=Y` the same line would mean 256 GB. Here it means 32 GB.
Confirmed on your own running job: `-n 6 -R "rusage[mem=24000]"` →
`bjobs -l` reports `rusage[mem=24000.00]` and `MEMLIMIT 23.4 G`, not 140 GB.

**Use both `-R "rusage[mem=N]"` and `-M N` with the same number.** `rusage` is what makes the
scheduler hold the memory for you; `-M` is a label that shows up in `bjobs -l` / `bhist -l` and
documents your intent.

#### Threads: let LSF tell your program how many cores it has

```bash
# Inside your job script — never hardcode a thread count
STAR --runThreadN "$LSB_DJOB_NUMPROC" ...
samtools sort -@ "$LSB_DJOB_NUMPROC" ...
```

`$LSB_DJOB_NUMPROC` is the number of slots LSF actually gave you. Hardcoding `-@ 16` in a `-n 4`
job means 16 threads fighting over 4 cgroup-pinned cores — slower than 4 threads, and it looks
to the scheduler like you're wasting the node.

Other useful variables LSF sets inside a job: `$LSB_JOBID`, `$LSB_JOBINDEX` (array index),
`$LSB_JOBNAME`, `$LSB_HOSTS`, `$LSB_QUEUE`, `$LSB_SUB_CWD`.

### 4.3 Monitoring and cancelling

```bash
bjobs                      # your jobs (finished ones disappear after 1 hour)
bjobs -w                   # wide: full job names, no truncation
bjobs -r                   # running only
bjobs -p                   # pending only, WITH the reason — the first thing to run when stuck
bjobs -l 627325            # everything about one job: resources, memory used, host
bpeek 627325               # live stdout of a running job
bpeek -f 627325            # like tail -f
bkill 627325               # cancel one job
bkill 0                    # cancel ALL your jobs  ⚠️ no confirmation prompt
bkill -J 'prod_map_*'      # cancel by name pattern
bkill -q bigmem 0          # cancel all your jobs in one queue
bhist -l 627318            # history of a finished job (use after the 1-hour window)
```

### 4.4 Three practical examples

Create `logs/` first — ✅ **LSF does not create the output directory for you**, and the job will
fail silently if it is missing:

```bash
mkdir -p ~/squire_analysis/logs
cd ~/squire_analysis
```

#### Example 1 — a 1-minute test job

```bash
bsub -P hphr -q normal -J hello -n 1 \
     -R "rusage[mem=1000]" -M 1000 -W 0:05 \
     -o logs/hello.%J.out -e logs/hello.%J.err \
     "echo Running on \$(hostname); echo Slots: \$LSB_DJOB_NUMPROC; sleep 60; echo Done"
```

Then:

```bash
bjobs                        # watch it go PEND → RUN → (gone)
cat logs/hello.*.out         # should show a compute node name like n190, NOT login1
```

Run this first whenever something is wrong. If it works, LSF and your project are fine and the
problem is in your own script.

#### Example 2 — a Python script

`scripts/demo.py`:

```python
import os, sys, platform
print("host:", platform.node())
print("slots:", os.environ.get("LSB_DJOB_NUMPROC"))
print("python:", sys.version)
```

`scripts/run_python.sh`:

```bash
#!/bin/bash
#BSUB -P hphr
#BSUB -q normal
#BSUB -J py_demo
#BSUB -n 2
#BSUB -R "span[hosts=1] rusage[mem=4000]"
#BSUB -M 4000
#BSUB -W 0:30
#BSUB -o logs/py_demo.%J.out
#BSUB -e logs/py_demo.%J.err
set -euo pipefail

module purge
module load python/3.8.7          # ✅ verified available

python3 scripts/demo.py
```

Submit it — note the `<` so the `#BSUB` lines are read:

```bash
chmod +x scripts/run_python.sh
bsub < scripts/run_python.sh
```

#### Example 3 — an R script

`scripts/demo.R`:

```r
cat("host:", Sys.info()[["nodename"]], "\n")
cat("slots:", Sys.getenv("LSB_DJOB_NUMPROC"), "\n")
cat("R:", R.version.string, "\n")
```

`scripts/run_R.sh`:

```bash
#!/bin/bash
#BSUB -P hphr
#BSUB -q normal
#BSUB -J r_demo
#BSUB -n 1
#BSUB -R "span[hosts=1] rusage[mem=8000]"
#BSUB -M 8000
#BSUB -W 1:00
#BSUB -o logs/r_demo.%J.out
#BSUB -e logs/r_demo.%J.err
set -euo pipefail

module purge
module load R/4.4.3              # ✅ verified available (/share/rocky/Modules)

Rscript scripts/demo.R
```

```bash
bsub < scripts/run_R.sh
```

### 4.5 Reusable submission template

Save as `~/squire_analysis/01_lsf/template.sh` and copy it for each new job.

```bash
#!/bin/bash
# =============================================================================
#  LSF job template — Pegasus (UM)
#  Submit with:   bsub < 01_lsf/template.sh
#  (The `<` matters: LSF only reads #BSUB lines from stdin.)
# =============================================================================

#BSUB -P hphr
#   Project. MANDATORY on Pegasus — submission is rejected without it.
#   Your only project is `hphr`.

#BSUB -q bigmem
#   Queue. normal = ≤16 cores / ~60 GB / no time limit
#          bigmem = ≤16 cores / ~250 GB / max 240 h, no interactive
#          sccc   = ≤64 cores / ~500 GB (preemptable)
#          sccc_bigmem = ≤64 cores / 3.9 TB (preemptable)

#BSUB -J my_job
#   Job name. Shows in bjobs. Use something greppable, e.g. star_${SAMPLE}.

#BSUB -n 8
#   Number of SLOTS (= cores), not nodes.
#   Hard cap: 16 on normal, 512 on parallel, 4 on transfers.

#BSUB -R "span[hosts=1]"
#   Keep all slots on ONE node. Required for any threaded (non-MPI) program.

#BSUB -R "rusage[mem=32000]"
#   Reserve 32000 MB = ~31 GB for the WHOLE JOB.
#   Units are MB (LSF_UNIT_FOR_LIMITS=MB).
#   This is per-JOB, not per-core (RESOURCE_RESERVE_PER_TASK=N).
#   This is the line that actually reserves memory. Don't omit it.

#BSUB -M 32000
#   Memory limit in MB. Advisory on this cluster (LSF_ENFORCE_MEMLIMIT=N):
#   LSF will NOT kill you for exceeding it. Keep it equal to rusage so that
#   bjobs -l / bhist -l document what you intended.

#BSUB -W 24:00
#   Wall-clock limit, hh:mm. Real time (ABS_RUNLIMIT=Y).
#   Not required outside bigmem, but ALWAYS set it: a job with a time limit
#   can be backfilled into scheduling gaps and usually starts sooner.

#BSUB -o logs/my_job.%J.out
#BSUB -e logs/my_job.%J.err
#   stdout / stderr. %J = job ID, %I = array index.
#   ⚠️ `logs/` MUST already exist — LSF will not create it.
#   Paths are relative to the directory you ran bsub from.

# #BSUB -N -u swarajmthorat@gmail.com    # email a summary when the job ends
# #BSUB -w "done(123456)"                # wait for job 123456 to succeed
# #BSUB -g /sxt1558/mygroup              # job group, for throttling (see §8.7)
# #BSUB -gpu "num=1"                     # request 1 GPU (sccc_gpu / gpu_h100)

# ----------------------------- job body --------------------------------------
set -euo pipefail          # stop on first error, on unset variable, and in pipes
                           # WITHOUT this, a failed step still "succeeds" (DONE)

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Job name : $LSB_JOBNAME"
echo "Queue    : $LSB_QUEUE"
echo "Host     : $(hostname)"
echo "Slots    : $LSB_DJOB_NUMPROC"
echo "Workdir  : $(pwd)"
echo "Started  : $(date)"
echo "=========================================="

# --- node-local scratch, cleaned up even if the job fails or is killed -------
export TMPDIR=/tmp/$USER.$LSB_JOBID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# --- software ----------------------------------------------------------------
module purge                    # always start clean; inherited modules cause
                                # "works on login, fails in job" bugs
module load python/3.8.7
# module load R/4.4.3
# module load nextflow/24.10.6

# --- your work ---------------------------------------------------------------
# Pass $LSB_DJOB_NUMPROC to anything threaded, never a hardcoded number.
# e.g.  STAR --runThreadN "$LSB_DJOB_NUMPROC" ...

echo "...do the actual work here..."

# -----------------------------------------------------------------------------
echo "Finished : $(date)"
echo "Check actual usage with:   bhist -l $LSB_JOBID | grep -A2 'MEMORY USAGE'"
```

---

## 5. Common LSF commands

### 5.1 Cheat sheet

| Command | What it does | Practical example |
|---|---|---|
| `bsub` | Submit a job | `bsub -P hphr -q normal -n 4 -R "rusage[mem=8000]" -W 1:00 -o logs/j.%J.out ./run.sh` |
| `bjobs` | Your current jobs | `bjobs -w` |
| `bjobs -l` | Full detail on one job: resources, host, memory used | `bjobs -l 627325` |
| `bjobs -p` | **Pending jobs + why** | `bjobs -p` |
| `bqueues` | Queue list, priority, status, backlog | `bqueues -u sxt1558` |
| `bqueues -l` | Queue config: limits, hosts, fairshare, preemption | `bqueues -l bigmem` |
| `bhosts` | Host status and slot usage per queue | `bhosts bigmem` |
| `bhosts -l` | One host in detail: load, reserved memory, why closed | `bhosts -l n521` |
| `lsload` | Live load + **free** memory per node | `lsload n061 n190` |
| `lshosts` | **Static** hardware: cores, total RAM, model | `lshosts n521 n128` |
| `bkill` | Cancel a job | `bkill 627325` / `bkill -J 'prod_*'` |
| `bpeek` | Live stdout of a running job | `bpeek -f 627325` |
| `bhist -l` | Full history of a finished job, **incl. MAX MEM** | `bhist -l 627318` |
| `bacct` | Accounting summary over many jobs | `bacct -u sxt1558 -q bigmem` ⚠️ **slow** |
| `busers` | Your job counts and any user limits | `busers sxt1558` |
| `blimits` | Active resource limits | `blimits` (→ none configured here) |
| `bparams -a` | Every cluster scheduling parameter | `bparams -a \| grep ARRAY` |
| `bjgroup -s` | Job groups and their throttles | `bjgroup -s /sxt1558` |
| `bmgroup -w` | Which hosts are in which host group | `bmgroup -w \| grep ^sccc` |
| `bstop` / `bresume` | Suspend / resume your own job | `bstop 627325` |
| `bmod` | Change a **pending** job's request | `bmod -W 48:00 627336` |
| `brequeue` | Re-run a finished/failed job | `brequeue 627318` |
| `lsid` | Cluster name and LSF version | `lsid` |

> ⚠️ **`bacct` is very slow on this cluster.** It scans the full accounting log — `bacct -l <jobid>`
> did not return within 120 s during this inspection. For a single job use **`bhist -l <jobid>`**,
> which returned instantly and includes `MAX MEM`. Reserve `bacct` for bulk reports and expect to wait.

### 5.2 Checking job efficiency — did I ask for the right resources?

This is the highest-value habit on the whole cluster. After a job finishes:

```bash
bhist -l 627318 | grep -A2 'MEMORY USAGE'
```

Real output from your own job `627318` (`prod_map_LacZ-DMSO-Rep1`, `-n 6 -M 48000 -W 24:00`):

```
 RUNLIMIT          1440.0 min
 MEMLIMIT          46.8 G
 MEMORY USAGE:     MAX MEM: 36.8 Gbytes;  AVG MEM: 25.4 Gbytes
 Done successfully. The CPU time used is 18447.0 seconds.
 Dispatched 09:32:14 → Done 10:37:53   (65 min wall)
```

**How to read this:**

| Metric | Value | Verdict |
|---|---|---|
| Memory | 36.8 GB peak of 46.8 GB requested | **Good.** ~79% used — healthy headroom |
| Wall time | 65 min of 1440 min (`-W 24:00`) | Over-requested, but harmless here |
| CPU efficiency | 18447 s CPU ÷ (3939 s wall × 6 cores) = **78%** | **Good.** Real parallelism |

The CPU-efficiency formula is the useful one:

```
efficiency = CPU_time / (wall_time × cores)
```

- **>70%** — well-sized. Leave it alone.
- **~1/cores** (e.g. 16% on 6 cores) — your program is **single-threaded**. You are wasting
  5 of 6 cores. Drop `-n` to 1, or pass `$LSB_DJOB_NUMPROC` to the tool.
- **>100%** — more threads than slots. The cgroup is throttling you. Raise `-n` or lower the thread count.

Compare with a second job of yours, `627325` (`prod_count`, `-n 6 -M 24000`):

```
 MEMORY USAGE:  MAX MEM: 3.3 Gbytes;  AVG MEM: 880 Mbytes
 NTHREAD: 8      The CPU time used is 3738 seconds
```

3.3 GB peak against 24 GB requested — **7× over-requested**. That 21 GB of unused reservation is
memory LSF held out of the pool, which is part of why `bigmem` showed 262 pending jobs. Right-sizing
this to `-M 6000` would let more of your own jobs run concurrently.

```bash
# Live check on a RUNNING job
bjobs -l 627325 | grep -A3 'MEMORY USAGE'

# Right-size from history: peak memory across your recent jobs
for j in $(bhist -n 1 -d 2>/dev/null | awk '$1 ~ /^[0-9]+$/{print $1}' | head -20); do
  printf "%s  " "$j"
  bhist -l "$j" 2>/dev/null | grep -o 'MAX MEM: [^;]*' || echo "n/a"
done
```

**Rule:** request **peak × 1.3**, rounded up. Then stop guessing — measure once per pipeline step
and reuse the number.

### 5.3 Pending reasons

```bash
bjobs -p                # the reason is printed under each job
bjobs -l <jobid> | grep -A4 'PENDING REASON'
```

✅ **VERIFIED** — `lsb.params` sets `INCLUDE_DETAIL_REASONS=y` and
`PENDING_TIME_RANKING = short[1,20] medium[21,400] long[401,]`, so reasons here are specific.
Real reasons from your pending jobs on 2026-11-03:

| Reason seen | Means | What to do |
|---|---|---|
| `Job slot limit reached: 6 hosts` | Those nodes are full | Normal. Wait, or try a less-busy queue (§2.6) |
| `Job dependency condition not satisfied` | Waiting on `-w "done(...)"` | Normal. ⚠️ If the parent **failed**, this pends forever — `bkill` it |
| `Not enough job slot(s)` | Queue has no free slots | Reduce `-n`, or switch queue |
| `Job's requirements for reserving resource (mem) not satisfied` | No node has that much free RAM | Lower `rusage[mem=]`, or use `bigmem`/`sccc_bigmem` |
| `Job requirements for reserving resources not satisfiable` | **No node can *ever* satisfy this** | You asked for more than any node has. Fix the request |
| `User has reached ... job group limit` | Your own `-g` throttle | Expected — `bgmod -L N` to change it |

**The "pending forever" check:** if a request is physically impossible, LSF will not tell you at
submit time (`LSF_STRICT_RESREQ=Y` means it silently never matches). Confirm a host exists:

```bash
# Does any host in this queue have ≥400 GB total RAM and ≥32 cores?
lshosts $(bhosts sccc | awk 'NR>1{print $1}') | awk 'NR>1 && $5>=32'
```

---

## 6. Software environments

### 6.1 Modules

✅ **VERIFIED** — Pegasus uses **Environment Modules** (`module` is a bash function calling
`/share/Modules/$MODULE_VERSION/bin/modulecmd`), with modulefiles spread over several trees:

```
/share/Modules/{general,bio,phy,gw-X11,hihg}
/share/mfiles/{Core,Libs,Compiler/gcc/8.3.0,Compilers/gcc,Labs/sccc,Labs/cheminfo,Labs/cpdipt}
/share/rocky/Modules
/etc/scl/modulefiles
```

```bash
module avail                    # everything (long — it prints to stderr)
module avail 2>&1 | grep -i star    # search for a tool
module avail R/                 # all R versions
module list                     # what you have loaded
module load python/3.8.7
module unload python/3.8.7
module purge                    # unload everything — do this at the top of every job script
module show nextflow/24.10.6    # exactly what a module changes
module whatis igenomes/20240106 # one-line description
```

⚠️ **Nothing useful is on your PATH by default.** ✅ Verified on `login1`: `R`, `Rscript`, `conda`,
`mamba` and `nextflow` are all **absent** until you `module load`. And `python` on the default PATH
is LSF's internal interpreter (`/opt/ibm/lsfsuite/ext/gui/3.0/bin/python`, 3.6.8) — **not** a
module. Always `module load` the Python you mean.

> **`module purge` at the top of every job script.** Your interactive shell's modules do not
> reliably reach the compute node, and whatever *is* inherited may conflict. `module purge` followed
> by explicit `module load` lines is the difference between a reproducible script and a mystery.

### 6.2 Python

✅ **VERIFIED available:**

| Module | Tree |
|---|---|
| `python/3.8.7` **(default)** | `/share/Modules/general`, `/share/rocky/Modules` |
| `python/3.10.8` | `/share/mfiles/Labs/sccc` |
| `python/3.6.5`, `3.3.1` | `/share/Modules/general` |
| `python/2.7.3`, `2.7.11`, `2.7.15` | `/share/Modules/general` (legacy) |
| `py-pip/20.2`, `py-matplotlib/3.3.3` | `Core`, `Compiler/gcc/8.3.0` |

```bash
module load python/3.8.7
python3 -V

# Install packages for yourself (no root needed)
pip install --user pandas numpy
# ⚠️ --user installs to $HOME/.local, which counts against your 250 GB home.
# For anything large, use conda on scratch instead.
```

### 6.3 R

✅ **VERIFIED available:**

| Module | Tree |
|---|---|
| `R/4.4.3` | `/share/rocky/Modules` ← **newest** |
| `R/4.2.2`, `R/4.1.2` | `/share/mfiles/Labs/sccc` |
| `R/4.1.0` **(default)** | `/share/mfiles/Compiler/gcc/8.3.0`, `/share/rocky/Modules` |
| `R/4.0.3`, `R/3.6.3` | `/share/mfiles/Compiler/gcc/8.3.0` |
| `rstudio/2022.05.999` | `/share/mfiles/Compiler/gcc/8.3.0` |
| `JAGS/4.3.0`, `JAGS/3.4.0` | `Core`, `general` |

```bash
module load R/4.4.3
Rscript -e 'sessionInfo()'
```

For R packages, set a per-version library so an R upgrade doesn't break your installs:

```bash
# Add to ~/.Rprofile or your job script
export R_LIBS_USER=$HOME/R/%p-library/%v     # R expands %p (platform) and %v (version)
mkdir -p "$HOME/R"
Rscript -e 'install.packages("ggplot2", repos="https://cloud.r-project.org")'
```

⚠️ Bioconductor packages with heavy compiled dependencies (DESeq2 pulls in dozens) are often
easier via conda — see below. You already have `scripts/make_deseq2_env.sh`, which suggests
you took that route.

### 6.4 Conda

✅ **VERIFIED** — `module load miniforge3` (from `/share/mfiles/Core`). `module show miniforge3`:

```
conflict      conda anaconda
setenv        PYTHONUSERBASE    /nethome/sxt1558/.local
setenv        CONDA_ENVS_PATH   /nethome/sxt1558/.conda/envs
setenv        CONDARC           /nethome/sxt1558/.conda
setenv        CONDA_SOLVER      libmamba
setenv        CONDA_YES         true
prepend-path  MANPATH           /share/apps/c7/miniforge3/man
```

So: **conda environments land in `$HOME/.conda/envs` by default**, the fast `libmamba` solver is
already on, and `conda` will not prompt for confirmation (`CONDA_YES=true`).
`anaconda3/2022.10.lua` also exists but **conflicts** with `miniforge3` — load only one.

```bash
module purge
module load miniforge3

# IMPORTANT: this module does NOT run conda's shell hook, so `conda activate`
# may fail with "Your shell has not been properly configured". Fix it per-shell:
eval "$(conda shell.bash hook)"

conda env list
conda create -n rnaseq -c conda-forge -c bioconda star samtools salmon subread
conda activate rnaseq
```

⚠️ A bioinformatics conda env is easily 3–8 GB. Several of them will eat a visible fraction of
your **250 GB** home. To put them on scratch instead:

```bash
mkdir -p /scratch/hphr/rna_analysis/conda_envs
export CONDA_ENVS_PATH=/scratch/hphr/rna_analysis/conda_envs   # override the module's default
export CONDA_PKGS_DIRS=/scratch/hphr/rna_analysis/conda_pkgs   # the package cache is also large
conda create -n rnaseq ...
```

#### Using conda inside a batch script

`conda activate` needs the shell hook, which is **not** set up in a non-interactive job shell.
This is the single most common "works on login, fails in the job" error:

```bash
#!/bin/bash
#BSUB -P hphr
#BSUB -q bigmem
#BSUB -n 8
#BSUB -R "span[hosts=1] rusage[mem=40000]"
#BSUB -M 40000
#BSUB -W 12:00
#BSUB -o logs/star.%J.out
#BSUB -e logs/star.%J.err
set -euo pipefail

module purge
module load miniforge3
eval "$(conda shell.bash hook)"     # ← REQUIRED in batch scripts
conda activate rnaseq

STAR --runThreadN "$LSB_DJOB_NUMPROC" --genomeDir "$IDX" --readFilesIn "$R1" "$R2" ...
```

> The robust alternative, if the hook still misbehaves:
> `source "$(dirname "$(which conda)")/../etc/profile.d/conda.sh" && conda activate rnaseq`

### 6.5 Apptainer / Singularity containers

✅ **VERIFIED** — **Apptainer 1.4.2-1.el8 is already at `/usr/bin/apptainer`. No module needed.**
`/usr/bin/singularity` is a symlink to the same binary, so both names work.

Also available as modules (older, generally unnecessary):

| Module | Version |
|---|---|
| `apptainer/1.0.2` | `/share/Modules/general` |
| `singularity/3.8.7`, `singularity/3.5.2` | `/share/mfiles/Labs/sccc` |
| `singularity-ce/3.9.8` | `/share/mfiles/Labs/sccc` |

❌ **Docker is not installed and will never be** — it requires root. Apptainer runs Docker images
directly, so this is not a limitation:

```bash
# Pull a Docker image into a .sif file (do this ONCE, on the transfers queue or interactively)
cd /scratch/hphr/rna_analysis/apptainer_cache
apptainer pull docker://quay.io/biocontainers/star:2.7.11a--h0033a41_0

# Run a command inside it
apptainer exec star_2.7.11a--h0033a41_0.sif STAR --version

# ⚠️ Bind-mount the paths you need. Apptainer auto-mounts $HOME and $PWD but NOT /scratch.
apptainer exec -B /scratch/hphr:/scratch/hphr \
  star_2.7.11a--h0033a41_0.sif \
  STAR --runThreadN "$LSB_DJOB_NUMPROC" --genomeDir /scratch/hphr/rna_analysis/references/star
```

✅ **VERIFIED** — you already have `~/.apptainer` and `/scratch/hphr/rna_analysis/apptainer_cache`.
Point the cache at scratch so multi-GB image layers don't fill your home:

```bash
# Add to ~/.bashrc
export APPTAINER_CACHEDIR=/scratch/hphr/rna_analysis/apptainer_cache
export APPTAINER_TMPDIR=/scratch/hphr/rna_analysis/apptainer_cache/tmp
export NXF_SINGULARITY_CACHEDIR=/scratch/hphr/rna_analysis/apptainer_cache   # for Nextflow
```

✅ **VERIFIED** — none of these are currently set in your environment. Set them before your next
container pull, or images go to `$HOME/.apptainer/cache`.

### 6.6 Nextflow

✅ **VERIFIED available** (`module avail nextflow`):

| Module | Tree |
|---|---|
| `nextflow/25.10.5` | `/share/rocky/Modules` ← **newest** |
| `nextflow/25.04.2` | `/share/rocky/Modules` |
| `nextflow/24.10.6` | `/share/mfiles/Core` |
| `nextflow/24.10.0`, `24.04.4`, `23.10.0`, `22.10.4` | `Core` / `rocky` |

✅ `module show nextflow/24.10.6` reveals it also loads Java for you:

```
module       load java/17
prepend-path PATH /projectnb/pegasus/sw/nextflow/24.10.6
```

So `module load nextflow/24.10.6` is enough — no separate `module load java`.

```bash
module purge
module load nextflow/24.10.6
nextflow -version
```

For **nf-core/rnaseq**, prefer `24.10.6` or newer (nf-core pipelines now require Nextflow ≥23.04,
and recent releases ≥24.04).

### 6.7 Reference genomes

✅ **VERIFIED** — an `igenomes` module exists: `module load igenomes/20240106` adds
`/sccc/share/apps/igenomes/bin` to `PATH`.

⚠️ **UNVERIFIED** — I did not enumerate which genomes/builds are present, and note that
**`/sccc` is 100% full** (`df`: 2.4 PB used of 2.4 PB). Check before depending on it:

```bash
module load igenomes/20240106
ls /sccc/share/apps/igenomes/
df -h /sccc
```

If iGenomes doesn't have what you need, keep your own index in
`/scratch/hphr/rna_analysis/references` — which is what you already do.

---

## 6b. Nextflow + LSF

This is the part most worth getting right, because Nextflow and LSF can fight each other if
configured naively.

### 6b.1 How Nextflow integrates with LSF

Nextflow has an **LSF executor** built in. Set it and Nextflow stops running tasks locally and
starts translating each one into a `bsub`:

```groovy
process.executor = 'lsf'
```

The mental model:

```
You:       nextflow run nf-core/rnaseq ...
              │
              ▼
   ┌──────────────────────────┐
   │ Nextflow DRIVER process  │   1 long-lived process. Builds the DAG,
   │ (one `bsub`, 1–2 cores)  │   submits tasks, polls bjobs, handles retries.
   └──────────────────────────┘   Does almost no computation itself.
              │ bsub ×N
    ┌─────────┼─────────┬─────────┐
    ▼         ▼         ▼         ▼
 TRIM_1    TRIM_2    STAR_1    STAR_2      ← each is its own LSF job,
 (4 cores) (4 cores) (12 cores)(12 cores)    with its own PEND→RUN→DONE
```

### 6b.2 One LSF job per task — automatically

For every process instance, Nextflow:

1. creates a work directory `work/ab/cdef12.../` containing `.command.sh`, `.command.run`, and
   symlinks to the inputs;
2. builds a `bsub` line from that process's `cpus`, `memory`, `time`, `queue` directives;
3. submits it and records the job ID;
4. polls LSF for status;
5. on failure, applies `errorStrategy` (e.g. `retry` with more memory) and resubmits.

So a 24-sample nf-core/rnaseq run submits **hundreds** of LSF jobs. You will see them all in
`bjobs` with names like `nf-NFCORE_RNASEQ_...`. This is normal and correct.

> ⚠️ **Nextflow's LSF executor does not add `-P hphr`.** Because `esub.project` **rejects every
> job without a project**, a default nf-core run on Pegasus will fail at the *first task* with
> *"Error: no project specified."* You must set `clusterOptions = '-P hphr'`. This is the single
> most important line in your `nextflow.config` on this cluster.

### 6b.3 Launching Nextflow: directly vs through `bsub`

| | Run `nextflow` on the login node | **`bsub` the whole `nextflow` command** |
|---|---|---|
| Where the driver lives | `login1` | a compute node |
| Survives SSH disconnect | ❌ no — pipeline dies, orphaning running tasks | ✅ yes |
| Login-node load | 1 JVM (~1–2 GB) per pipeline | none |
| Logs | your terminal | `-o` file |
| Verdict | OK for a 5-minute `-profile test` | ✅ **Use this for every real run** |

Nextflow's driver is cheap but **long-lived** — an nf-core/rnaseq run is many hours. A dropped
SSH session kills it. Submit the driver as a job:

`01_lsf/run_rnaseq.sh`:

```bash
#!/bin/bash
#BSUB -P hphr
#BSUB -q normal
#BSUB -J nf_rnaseq
#BSUB -n 2
#BSUB -R "span[hosts=1] rusage[mem=8000]"
#BSUB -M 8000
#BSUB -W 168:00
#BSUB -o logs/nf_rnaseq.%J.out
#BSUB -e logs/nf_rnaseq.%J.err
set -euo pipefail

# Why these numbers: the driver is a JVM that submits and waits. 2 cores and 8 GB
# is plenty. -W must cover the WHOLE pipeline, not one task — note that `normal`
# has no queue RUNLIMIT, so 168:00 (7 days) is allowed. On `bigmem` the cap is 240 h.

module purge
module load nextflow/24.10.6        # also loads java/17

export NXF_SINGULARITY_CACHEDIR=/scratch/hphr/rna_analysis/apptainer_cache
export NXF_OPTS='-Xms1g -Xmx6g'     # keep the JVM inside the 8 GB we reserved

cd /nethome/sxt1558/squire_analysis

nextflow run nf-core/rnaseq \
  -r 3.14.0 \
  -profile apptainer \
  -c 01_lsf/pegasus.config \
  --input samples_prod.tsv \
  --outdir /scratch/hphr/rna_analysis/results/rnaseq \
  --fasta  /scratch/hphr/rna_analysis/references/genome.fa \
  --gtf    /scratch/hphr/rna_analysis/references/genes.gtf \
  -work-dir /scratch/hphr/rna_analysis/work \
  -with-report  logs/nf_report.html \
  -with-timeline logs/nf_timeline.html \
  -with-trace   logs/nf_trace.txt \
  -resume
```

```bash
bsub < 01_lsf/run_rnaseq.sh
```

> Keeping `-resume` in the script is deliberate: if the driver job itself dies and you resubmit,
> it picks up where it left off instead of starting over. On a first run it is a harmless no-op.

### 6b.4 `nextflow.config` for Pegasus

`01_lsf/pegasus.config`:

```groovy
// ---------------------------------------------------------------------------
//  Nextflow config for Pegasus (UM) + LSF
//  Verified against: LSF 10.1.0.12, LSF_UNIT_FOR_LIMITS=MB,
//                    RESOURCE_RESERVE_PER_TASK=N, esub.project mandatory
// ---------------------------------------------------------------------------

params {
    project = 'hphr'          // your only project
}

process {
    executor = 'lsf'

    // MANDATORY. Without -P the esub rejects every task.
    clusterOptions = "-P ${params.project}"

    queue = 'normal'          // ≤16 cores, ≤~60 GB, no run limit

    // Retry on transient failures and on out-of-memory.
    // 130 = SIGKILL (OOM killer), 137 = 128+9, 140 = runlimit, 143 = SIGTERM.
    errorStrategy = { task.exitStatus in [104,130,134,137,139,140,143,247] && task.attempt <= 3
                      ? 'retry' : 'finish' }
    maxRetries = 3
    maxErrors  = '-1'

    // ---- per-label resources -------------------------------------------
    // nf-core pipelines tag every process with one of these labels.
    withLabel: process_single {
        cpus = 1 ; memory = { 6.GB  * task.attempt } ; time = { 4.h  * task.attempt }
    }
    withLabel: process_low {
        cpus = 2 ; memory = { 12.GB * task.attempt } ; time = { 4.h  * task.attempt }
    }
    withLabel: process_medium {
        cpus = 6 ; memory = { 36.GB * task.attempt } ; time = { 8.h  * task.attempt }
        queue = 'bigmem'      // 251 GB nodes; `normal` tops out near 60 GB
    }
    withLabel: process_high {
        cpus = 12 ; memory = { 72.GB * task.attempt } ; time = { 16.h * task.attempt }
        queue = 'bigmem'
    }
    withLabel: process_long {
        time = { 48.h * task.attempt }
        queue = 'bigmem'      // but remember: bigmem caps at 240 h
    }
    withLabel: process_high_memory {
        cpus = 12 ; memory = { 200.GB * task.attempt } ; time = { 24.h * task.attempt }
        queue = 'bigmem'
    }

    // STAR genome generation is the usual memory spike in RNA-seq
    withName: 'STAR_GENOMEGENERATE' {
        cpus = 12 ; memory = 80.GB ; time = 12.h ; queue = 'bigmem'
    }
    withName: '.*STAR_ALIGN.*' {
        cpus = 12 ; memory = { 48.GB * task.attempt } ; time = { 12.h * task.attempt }
        queue = 'bigmem'
    }
}

executor {
    // Be a good neighbour: this cluster already had ~2900 jobs pending in `normal`.
    queueSize      = 50        // max concurrent LSF jobs from this pipeline
    submitRateLimit = '10/1min' // don't hammer mbatchd
    pollInterval   = '30 sec'
    queueStatInterval = '2 min'
    exitReadTimeout = '5 min'   // GPFS can lag; avoids spurious "missing .exitcode"
    perJobMemLimit = true       // ⚠️ see the note below — important on this cluster
    jobName = { "nf_${task.process.tokenize(':').last()}_${task.index}" }
}

apptainer {
    enabled    = true
    autoMounts = true
    cacheDir   = '/scratch/hphr/rna_analysis/apptainer_cache'
    // Apptainer auto-mounts $HOME and $PWD but not /scratch:
    runOptions = '-B /scratch/hphr:/scratch/hphr'
}

workDir = '/scratch/hphr/rna_analysis/work'    // never in $HOME

timeline { enabled = true ; file = 'logs/nf_timeline.html' ; overwrite = true }
report   { enabled = true ; file = 'logs/nf_report.html'   ; overwrite = true }
trace    { enabled = true ; file = 'logs/nf_trace.txt'     ; overwrite = true
           fields = 'task_id,name,status,exit,attempt,duration,realtime,%cpu,peak_rss,peak_vmem' }
dag      { enabled = true ; file = 'logs/nf_dag.html'      ; overwrite = true }
```

#### ⚠️ `perJobMemLimit = true` — why it matters here

📖 **GENERAL LSF / Nextflow:** by default the Nextflow LSF executor emits
`-M <memory/cpus>`, i.e. it divides your `memory` by `cpus`, because many LSF clusters set
`RESOURCE_RESERVE_PER_TASK=Y` (memory interpreted per-slot).

✅ **VERIFIED** — Pegasus sets **`RESOURCE_RESERVE_PER_TASK = N`** (`bparams -a`): memory here is
**per job**. So the default division would make Nextflow request **`cpus`× too little memory** —
a `memory = 48.GB, cpus = 12` task would ask for only 4 GB.

Setting `executor.perJobMemLimit = true` makes Nextflow pass the **full** value, which is what
this cluster expects. Verify it on your first run:

```bash
# Look at what Nextflow actually submitted for one task
cat /scratch/hphr/rna_analysis/work/ab/cdef12*/.command.run | grep '^#BSUB'
```

You should see `#BSUB -M 49152` (48 GB in MB) for a 48 GB task — **not** `#BSUB -M 4096`.
If you see the small number, `perJobMemLimit` isn't taking effect.

#### Queue choice in Nextflow, given what we measured

Your own `prod_map` step peaked at **36.8 GB**. `normal` nodes have 31.3 GB or 62.8 GB, mixed —
so a 36.8 GB task would only fit the 62.8 GB subset and pend a long time, or land on a 31 GB node
and get OOM-killed. That is exactly why the config above routes `process_medium` and above to
`bigmem` (251 GB nodes).

### 6b.5 Monitoring a Nextflow run

```bash
# --- From LSF's side ---
bjobs -w | grep nf_                      # all Nextflow task jobs
bjobs -w | grep -c nf_                   # how many are live
bjobs -p | grep -A1 nf_                  # which are stuck and why
bjobs -r -w | grep nf_ | wc -l           # how many are actually running

# --- From Nextflow's side ---
tail -f logs/nf_rnaseq.*.out             # the driver's live progress table
nextflow log                             # all runs: name, status, duration
nextflow log <run_name> -f name,status,exit,duration,peak_rss,workdir
nextflow log <run_name> -F 'status == "FAILED"'   # just the failures

# --- The trace file is the best tool for right-sizing ---
# Columns: realtime, %cpu, peak_rss — compare peak_rss to what you requested
column -t -s$'\t' logs/nf_trace.txt | less -S
awk -F'\t' 'NR>1 {print $NF, $(NF-1), $2}' logs/nf_trace.txt | sort -h | tail -20
```

After the run, open `logs/nf_report.html` — it has a per-process resource-usage section showing
requested vs actual CPU and memory. That is the fastest way to fix over-requests for the next run.

### 6b.6 `-resume`

Nextflow caches each task by a hash of its inputs, parameters, container and script.
`-resume` reuses every cached task and only re-runs what changed or failed.

```bash
# Resume the most recent run
nextflow run nf-core/rnaseq -r 3.14.0 -profile apptainer -c 01_lsf/pegasus.config ... -resume

# Resume a specific earlier run (names come from `nextflow log`)
nextflow run ... -resume cheeky_euler
```

**When `-resume` silently re-runs everything** (the usual frustrations):

| Cause | Fix |
|---|---|
| `work/` was deleted or purged | Nothing to resume from. Re-run. **Don't delete `work/` until you're done.** |
| `-work-dir` changed between runs | Always pass the same `-work-dir`. Better: set `workDir` in the config, as above. |
| Input file **timestamp or content** changed | Re-staging a FASTQ with `cp` changes its mtime. Use symlinks, or don't touch inputs. |
| Pipeline version (`-r`) changed | Expected — the script hash changed. |
| Container image changed | Expected. |
| Launched from a different directory | Run from the same directory, or use absolute paths everywhere. |

```bash
# Clean up old work dirs but KEEP the latest resumable one
nextflow log                             # list runs
nextflow clean -f -before <run_name>     # delete work dirs of runs before this one
nextflow clean -f -but <run_name>        # keep only this run's
```

### 6b.7 `work/` vs results, and where they belong

| Directory | What's in it | Size | Keep? |
|---|---|---|---|
| `work/` | one dir per task: `.command.sh`, `.command.run`, `.command.log`, `.command.err`, `.exitcode`, input symlinks, **and all intermediate output files** | **Very large** — often bigger than the results | Until the run is fully done, then delete |
| `--outdir` | published results only (counts, BAMs, MultiQC, QC) | Moderate | ✅ Yes |
| `.nextflow/` | the resume cache database | Small | Keep while you might `-resume` |
| `.nextflow.log` | driver log — the first place to look on failure | Small | Keep |

✅ **VERIFIED** — your `work/` is already correctly on scratch:
`/scratch/hphr/rna_analysis/work` with **253 subdirectories**. Good. Keeping it there rather than
in your 250 GB `$HOME` is the right call — an nf-core/rnaseq `work/` for 24 samples can exceed 500 GB,
and you have 858 GB free on `/scratch/hphr`.

**Check before and after a big run:**

```bash
df -h /scratch/hphr
du -sh /scratch/hphr/rna_analysis/work
```

### 6b.8 Troubleshooting failed Nextflow tasks

Nextflow tells you the work directory of every failure. **Go there** — everything you need is in it.

```bash
# The error message names the dir, e.g.:
#   Work dir: /scratch/hphr/rna_analysis/work/3f/a1b2c3d4e5...
cd /scratch/hphr/rna_analysis/work/3f/a1b2c3d4e5*

cat .command.sh      # the exact script that ran — read this first
cat .command.err     # the tool's stderr — the actual error
cat .command.log     # stdout + stderr combined
cat .exitcode        # the exit status
cat .command.run     # the LSF wrapper, incl. the #BSUB lines Nextflow generated
ls -la               # are the input symlinks intact, or dangling?

# Reproduce it by hand, exactly as the task ran:
bash .command.run    # runs locally — fine for a quick check, don't do heavy work on login1
```

| Symptom | Likely cause | Fix |
|---|---|---|
| **Every** task fails instantly; `.command.err` says *"no project specified"* | `clusterOptions` missing | Add `clusterOptions = '-P hphr'` |
| Exit 130 / 137, `.command.err` truncated or empty | **OOM** — killed by the kernel | Raise `memory`; route to `bigmem`; check `peak_rss` in the trace |
| Exit 140 | LSF run limit hit | Raise `time` |
| Tasks pend forever, `bjobs -p` → *"requirements for reserving resources not satisfiable"* | Asked for more RAM/cores than any node in that queue has | Lower the request or change `queue` |
| `#BSUB -M` is `cpus`× too small | `perJobMemLimit` not set | Set `executor.perJobMemLimit = true` |
| *"Missing output file(s) ... expected by process"* | The tool wrote nothing, or to a different name | Read `.command.err`; check the `output:` glob |
| *"Failed to get exit status ... .exitcode is empty"* | GPFS metadata lag, or the job was preempted | Raise `executor.exitReadTimeout`; avoid `sccc*` queues for Nextflow |
| Container errors: *"no such file or directory"* for a path under `/scratch` | `/scratch` not bind-mounted | `runOptions = '-B /scratch/hphr:/scratch/hphr'` |
| Driver dies after hours with no error | SSH disconnected | `bsub` the driver (§6b.3) |
| Thousands of jobs flood the queue | No throttle | `executor.queueSize = 50`, `submitRateLimit = '10/1min'` |

**The resource-aware retry pattern** (already in the config above) is the most useful trick:
`memory = { 36.GB * task.attempt }` means attempt 1 asks 36 GB, attempt 2 asks 72 GB, attempt 3
asks 108 GB. Combined with `errorStrategy = 'retry'` on OOM exit codes, the pipeline
self-corrects instead of making you re-run it by hand.

---

## 7. Common errors and troubleshooting

### 7.1 Job stuck in `PEND`

```bash
bjobs -p                                    # the reason, for all pending jobs
bjobs -l <jobid> | grep -A4 'PENDING REASON'
```

| Reason | Fix |
|---|---|
| `Job slot limit reached` / `Not enough job slots` | Queue is busy. Check §2.6 and switch to a queue with free hosts — on 2026-11-03 `sccc` was nearly idle while `bigmem` was full. |
| `Job dependency condition not satisfied` | Normal **if the parent is still running**. ⚠️ If the parent **EXITed**, the child pends forever: `bkill` it. |
| `requirements for reserving resource (mem) not satisfied` | No node has that much free right now. Wait, lower it, or move to `bigmem`/`sccc_bigmem`. |
| `requirements ... not satisfiable` | **Impossible request.** You asked for more than any node in the queue physically has. |
| Low fairshare priority | You've run a lot recently. Submitting more jobs makes this worse, not better. |

```bash
# Is my request even satisfiable? List hosts in the queue with their specs:
lshosts $(bhosts bigmem | awk 'NR>1{print $1}') 2>/dev/null | awk 'NR==1 || $5!="-"'

# Where is my pending job's dependency?
bjobs -l <jobid> | grep -i depend
```

### 7.2 Job killed for memory / out-of-memory

⚠️ **Pegasus-specific:** because `LSF_ENFORCE_MEMLIMIT=N`, LSF does **not** enforce `-M`. You will
**not** see LSF's usual *"TERM_MEMLIMIT: job killed after reaching memory usage limit"*. Instead the
**Linux OOM killer** takes your process, which looks like:

- exit code **130**, **137** (128+9 = SIGKILL), or just a truncated log;
- `.command.err` / your `-e` file ending mid-sentence with no error message;
- `Killed` in the output;
- sometimes *nothing at all* — the job just says `EXIT`.

```bash
# What did it actually use before dying?
bhist -l <jobid> | grep -A2 'MEMORY USAGE'
```

**Fix:** raise `-R "rusage[mem=N]"` **and** `-M N` together, and move to a bigger queue if the
node can't hold it:

| Peak memory needed | Queue |
|---|---|
| ≤ 28 GB | `normal` (safe on both 31 GB and 62 GB nodes) |
| 28–55 GB | `normal` ⚠️ only the 62.8 GB nodes — expect longer pends. `bigmem` is often faster |
| 55–240 GB | `bigmem` |
| 240–490 GB | `sccc` |
| 490 GB – 3.8 TB | `sccc_bigmem` |

⚠️ **The 31 GB / 62.8 GB mix in `normal` is a real trap.** ✅ Verified: 86 nodes have 31.3 GB and
101 have 62.8 GB. A job asking 40 GB with `rusage` set correctly will be placed only on a 62 GB
node — fine. But a job that *declares* 8 GB and actually uses 40 GB will be OOM-killed on a 31 GB
node and succeed on a 62 GB one. **Same script, random failures.** This is almost always the
explanation for "it worked yesterday". Always declare your real peak.

### 7.3 Runtime limit exceeded

LSF sends SIGUSR2 then SIGKILL. You'll see `TERM_RUNLIMIT` in the log, or exit **140**.

```bash
bhist -l <jobid> | grep -E 'RUNLIMIT|TERM_'
```

**Fix:** raise `-W`. Remember `ABS_RUNLIMIT=Y` (wall-clock) and that **`bigmem` caps at 240 h**.
For a pending job you can change it in place:

```bash
bmod -W 48:00 <jobid>     # pending jobs only — a running job's limit cannot be raised
```

For anything that genuinely needs more than 10 days: split it into chunks with `-w` dependencies,
or use `normal` (no run limit).

### 7.4 CPU or memory request rejected / never scheduled

Requests above a queue's `PROCLIMIT` are **rejected at submit time**:

```bash
bsub -P hphr -q normal -n 32 ...
# → Too many tasks requested. Job not submitted.   (PROCLIMIT=16 on normal)
```

| Queue | Max `-n` |
|---|---|
| `normal` | 16 |
| `transfers` | 4 |
| `interactive` | 1 |
| `parallel` | 512 |
| `bigmem`, `sccc`, `sccc_bigmem`, `sccc_gpu` | node core count (16 / 64) |

Memory over-requests are **not** rejected — they pend forever. See §7.1.

If you need >16 cores on one node, use `sccc` (64-core nodes). If you need >16 cores across
several nodes, use `parallel` — but note ✅ `RES_REQ = span[ptile=16]`, which forces exactly 16
cores per node, and ✅ `INTERACTIVE = NO`.

### 7.5 Missing module or command

```
scripts/run.sh: line 12: Rscript: command not found
```

Almost always **a module loaded in your shell but not in the job script**. ✅ Verified: `R`,
`Rscript`, `conda`, `nextflow` are **not** on the default PATH.

```bash
module avail 2>&1 | grep -i <tool>    # find the right name/version
module show <module>                  # confirm what it adds to PATH
```

**Fix:** put explicit `module purge` + `module load` lines in the script. Then verify, early:

```bash
module purge
module load R/4.4.3
command -v Rscript || { echo "FATAL: Rscript not on PATH"; exit 1; }
```

That two-line guard turns a 6-hour job that fails at hour 5 into one that fails in 3 seconds.

### 7.6 Permission denied

```bash
chmod +x scripts/myscript.sh          # most common cause

ls -ld /scratch/hphr                  # drwxrws---+ root hphr → you need group hphr
id                                    # ✅ you are in: ccsuser, hphr

# The `+` means an ACL is set. Inspect it:
getfacl /scratch/hphr/rna_analysis
```

⚠️ `/scratch/hphr` is **setgid** (`drwxrws---`): new files there get group `hphr` automatically, so
labmates can read them. If you `cp` a file in from elsewhere and it ends up group `ccsuser`,
fix it with `chgrp -R hphr <path>`.

For a job that can't write its output: check the **execution** directory, not your login one.
`bjobs -l` shows `Execution CWD`. A relative `-o logs/x.out` resolves against the directory you
ran `bsub` from — if that path doesn't exist on the compute node, the job fails.

### 7.7 Disk quota exceeded

```
cannot write: Disk quota exceeded
```

```bash
df -h $HOME /scratch/hphr
du -sh $HOME/* | sort -rh | head
du -sh /scratch/hphr/rna_analysis/* | sort -rh
```

⚠️ **UNVERIFIED** — `mmlsquota` does not work from your account (§3.3), so you cannot see the
official number. Use `df` as the proxy: **250 GB home**, **2.0 TB `/scratch/hphr`**
(858 GB free on 2026-11-03).

Usual culprits, in order:
1. `work/` from an old Nextflow run → `nextflow clean -f -before <run>`
2. Conda environments and the conda package cache in `$HOME/.conda` → move to scratch (§6.4)
3. Apptainer image cache in `$HOME/.apptainer` → set `APPTAINER_CACHEDIR` (§6.5)
4. `pip install --user` packages in `$HOME/.local`
5. Intermediate BAMs you never deleted

### 7.8 Job finishes but there is no output

**This is the error `set -euo pipefail` exists to prevent.** Without it, a bash script whose
commands fail still exits 0, and LSF reports `DONE`.

Checklist:

```bash
# 1. Did it really succeed?
bhist -l <jobid> | grep -E 'Done successfully|Exited'

# 2. Read stderr. Actually read it.
cat logs/myjob.<jobid>.err

# 3. Where did it run, and from where?
bjobs -l <jobid> | grep -E 'CWD|Output File|Error File'
```

| Cause | Fix |
|---|---|
| No `set -euo pipefail` → a failed step was ignored | Add it as the first line after the `#BSUB` block |
| Relative output path, different CWD than you expected | Use absolute paths, or `cd` explicitly at the top of the script |
| Output written to `/tmp` on the compute node | `/tmp` is node-local and wiped. Copy results to `/scratch` before the job ends |
| The tool wrote to stdout, which went to the `-o` file | That's where it is — check the `.out` file |
| A `trap ... EXIT` cleanup deleted the output | Make sure the trap only removes `$TMPDIR` |
| OOM-killed mid-write | See §7.2 |

### 7.9 Problems with log paths

✅ **LSF does not create the output directory.** If `logs/` doesn't exist, you lose the log — and
with it any chance of diagnosing the failure.

```bash
mkdir -p logs          # once per project, before your first bsub
```

| Problem | Fix |
|---|---|
| Directory doesn't exist | `mkdir -p logs` first |
| All jobs overwrite one file | Use `%J`: `-o logs/job.%J.out`. For arrays: `-o logs/job.%J.%I.out` |
| Relative path resolved somewhere unexpected | `bsub` resolves it against your submit CWD. Use absolute paths if unsure |
| stdout and stderr interleaved confusingly | Use separate `-o` and `-e` — don't merge them |
| Log is empty while the job runs | Normal: LSF buffers and writes at completion. Use `bpeek -f <jobid>` to watch live |
| Want to append across runs | `-oo` / `-eo` overwrite; plain `-o` / `-e` append to an existing file |

```bash
# A pattern that keeps logs organised and never collides
mkdir -p logs/$(date +%Y%m%d)
bsub -P hphr -q normal -J step1 \
     -o logs/$(date +%Y%m%d)/step1.%J.out \
     -e logs/$(date +%Y%m%d)/step1.%J.err ./step1.sh
```

---

## 8. Best practices

### 8.1 Choosing CPU and RAM

**Measure, don't guess.** The workflow is always the same:

1. Run **one** sample with a generous request.
2. `bhist -l <jobid> | grep -A2 'MEMORY USAGE'` → read `MAX MEM`.
3. Request `MAX MEM × 1.3` for the rest.
4. Compute `CPU_time / (wall × cores)`. Below ~50%? Lower `-n`.

Starting points for your work, with the reasoning:

| Task | `-n` | `rusage[mem=]` | `-W` | Queue | Why |
|---|---|---|---|---|---|
| FastQC | 2 | 4000 | 1:00 | `normal` | Single-threaded per file, tiny memory |
| Trimming (fastp/trimmomatic) | 4–6 | 16000 | 6:00 | `normal` | I/O-bound; your own runs used `-n 6 -M 16000` |
| STAR index build (human) | 12 | 80000 | 12:00 | `bigmem` | The memory spike of the whole pipeline |
| STAR align (human) | 8–12 | 48000 | 12:00 | `bigmem` | Your `prod_map` peaked at **36.8 GB** → 48 GB is right |
| salmon / kallisto | 8 | 16000 | 2:00 | `normal` | Much lighter than STAR |
| samtools sort | 4 | 16000 | 4:00 | `normal` | Use `-T $TMPDIR` on local disk |
| featureCounts | 4 | 8000 | 2:00 | `normal` | Modest |
| SQuIRE Count | 6 | **6000** | 48:00 | `bigmem` | ⚠️ Your runs asked 24 GB but **peaked at 3.3 GB** — see below |
| DESeq2 / R analysis | 1 | 16000 | 2:00 | `normal` | Mostly single-threaded |
| Nextflow driver | 2 | 8000 | 168:00 | `normal` | Just submits and waits |

> ⚠️ **A concrete over-request in your current jobs.** `prod_count` (job 627325) requested
> `-M 24000` and peaked at **3.3 GB** (`MAX MEM: 3.3 Gbytes; AVG MEM: 880 Mbytes`). That is **7×**
> more than needed. Each such job reserves ~21 GB that nobody can use. With `bigmem` showing
> **262 pending jobs** and 0 free hosts on 2026-11-03, cutting this to `-M 6000` would let several
> more of your own jobs run at once. Verify across a few samples first — peak memory can be
> input-dependent — but this looks like a clear win.

### 8.2 Don't run analyses on the login node

`login1` is ✅ 16 cores and 188 GB shared by **everyone logged in**. One STAR run makes the cluster
unusable for dozens of people; admins kill such processes, and repeat offenders lose access.

**Fine on `login1`:** editing files, `bsub`/`bjobs`, `ls`, `grep`, `head`, small `awk`, `git`,
`module avail`, reading logs.

**Use a job instead for:** anything with `STAR`/`bwa`/`samtools sort`/`salmon`, R or Python
analysis, `gzip`/`tar` of big files, `conda create`, building indexes, `apptainer build`.

```bash
# Need a shell to test something interactively?
bsub -P hphr -q normal -n 4 -R "span[hosts=1] rusage[mem=16000]" -W 2:00 -Is /bin/bash

# Downloading many GB? Use the transfers queue (dtn1 has the fast external link)
bsub -P hphr -q transfers -n 2 -W 12:00 -o logs/dl.%J.out ./download.sh
```

### 8.3 Don't waste resources

1. **Right-size memory.** Reserved-but-unused RAM blocks other jobs. See §8.1.
2. **Right-size cores.** `-n 16` on a single-threaded program wastes 15 cores for the whole run.
3. **Pass `$LSB_DJOB_NUMPROC`**, never a hardcoded thread count.
4. **Always set `-W`.** A time-limited job can be backfilled into scheduling gaps and often
   starts *sooner* than one with no limit.
5. **Don't flood the queue.** You had **162 pending jobs** on 2026-11-03. Fairshare
   (`RUN_JOB_FACTOR=3.0`) penalises exactly this. Use job groups (§8.7) to throttle instead.
6. **`bkill` jobs you no longer need** — especially children of a failed dependency, which pend forever.
7. **Test on one sample before launching 24.**
8. **Prefer `normal` when it fits.** It has the most nodes (82) and is never preempted.

### 8.4 Organising a project

The layout you already use is good. Formalised:

```
~/squire_analysis/                      # $HOME — small, versioned, backed up
├── 01_lsf/                             # LSF submission scripts + nextflow configs
│   ├── template.sh
│   ├── pegasus.config
│   └── run_rnaseq.sh
├── scripts/                            # analysis code
├── logs/                               # all -o / -e output
├── samples_prod.tsv                    # sample sheet
├── documentation.md
└── fastq -> /scratch/.../fastq         # symlinks to big data

/scratch/hphr/rna_analysis/             # group scratch — big, fast, shared
├── raw_data/
├── references/
├── apptainer_cache/
├── work/                               # Nextflow work dir
├── squire/
└── results/
```

Conventions that pay off:
- **Symlink** big data into the project dir (you already do this) — short paths, data stays on scratch.
- **Numbered prefixes** (`01_lsf`, `02_qc`) keep steps in order.
- **`%J` in every log name** so nothing is overwritten.
- **Record versions.** You already have `versions.txt` — keep it current:
  `module list 2>&1 | tee -a versions.txt`.
- **Keep the submit script, not the bsub command.** Six months from now, `scripts/submit_prod.sh`
  is the only record of what you actually ran.

### 8.5 Monitoring and optimising

```bash
# Daily: what's mine doing?
bjobs -w
bjobs -p                                 # anything stuck, and why

# After each pipeline step: was the request right?
bhist -l <jobid> | grep -A2 'MEMORY USAGE'

# Across recent jobs — build a table of peak memory
for j in $(bjobs -a -o 'jobid' -noheader 2>/dev/null | head -30); do
  printf "%-8s %s\n" "$j" "$(bhist -l $j 2>/dev/null | grep -o 'MAX MEM: [^;]*')"
done

# My fairshare priority (lower = you'll wait longer)
bqueues -l bigmem | grep -A50 SHARE_INFO_FOR | grep sxt1558

# My job-group throttles
bjgroup -s /sxt1558
```

Keep a small table per pipeline step — requested vs peak vs wall time. After one pass you stop
guessing, your jobs start sooner (smaller requests schedule faster), and you stop blocking others.

### 8.6 Testing safely before a large run

1. **Smoke test.** The 1-minute job from §4.4 — proves LSF, project and queue work.
2. **Shrink the data.** Subsample a FASTQ to 100k reads. You already have `fastq_sub/` for this.
   ```bash
   # 100k reads = 400k lines
   zcat sample_R1.fastq.gz | head -400000 | gzip > test_R1.fastq.gz
   ```
3. **One sample, end to end.** Full pipeline, one sample, generous resources. Fix every error.
4. **Record the real numbers.** `bhist -l` for each step → set resources from measurements.
5. **Two samples in parallel.** Catches race conditions, shared-temp-file bugs and filename collisions.
6. **Then the full run**, with measured resources and a job-group throttle.

For Nextflow specifically:

```bash
# nf-core pipelines ship a tiny test profile — run it first
nextflow run nf-core/rnaseq -r 3.14.0 -profile test,apptainer -c 01_lsf/pegasus.config \
  --outdir /scratch/hphr/rna_analysis/results/nftest

# Then a dry-run of the DAG on your real inputs, without executing anything
nextflow run nf-core/rnaseq ... -preview
```

### 8.7 Job arrays and dependencies

#### Job array — the same work on many inputs, as **one** job

```bash
# samples_prod.tsv has a header + one sample per line
mkdir -p logs

bsub -P hphr -q normal -J "fqc[1-24]" -n 2 \
     -R "span[hosts=1] rusage[mem=4000]" -M 4000 -W 1:00 \
     -o logs/fqc.%J.%I.out -e logs/fqc.%J.%I.err \
     'SAMPLE=$(awk -v i=$LSB_JOBINDEX "NR==i+1 {print \$1}" samples_prod.tsv); \
      echo "index $LSB_JOBINDEX = $SAMPLE"; \
      fastqc -t $LSB_DJOB_NUMPROC -o qc raw_data/${SAMPLE}_R1.fastq.gz'
```

- `$LSB_JOBINDEX` is the array index (1–24); `NR==i+1` skips the header.
- `%I` in the log name gives one log per element.
- ✅ **VERIFIED** — `MAX_JOB_ARRAY_SIZE = 1000`, so `[1-1000]` is the ceiling.
- Throttle concurrency with `%`: `-J "fqc[1-24]%6"` runs at most 6 at a time.

```bash
bjobs -A                       # array summary (one line per array)
bjobs "627900[3]"              # one element
bkill "627900[5]"              # kill one element
bkill 627900                   # kill the whole array
```

#### Dependency — step B waits for step A

```bash
A=$(bsub -P hphr -q normal -J trim -W 2:00 -o logs/trim.%J.out ./trim.sh \
    | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

B=$(bsub -P hphr -q bigmem -J align -n 8 -R "span[hosts=1] rusage[mem=48000]" -M 48000 \
       -W 12:00 -w "done($A)" -o logs/align.%J.out ./align.sh \
    | sed -n 's/.*<\([0-9]*\)>.*/\1/p')

bsub -P hphr -q normal -J count -W 2:00 -w "done($B)" -o logs/count.%J.out ./count.sh
```

Conditions: `done(J)` (exit 0), `ended(J)` (any exit), `exit(J)`, `started(J)`,
`done(J1) && done(J2)`, and `done("jobname")`.
✅ `JOB_DEP_LAST_SUB=1` — a **name**-based dependency matches only the most recently submitted
job with that name, which makes name-based deps safe to reuse.

> ⚠️ **If a parent EXITs, its `done(...)` children pend forever.** Nothing cleans them up. Your
> `scripts/submit_prod.sh` already warns about this. Check with:
> ```bash
> bjobs -p | grep -B1 'dependency condition not satisfied'
> ```
> then `bkill` the orphans.

#### Job groups — throttle yourself

✅ **VERIFIED** — you already use this well. `bjgroup -s /sxt1558` on 2026-11-03:

```
GROUP_NAME            NJOBS  PEND  RUN  JLIMIT
/sxt1558                 17    11    6     6/-
/sxt1558/sq_count        17    11    6     6/8
```

```bash
bgadd -L 8 /sxt1558/mygroup        # at most 8 running at once
bgmod -L 4 /sxt1558/mygroup        # change the limit later
bsub -P hphr -g /sxt1558/mygroup ... ./run.sh
bjgroup -s /sxt1558                # inspect
bkill -g /sxt1558/mygroup 0        # kill everything in the group
```

This is the right way to submit 200 jobs without wrecking your fairshare or monopolising a queue.

---

## 9. Quick reference

### 9.1 The 20 commands you'll actually use

```bash
# ---- Submit -------------------------------------------------------------
bsub -P hphr -q normal -n 4 -R "span[hosts=1] rusage[mem=16000]" -M 16000 \
     -W 2:00 -J myjob -o logs/myjob.%J.out -e logs/myjob.%J.err ./run.sh
bsub < 01_lsf/myjob.sh                  # 2. submit a script with #BSUB directives
bsub -P hphr -q normal -n 4 -R "span[hosts=1] rusage[mem=16000]" -W 2:00 -Is /bin/bash

# ---- Monitor ------------------------------------------------------------
bjobs -w                                # 4. my jobs, full names
bjobs -p                                # 5. pending + WHY  ← first stop when stuck
bjobs -l <jobid>                        # 6. everything about one job
bjobs -A                                # 7. array summary
bpeek -f <jobid>                        # 8. live stdout of a running job
bjgroup -s /sxt1558                     # 9. my job groups and throttles

# ---- Diagnose -----------------------------------------------------------
bhist -l <jobid>                        # 10. finished job: MAX MEM, timings, exit
bhist -l <jobid> | grep -A2 'MEMORY USAGE'   # 11. peak memory  ← use this constantly
busers sxt1558                          # 12. my job counts

# ---- Cluster state -----------------------------------------------------
bqueues -u sxt1558                      # 13. my queues + backlog
bqueues -l bigmem                       # 14. one queue's full config
bhosts bigmem                           # 15. host status / free slots
lshosts n521 n128                       # 16. static hardware (cores, total RAM)
lsload n061                             # 17. live load + FREE memory

# ---- Control ------------------------------------------------------------
bkill <jobid>                           # 18. cancel
bmod -W 48:00 <jobid>                   # 19. change a PENDING job's request
brequeue <jobid>                        # 20. re-run a finished/failed job

# ---- Storage ------------------------------------------------------------
df -h $HOME /scratch/hphr
du -sh /scratch/hphr/rna_analysis/* | sort -rh

# ---- Software -----------------------------------------------------------
module avail 2>&1 | grep -i <tool>
module purge && module load python/3.8.7
```

### 9.2 Workflow: prepare → submit → monitor → troubleshoot → retrieve

```
┌─ 1. PREPARE ───────────────────────────────────────────────────────────┐
│  mkdir -p logs                                                         │
│  Write the script; copy 01_lsf/template.sh                             │
│  set -euo pipefail  +  module purge  +  explicit module load           │
│  Use $LSB_DJOB_NUMPROC for threads                                     │
│  Big data on /scratch/hphr, scripts in $HOME                           │
│  → Test on ONE small sample first                                      │
└────────────────────────────────────────────────────────────────────────┘
                              ▼
┌─ 2. SUBMIT ────────────────────────────────────────────────────────────┐
│  Check availability:  bqueues -u sxt1558 ; bhosts <queue>              │
│  bsub < 01_lsf/myjob.sh          → note the job ID                     │
│  -P hphr is MANDATORY. -W always. rusage[mem=] always.                 │
└────────────────────────────────────────────────────────────────────────┘
                              ▼
┌─ 3. MONITOR ───────────────────────────────────────────────────────────┐
│  bjobs -w              PEND? → bjobs -p for the reason                 │
│  bpeek -f <jobid>      watch live output                               │
│  bjobs -l <jobid>      memory so far, exec host                        │
└────────────────────────────────────────────────────────────────────────┘
                              ▼
┌─ 4. TROUBLESHOOT ──────────────────────────────────────────────────────┐
│  cat logs/myjob.<jobid>.err        ← read stderr, properly             │
│  bhist -l <jobid>                  ← exit reason + MAX MEM             │
│  Truncated log / exit 130,137 → OOM. Raise rusage+M, use bigmem        │
│  exit 140 → runlimit. Raise -W                                         │
│  "command not found" → missing module load in the script               │
│  No output but DONE → you forgot set -euo pipefail                     │
└────────────────────────────────────────────────────────────────────────┘
                              ▼
┌─ 5. RETRIEVE & OPTIMISE ───────────────────────────────────────────────┐
│  Results → /scratch/hphr/.../results ; copy small ones to $HOME        │
│  bhist -l <jobid> | grep -A2 'MEMORY USAGE'                            │
│  → record peak; request peak × 1.3 next time                           │
│  Clean up:  nextflow clean -f -before <run> ; rm intermediate BAMs     │
└────────────────────────────────────────────────────────────────────────┘
```

### 9.3 Minimal reusable template

Full annotated version in §4.5. The short one:

```bash
#!/bin/bash
#BSUB -P hphr                              # MANDATORY
#BSUB -q normal                            # normal | bigmem | sccc | sccc_bigmem
#BSUB -J myjob
#BSUB -n 4                                 # slots (cores); max 16 on normal
#BSUB -R "span[hosts=1] rusage[mem=16000]" # one node; 16000 MB for the WHOLE job
#BSUB -M 16000                             # MB (advisory here)
#BSUB -W 4:00                              # wall-clock hh:mm
#BSUB -o logs/myjob.%J.out                 # logs/ must already exist
#BSUB -e logs/myjob.%J.err
set -euo pipefail

export TMPDIR=/tmp/$USER.$LSB_JOBID
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT

module purge
module load python/3.8.7

echo "host=$(hostname) slots=$LSB_DJOB_NUMPROC started=$(date)"

# ... your work, using "$LSB_DJOB_NUMPROC" for threads ...

echo "done=$(date)  check: bhist -l $LSB_JOBID | grep -A2 'MEMORY USAGE'"
```

Submit with `bsub < thisfile.sh`.

### 9.4 Queue decision table

| Cores | Memory | Runtime | → Queue | Preemptable? |
|---|---|---|---|---|
| ≤16 | ≤28 GB | any | **`normal`** | No |
| ≤16 | 28–55 GB | any | `normal` (62 GB nodes only) or **`bigmem`** | No |
| ≤16 | 55–240 GB | ≤240 h | **`bigmem`** | No |
| ≤16 | ≤55 GB | >240 h | **`normal`** (no run limit) | No |
| 17–64 | ≤490 GB | any | **`sccc`** | ⚠️ Yes |
| ≤64 | 240 GB – 3.8 TB | any | **`sccc_bigmem`** | ⚠️ Yes |
| any | any | any, needs GPU | **`sccc_gpu`** or `gpu_h100` | ⚠️ Yes (`sccc_gpu`) |
| 17–512 across nodes (MPI) | — | any | **`parallel`** (`ptile=16`) | No |
| ≤4 | — | data transfer | **`transfers`** (`dtn1`) | No |
| 1–2 | ≤8 GB | days | **`normal`** (Nextflow driver) | No |

**Then check it's actually free** — on 2026-11-03 `bigmem` was 100% full with 262 jobs pending
while `sccc` had ~126 of 128 slots idle:

```bash
bhosts <queue> | awk 'NR>1 && $2=="ok" {f+=$4-$5; t+=$4} END{print f" free of "t}'
```

### 9.5 Five mistakes to avoid

1. **Forgetting `-P hphr`.** The job is rejected outright — `esub.project` has no fallback.
   This is Pegasus-specific and catches everyone once, including every default Nextflow config
   (set `clusterOptions = '-P hphr'`).

2. **Assuming memory is in GB.** `LSF_UNIT_FOR_LIMITS=MB`. `-M 16` means **16 megabytes**, not
   16 GB. Write `-M 16000`.

3. **Thinking `rusage[mem=]` is per-core.** `RESOURCE_RESERVE_PER_TASK=N` here, so it is
   **per job**. `-n 8 -R "rusage[mem=32000]"` reserves 32 GB total, not 256 GB. (And Nextflow's
   default halves it the other way — set `executor.perJobMemLimit = true`.)

4. **Omitting `set -euo pipefail`.** A script whose steps fail still exits 0, LSF says `DONE`,
   and you spend an hour wondering where your output went.

5. **Running analyses on `login1`, or hardcoding thread counts.** Both are antisocial and both
   get noticed. `login1`'s 16 cores are shared by everyone; and 16 threads inside a `-n 4` cgroup
   is *slower* than 4. Use `bsub`, and pass `$LSB_DJOB_NUMPROC`.

---

## Appendix A — Verified cluster configuration

Everything below was read directly from this cluster on **2026-11-03**.

### LSF settings that change how you write `bsub` lines

| Setting | Value | Source | Why it matters |
|---|---|---|---|
| `LSF_UNIT_FOR_LIMITS` | **`MB`** | `lsf.conf` | `-M` and `rusage[mem=]` are megabytes |
| `LSF_ENFORCE_MEMLIMIT` | **`N`** | `lsf.conf` | LSF won't kill you for exceeding `-M`; the OOM killer will |
| `LSB_RESOURCE_ENFORCE` | `"cpu gpu"` | `lsf.conf` | CPU/GPU enforced by cgroups; memory is not |
| `RESOURCE_RESERVE_PER_TASK` | **`N`** | `bparams -a` | `rusage[mem=]` is **per job**, not per slot |
| `PARALLEL_SCHED_BY_SLOT` | `y` | `lsb.params` | `-n` means slots/cores |
| `ABS_RUNLIMIT` | `Y` | `lsb.params` | `-W` is wall-clock |
| `LSF_STRICT_RESREQ` | `Y` | `lsf.conf` | Impossible requests pend silently forever |
| `LSB_ESUB_METHOD` | `"project"` | `lsf.conf` | **`-P` is mandatory** |
| `MAX_JOB_ARRAY_SIZE` | `1000` | `bparams -a` | Array ceiling |
| `CLEAN_PERIOD` | `3600` | `bparams -a` | Finished jobs leave `bjobs` after 1 h → use `bhist` |
| `DEFAULT_QUEUE` | `normal interactive` | `lsb.params` | Queue if you omit `-q` |
| `JOB_DEP_LAST_SUB` | `1` | `lsb.params` | Name-based deps match the newest job |
| `MAX_PEND_JOBS` | `2147483647` | `bparams -a` | No cap on queued jobs (fairshare throttles instead) |
| `JOB_SPOOL_DIR` | `/scratch/lsf_tmp` | `bparams -a` | LSF's own spool — not for your files |
| `LSB_GPU_NEW_SYNTAX` | `extend` | `lsf.conf` | Use `-gpu "num=1"` syntax |
| `CPU_TIME_FACTOR` / `RUN_TIME_FACTOR` / `RUN_JOB_FACTOR` | `0.7` / `0.7` / `3.0` | `bparams -a` | Fairshare: recent usage lowers your priority |
| `PREEMPTION_WAIT_TIME` | `300` s | `bparams -a` | Preemption timing on `sccc*` |
| `MAX_JOBID` | `999999` | `bparams -a` | Job IDs wrap |

**No** `DEFAULT_RUNLIMIT`, **no** `MEMLIMIT` on any queue, and `blimits` reports **no per-user limits**.

### Commands used for this inspection

```bash
lsid ; lsclusters ; bparams -a
bqueues ; bqueues -u sxt1558 ; bqueues -l <queue>
bhosts <queue> ; bhosts -l <host> ; lshosts ; lsload ; lsload -gpu
bmgroup -w ; busers ; blimits ; bjgroup -s ; bapp
bjobs -w ; bjobs -p ; bjobs -l <id> ; bhist -l <id>
df -hT ; du -sh ; mmlsquota ; ls -ld ; getfacl
module avail ; module show <mod> ; command -v <tool>
grep -v '^#' /opt/ibm/lsfsuite/lsf/conf/lsf.conf
cat /opt/ibm/lsfsuite/lsf/conf/lsbatch/pegasus/configdir/{lsb.params,lsb.queues}
cat $LSF_SERVERDIR/etc/esub.project
egrep '\bsxt1558\b' /opt/ibm/lsfsuite/lsf/conf/lsf.projects
```

No jobs were submitted or cancelled; no files, permissions or configurations were modified.

### Things to confirm with HPC support

<https://hpc.idsc.miami.edu>

1. ⚠️ **Scratch retention policy** — are files on `/scratch` or `/scratch/hphr` auto-deleted, and
   after how long? No policy document was readable from the CLI. **This is the most important one**
   — plan your storage around the answer.
2. ⚠️ **Official quotas** — `mmlsquota` fails with *"No quota enabled file system found"*.
   Are the `df` figures (250 GB home, 2.0 TB `/scratch/hphr`) the real limits?
3. ⚠️ **`gpu_h100` hardware** — `lshosts` reports `-` for cores and RAM on `gpu1`/`gpu2`;
   how many GPUs per node, and how should they be requested?
4. ⚠️ **GPU request syntax on `sccc_gpu`** — `lsb.queues` has the `GPU_REQ` line **commented out**
   with the note *"Pedro 2026/09/30: LSF config is messed up, share GPU jobs broken, so disabling
   default to request a shared GPU"*. Ask what `-gpu` options currently work.
5. ⚠️ **`sccc*` preemption behaviour** — are preempted jobs **suspended** (LSF default, which is
   what the config implies) or **killed**? This decides whether `sccc` is safe for long runs.
6. ⚠️ **Access to `sccc_premium` / `sccc_bigmem_premium`** — worth asking whether group `hphr`
   can be added, since the non-premium equivalents are preemptable.
7. ⚠️ **Compute-node `/tmp` size** — 460 GB on `login1`, unknown on compute nodes.
8. ⚠️ **`/sccc` is 100% full** (2.4 PB of 2.4 PB) — affects the `igenomes` module, which lives
   at `/sccc/share/apps/igenomes`.
