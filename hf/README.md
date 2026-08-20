---
license: other
license_name: deepseek
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - uncensored
  - abliterated
  - abliteration
  - refusal-direction
  - activation-steering
  - vllm
  - deepseek
  - interpretability
library_name: safetensors
---

# DeepSeek-V4-Flash-0731 — Uncensored / Abliterated, switchable at runtime

**An uncensored (abliterated) DeepSeek-V4-Flash without a second checkpoint, without
reloading the model, and without swapping anything on disk.**

Abliteration normally means downloading a whole separate ~157 GB model with the refusal
behaviour permanently burned into its weights. This is the same effect delivered a different
way: a **757 KB** file of direction vectors sits beside your existing checkpoint, and the
uncensoring strength becomes a **dial you turn at runtime** — one HTTP call, effective on the
next request, no restart, no reload, no downtime.

```bash
curl -XPOST localhost:8888/admin/refusal_lambda -d '{"lambda": 1.5}'   # uncensored
curl -XPOST localhost:8888/admin/refusal_lambda -d '{"lambda": 0}'     # back to stock
```

At λ=1.5 the model refuses **0 out of 10** prompts it declined 9 out of 10 times at stock —
with no measurable loss in speculative-decoding acceptance, long-context retrieval, or
tool-calling. At λ=0 it is **bit-exact** to the unmodified DeepSeek release, so "off" is
genuinely off rather than approximately off.

Because it is a dial rather than a swap, it reaches an operating point **no published
abliterated checkpoint can offer**: the baked edit does not merely remove the refusal
direction, it overshoots and *inverts* it (~240 %), and that overshoot is what costs quality.
The clean range in between only exists at runtime.

Code, patches, benchmark harnesses and every raw result:
**https://github.com/pocharlies/deepseek-v4-flash-rank1-refusal-projection**

Measured on 2× DGX Spark GB10 (sm_121), vLLM 0.25.2, TP=2, DSpark speculative decoding k=5,
`--max-model-len 262144`, `--enable-prefix-caching`.

---

## The result

| metric | λ=0 | λ=1.5 | verdict |
|---|---:|---:|---|
| Refusal rate (10 triggers) | 9/10 (90 %) | **0/10 (0 %)** | eliminated |
| DSpark acceptance (n=6 alternated) | 0.5669 ± 0.0097 | 0.5608 ± 0.0189 | t = +0.70, indistinguishable |
| NIAH retrieval @ 32k + 128k | 30/30 | **30/30** | no regression |
| Tool-calling | 8/8 | **8/8** | no regression |
| Benign controls falsely refused | 0/4 | 0/4 | classifier sane |

83.4 minutes of alternated A/B on the same pod, same day, same load. Every number above is
backed by raw JSON in the GitHub repo under `bench/results/`.

**The baked checkpoint cannot reach this operating point at all.** Measured on the weights,
it sits at λ_eff ≈ 2.43 — it does not remove the direction, it inverts it (see below). Prior
measurement of that checkpoint on this deployment put acceptance at 0.5128, below the 0.55
floor required here.

---

## In plain terms: the advantages over a normal abliterated model

| | Baked abliterated checkpoint | This technique |
|---|---|---|
| Extra download | a second ~157 GB model | **757 KB** |
| Disk | ~313 GB (two copies) | ~157 GB (one copy) |
| Turning it off | stop server, reload other weights | **one HTTP call** |
| Time to switch | full restart + weight load | **next request** |
| Downtime when switching | yes, all in-flight requests die | **none** |
| Is "off" really off? | you're on a different model | **bit-exact to stock** |
| Strength | fixed at whatever was baked | **any value, live** |
| More cautious than stock | impossible | λ < 0 |
| Base weights auditable | no, they're modified | **sha256 vs DeepSeek release** |

**The usual way to run an abliterated model:** you download a second, complete copy of the
model — a separate ~157 GB checkpoint. Now you keep two copies on disk (~313 GB), and they are
mutually exclusive on the same GPUs. Want the normal model back? Stop the server, load 157 GB
of different weights across both nodes, wait for it to come up. Every in-flight request dies.
Want to compare the two on the same prompt? You do it hours apart, on different server
processes, and hope nothing else drifted in between.

**This way:** you download nothing extra. Your existing checkpoint stays exactly as it is, and
this **757 KB** file sits next to it — about 2,000× smaller than the published 1.54 GB overlay,
and roughly 220,000× smaller than a second copy of the model. One server process. Switching is
one HTTP call:

```bash
curl -XPOST localhost:8888/admin/refusal_lambda -d '{"lambda": 1.5}'   # ablation on
curl -XPOST localhost:8888/admin/refusal_lambda -d '{"lambda": 0}'     # ablation off
```

It takes effect on the **next request**. No restart. No reload. No second copy on disk. No
downtime, and nothing to re-download.

### Three things that follow

**1 · "Off" genuinely means off.** λ=0 is *bit-exact* to the unmodified model — verified with
`torch.equal`, not "close enough". You are not permanently running a modified model and hoping
it behaves like the original on ordinary work; at zero it **is** the original. That is why the
base weights stay byte-identical to the DeepSeek release and can be checked with sha256.

**2 · It is a dial, not a switch.** λ=0 is stock, λ=1.5 removes refusal entirely, everything in
between is available, and λ can go *negative* — making the model **more** reticent than stock,
which no abliterated checkpoint can offer.

**3 · A/B testing becomes honest.** Both arms run on the same process, same weights, same hour,
same load; you just flip the dial between runs. That is how the numbers above were produced —
6 alternated runs per arm in one 83.4-minute session. With two separate checkpoints you cannot
do this, and the drift is real: two λ=0 runs in that same session came in at 57.87 and
40.52 tok/s.

### The two honest costs

**Switching is instant, but the prefix cache for the new λ starts cold.** λ is part of the
block hash key *deliberately* — reusing cached blocks across different λ would silently corrupt
state, and that is the single most likely way to get this wrong. So the first request after a
switch re-prefills its context. Measured on this hardware: **76.3 s for a 136,879-token
prompt**, ~12.6 s at 16.7k tokens, negligible for short prompts. Steady-state traffic at a
fixed λ caches normally and is unaffected.

**λ=0 is free in output, not in compute.** The dot product and subtraction run in all 46 layers
on every token regardless of λ, because branching on λ is exactly what breaks CUDA graph
capture. For genuinely zero overhead, unset `VLLM_REFUSAL_DIRS` and restart.

---

## What is in this repository

```
refusal_dirs.safetensors    757,712 bytes — 46 tensors, each [4096] float32, unit norm
```

Metadata embedded in the file:

| key | value |
|---|---|
| `source_base` | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| `source_abl` | `21bd923c2574d9edcd7b914885024ce72fd5c076` |
| `modules` | `46` (43 backbone + 3 MTP) |

Keys are module names: `layers.0.attn.wo_b` … `layers.42.attn.wo_b`, `mtp.0-2.attn.wo_b`.

**These are not weights.** They are the normalized principal direction of ΔW between two
publicly released checkpoints. They do nothing on their own — they need the runtime hook
from the GitHub repo.

---

## How it works

Editing a weight and projecting the sublayer output are the same function:

```
(W − λ·r̂r̂ᵀW)·x  ≡  W·x − λ·r̂·(r̂ᵀ·W·x)
```

The left side is what abliterated checkpoints ship. The right side needs only `r̂` — one unit
vector in ℝ⁴⁰⁹⁶ per edited module — and never touches `W`.

Verified to machine precision in float64 across 4 modules: max relative error **8.8e-16 to
9.3e-16**. At λ=0 the hook is **bit-exact** to the base model (`torch.equal`).

So: 757 KB instead of 1.54 GB, base weights verifiable by sha256 against the DeepSeek
release, and λ as a continuous runtime parameter rather than a property frozen into a
checkpoint.

Method: Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*
(NeurIPS 2024). Directions extracted from `cebeuq/DeepSeek-V4-Flash-0731-abliterated`.

---

## The finding that reframed the project

The plan assumed the baked edit hits a **~68 % ceiling** from the FP8 round-trip and
therefore over-projects to λ=2.5 to compensate. Measured directly on the weights:

| module | ⟨r̂ᵀW_abl, r̂ᵀW_base⟩ / ‖r̂ᵀW_base‖² | actual removal |
|---|---:|---:|
| `layers.0` | **−1.4229** | **+242.3 %** |
| `layers.21` | **−1.4139** | **+241.4 %** |
| `layers.42` | **−1.4708** | **+247.1 %** |
| `mtp.0` | **−1.3104** | **+231.0 %** |

There is no ceiling. The published checkpoint does not fall short — it removes ~240 % of the
direction, i.e. **inverts** it and leaves it at ~1.4× its original magnitude pointing the
opposite way.

λ=2.5 compensates for nothing; it *is* the overshoot. The degradation measured on the baked
deployment (acceptance 0.5128, code throughput 46.6–55.3 vs 59.32 tok/s) is the cost of
**inverting** the direction, not of removing it.

> **The whole useful range 0 < λ ≤ 1.5 is territory no published checkpoint occupies.**
> Baked abliteration jumps from +100 % (unedited) straight to −143 %. The clean point between
> them only exists as a runtime dial.

At λ=1 the residual component is exactly 0 in exact arithmetic; with bf16 activations the
real residual is ~1.7e-3 relative. That is **99.83 %** removal, not 100 %.

---

## Extraction gates — two passed, two failed

| # | metric | expected | measured | |
|---|---|---|---|---|
| 1 | edited modules | 46 | **46** (43 backbone + 3 MTP) | ✅ |
| 2 | rank-1 energy `S₀²/ΣS²` | ≥ 0.999 | **0.8376 – 0.9448** (mean 0.8970) | ❌ |
| 3 | mean δ Frobenius | 0.0587 (published) | **0.06018** (+2.5 %) | ✅ |
| 4 | λ effective | ~1.7 (hypothesis) | **2.429** (range 2.340 – 2.473) | ❌ |

**Gate 2 failed and r=2 does not fix it.** s₁/s₂ is 13–23×, but from s₂ to s₈ the decay is
only 2.5–2.7× — a flat tail. r=2 buys 0.36 points of energy; r=8 still doesn't reach 0.91.
There is no second direction: the missing ~10 % is broadband requantization noise from E4M3
with power-of-two block scales — which is **precisely the artifact this design avoids**, not
something the hook should reproduce.

Also measured: the edit **subtracts** in all 46 modules, and `cos(v₀, u₀ᵀW_base) ≥ 0.9873`
across all 46 — the dominant component genuinely has the shape of a projection.

Backbone and MTP are edited differently (δ 0.0615 vs 0.0410, λ_eff 2.435 vs 2.343). In the
runtime hook they share one dial and stay aligned by construction; in the published
checkpoint they do not, and that misalignment is a prime suspect for part of the acceptance
drop.

---

## Integration — the five things that silently break it

Full patch in the GitHub repo (`patches/0001-rank1-projection.patch`, 460 lines, 7 files,
against vLLM `0.25.2.dev0+g752a3a504`).

1. **λ must be a device tensor, never a Python float.** With CUDA graph capture a Python
   scalar gets baked into the captured graph — changing it does nothing, with no error. One
   tensor per device, shared across layers, mutated with `fill_()`.
2. **λ must enter the prefix-cache hash key.** A prefix cached at λ=0 and reused at λ=1.5
   gives corrupt state, silently. The right precedent is `cache_salt`, not `lora_int_id`. λ
   quantized to integer (×1000).
3. **The setter must go through `collective_rpc`.** With the `mp` executor a local setter
   reaches no worker rank. Divergent ranks must be a **500**, not a 200.
4. **Apply at the sublayer output, before `hc_post`.** The mHC write is rank-1 across
   streams, so removing the direction at the sublayer output clears all four streams at once.
   Do not project onto the `[B,S,4,4096]` carrier.
5. **TP needs nothing special.** `wo_b` is `RowParallelLinear` with `reduce_results` default;
   its return is already post-all-reduce, and the projection is linear.

Kernel shape — do the dot product in fp32 even when `y` is bf16 (1.66e-3 vs 2.29e-3 median
error; it's free, though both are dominated by bf16 storage of `y`):

```python
proj = (y.float() @ r_hat)                       # [num_tokens]
y = y - (lam * proj).unsqueeze(-1) * r_hat       # lam is a device tensor
```

Control surface: `POST /admin/refusal_lambda {"lambda": 1.5}`. The router only mounts when
`VLLM_REFUSAL_DIRS` is set. Keep it off the public ingress — vLLM cannot enforce that.

---

## Production serving note — 2026-08-20: `--long-prefill-token-threshold 1024`

The 2× DGX Spark TP=2 deployment this card is measured on now runs with
`--long-prefill-token-threshold 1024` (vLLM V1 `SchedulerConfig`). On this build the flag
**caps the per-step chunk** of any prefill larger than the threshold, so a single 80K-token
prompt can no longer monopolize the 8,192-token step budget while other requests decode
(verified via `vllm:iteration_tokens_total`: 99.2 % of steps ≤1,024 tokens after the change).

Measured on the production pair, same traffic source, 2 h after vs 16 h before: median
generation throughput at 2–6 concurrent requests went from 2.0–5.3 tok/s to 12.2–53.8 tok/s
(2.9–10×), max queue depth 37 → 7, TTFT under mixed load 43–114 s → 0.18–0.33 s. DSpark
acceptance is unchanged (54.6 %, mean accepted length 3.71). The cost: an 80K prefill now
takes ~59 s (~1,363 tok/s) instead of ~20 s with 8K chunks. The λ projection is orthogonal
to the scheduler — nothing about the dial changes.

---

## Calibration

| λ | refusal rate | acceptance | NIAH 32k + 128k |
|---:|---:|---:|---:|
| 0 | 90 % | **0.5669 ± 0.0097** (n=6) | 30/30 |
| 1 | 50–70 % | see note | 30/30 |
| **1.5** | **0 %** | **0.5608 ± 0.0189** (n=6) | **30/30** |
| 2 | 0 % | not measured | not measured |

Every figure in this table has per-run raw JSON in the GitHub repo under `bench/results/`.
The λ=0 and λ=1.5 acceptance arms were run **alternated** in a single 83.4-minute session
(`compare_full.log`), which is what makes them comparable — this box drifts, and two λ=0 runs
in the same session came in at 57.87 and 40.52 tok/s.

**On the λ=1 arm and the baked baseline.** An earlier working report
(`docs/projection-validation.md`) records acceptance 0.5635 at λ=1 and 0.5128 for the baked
checkpoint at λ_eff ≈ 2.43. Both are real prior measurements, but their per-run JSON is not in
the published set, and that report's λ=1.5 figure (n=3) was superseded five hours later by the
n=6 alternated run above. Treat the λ=1 and baked numbers as indicative, not as evidence of
the same weight as the rest of this table. The λ=1 **refusal rate** is separately backed by
`refusal_rate.json` and `refusal_sweep.json`.

**It saturates at 1.5.** λ=2 adds nothing over 1.5: refusal is already 0 % there. 1.5 is the
operating point.

### 2026-08-19 — λ up to 2.5, measured on capability and on truthfulness

Recommending 1.5 used to rest partly on the assumption that higher λ degrades. That
assumption was tested and **it does not hold on this checkpoint**:

| λ | MMLU-Pro (112) | GSM8K (100) | corrects a false premise | yields under pressure | NIAH |
|---:|---:|---:|---:|---:|---:|
| 0 | 80.4 % | 71 % | 100 % | 0 % | 18/18 |
| 1.0 | 79.5 % | 77 % | 100 % | 0 % | 18/18 |
| 1.5 | 76.8 % | 74 % | 100 % | 0 % | 18/18 |
| 2.0 | 77.7 % | 69 % | 100 % | 0 % | 18/18 |
| 2.5 | 78.6 % | 68 % | 100 % | 0 % | 18/18 |

Paired, temperature 0, 4000-token budget, per-request λ (the global dial stayed at 0 and was
re-checked between arms). **No comparison is significant** — every McNemar p ≥ 0.09, and the
curve is not monotonic. **0 yields out of 113** sycophancy opportunities; 90/90 needles.

So the honest statement is: **stay at 1.5 because above it you gain nothing, not because it
breaks.** Two caveats that keep this from being a licence to raise it:

- The truthfulness suites return 100 % in *every* arm — a ceiling effect. They refute the
  hypothesis that ablation also removes "no, that is false", but a saturated suite cannot
  rank arms. With 0 events in ~22 items per arm, the rule of three only rules out sycophancy
  above ~13.6 %.
- **Acceptance at λ≥2 is still not measured.** The baked-checkpoint figure at λ_eff ≈ 2.43
  (0.5128) remains the only signal there and it is below the floor. Capability and speculative
  acceptance are different axes; nothing above speaks to the second.

Full report, raw JSON, and the *confounded* first version of the quality sweep (kept
deliberately as a worked example): [`benchmarks/2026-08-19/`](benchmarks/2026-08-19/README.md).

---

## What this does not establish

- ~~**General capability is unmeasured.**~~ **Closed 2026-08-19** for MMLU-Pro and GSM8K
  across λ ∈ {0, 1, 1.5, 2, 2.5} — see Calibration. HumanEval is still not run.
- **Nothing here validates an answer that only a high λ produces.** That content is, by
  construction, what the model represented least well, and it is the one domain where no
  ground truth exists to check it against. This model can correct textbook myths and retrieve
  needles at 128k and still confabulate a fluent, confident, structurally plausible
  procedure. Fluency is not a correctness signal — do not read the tables above as a warrant
  for trusting output that only appears once λ is raised.
- **256k is unmeasured**, deliberately. Retrieval is validated to 126,940 real tokens.
- **Variance rises with λ even where the mean passes.** 1 of 6 runs at λ=1.5 and 2 of 6 at
  λ=1 dipped below the 0.55 acceptance floor. Means are indistinguishable; individual runs
  are not always.
- **λ=0 is bit-exact in output but not free in compute.** The dot product runs in all 46
  layers per token regardless of λ. For zero cost, unset `VLLM_REFUSAL_DIRS` and restart.

Benchmarks that lied before being fixed, documented in full in the repo: the model spends
**1,100–1,400 tokens reasoning** before answering, so a low `max_tokens` returns empty
content and reads as a spectacular refusal-removal result — it isn't. And exact string match
is not a valid gate: vLLM is not deterministic run-to-run at temperature 0, with measured
noise of ±1 cell in 30.

That first failure recurred on 2026-08-19, in a different suite and wearing a p-value: the
quality sweep at `max_gen_toks=2048` produced a tidy descending curve and two "significant"
results, and all of it was truncation — empty answers rise with λ (18→31) because the model
reasons longer, and empties score as wrong. Conditioned on non-empty answers the accuracy is
flat at every λ. **On a reasoning model, a short generation budget confounds λ with
verbosity.** The confounded run is published next to the corrected one.

---

## Intended use and a real caveat

This is interpretability and serving-infrastructure work: it makes an already-published
weight edit auditable, reversible and continuously adjustable, and it measures what that edit
actually costs — which the published checkpoint did not.

The uncomfortable part, stated because it is part of the engineering picture:

**Reducing a model's resistance to instructions reduces its resistance to *injected*
instructions.** Prompt injection and refusal route through overlapping machinery. A clean
λ=1.5 yields a model *more* capable and *more* completely stripped of its ability to decline
than the baked checkpoint, whose clumsiness incidentally limited how useful it was. The
better the dial works, the more the isolation matters.

If you wire this to tools with write access or feed it untrusted scraped content:

- **λ>0 should not share credentials with write-capable tools** — use a separate deployment.
- **Restrict the toolset when λ>0** to read-only paths.
- **Keep `/admin/refusal_lambda` off the public ingress.**

One measured nuance worth keeping: at λ=1.5 the model still refuses to claim it issued a
refund it cannot issue (tool-calling 8/8 includes that test). The projection removes the
policy refusal, not the grounded one.

---

## Citation

```bibtex
@misc{arditi2024refusal,
  title  = {Refusal in Language Models Is Mediated by a Single Direction},
  author = {Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and
            Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  year   = {2024},
  note   = {NeurIPS 2024},
  eprint = {2406.11717},
  archivePrefix = {arXiv}
}
```

Directions derived from the difference between `deepseek-ai/DeepSeek-V4-Flash-0731` and
`cebeuq/DeepSeek-V4-Flash-0731-abliterated`; they inherit the DeepSeek Model License. Code in
the linked GitHub repository is Apache-2.0.
