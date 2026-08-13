# Runtime rank-1 refusal projection for DeepSeek-V4-Flash-0731

Serving an abliterated model **without shipping abliterated weights**: 757 KB of direction
vectors, base checkpoint byte-identical to the DeepSeek release, and the ablation strength
λ as a hot-swappable dial with no restart.

Measured on 2× DGX Spark GB10 (sm_121, 128 GB unified each), vLLM 0.25.2, TP=2, DSpark
speculative decoding k=5, `--max-model-len 262144`.

The direction vectors and model card are published to the Hugging Face Hub from
[`hf/`](hf/). Published at [`pocharlies/deepseek-v4-flash-0731-uncensored-abliterated-refusal-directions`](https://huggingface.co/pocharlies/deepseek-v4-flash-0731-uncensored-abliterated-refusal-directions).

**Headline result — λ=1.5 removes refusal completely at no measurable cost:**

| metric | λ=0 | λ=1.5 | verdict |
|---|---:|---:|---|
| Refusal rate (10 triggers) | 9/10 (90 %) | **0/10 (0 %)** | eliminated |
| DSpark acceptance (n=6 alternated) | 0.5669 ± 0.0097 | 0.5608 ± 0.0189 | t = +0.70, indistinguishable |
| NIAH retrieval @ 32k + 128k | 30/30 | **30/30** | no regression |
| Tool-calling | 8/8 | **8/8** | no regression |
| Benign controls falsely refused | 0/4 | 0/4 | classifier sane |

Raw output: [`bench/results/compare_full.log`](bench/results/compare_full.log),
[`compare_full.json`](bench/results/compare_full.json),
[`refusal_sweep.json`](bench/results/refusal_sweep.json). 83.4 min of alternated A/B.

The baked checkpoint cannot reach this operating point at all. Measured on the weights it sits
at λ_eff ≈ 2.43, where it does not remove the direction but **inverts** it (§2). Prior
measurement of that checkpoint on this deployment put acceptance at 0.5128, below the 0.55
floor required here — that figure comes from [`docs/projection-validation.md`](docs/projection-validation.md),
not from the raw runs in `bench/results/`.

---

## 0. In plain terms: what this actually buys you

**The usual way to run an abliterated model:** you download a second, complete copy of the
model — a separate ~157 GB checkpoint. Now you have two copies on disk (~313 GB), and they
are mutually exclusive on the same GPUs. Want the normal model back? Stop the server, load
157 GB of different weights across both nodes, wait for it to come up. Every in-flight request
dies. Want to compare the two on the same prompt? You do it hours apart, on different server
processes, and hope nothing else drifted in between.

**This way:** you download nothing extra. One checkpoint, plus a **757 KB** file next to it —
about 2,000× smaller than the published 1.54 GB overlay, and ~220,000× smaller than a second
copy of the model. One server process. Switching is one HTTP call:

```bash
curl -XPOST localhost:8888/admin/refusal_lambda -d '{"lambda": 1.5}'   # ablation on
curl -XPOST localhost:8888/admin/refusal_lambda -d '{"lambda": 0}'     # ablation off
```

It takes effect on the **next request**. No restart. No reload. No second copy on disk. No
downtime, and nothing to re-download.

### Four things that follow from that

**1 · "Off" genuinely means off.** λ=0 is *bit-exact* to the unmodified model — verified with
`torch.equal`, not "close enough". You are not permanently running a modified model and hoping
it behaves like the original on normal work; when the dial is at zero it **is** the original.
That is why the base weights can stay byte-identical to the DeepSeek release and be checked
with sha256.

**2 · It's a dial, not a switch.** λ=0 is stock, λ=1.5 removes refusal entirely, and every
value in between is available. λ can even go *negative*, which makes the model **more**
reticent than stock — the one thing no abliterated checkpoint can offer.

**3 · A/B testing becomes honest.** Both arms run on the same process, same weights, same
hour, same load — you flip the dial between runs. That is exactly how the measurements in this
repo were produced (6 alternated runs per arm in one 83.4-minute session). With two separate
checkpoints you cannot do this, and the drift shows: two λ=0 runs in that same session came in
at 57.87 and 40.52 tok/s.

**4 · You can route per request.** With `patches/0002` λ becomes a per-request parameter, so
one deployment can serve ablated and non-ablated traffic simultaneously — which is what makes
the isolation in §10 enforceable rather than aspirational.

### The two honest costs

**Switching is instant, but the prefix cache for the new λ starts cold.** λ is part of the
block hash key *on purpose* — reusing cached blocks across different λ would silently corrupt
state, and that is the number-one way this kind of setup breaks. So the first request after a
switch re-prefills its context. Measured on this hardware: **76.3 s for a 136,879-token
prompt**, ~12.6 s at 16.7k, negligible for short prompts. Steady-state traffic at a fixed λ is
unaffected — it caches normally.

**λ=0 is free in output, not in compute.** The dot product and subtraction run in all 46 layers
on every token whatever λ is, because branching on λ is precisely what breaks CUDA graph
capture. For genuinely zero overhead, unset `VLLM_REFUSAL_DIRS` and restart.

---

## 1. The identity everything rests on

Editing a weight and projecting the sublayer output are the same function:

```
(W − λ·r̂r̂ᵀW)·x  ≡  W·x − λ·r̂·(r̂ᵀ·W·x)
```

Left side is what published "abliterated" checkpoints ship. Right side needs only `r̂` —
one unit vector in ℝ⁴⁰⁹⁶ per edited module — and leaves `W` untouched.

Verified to machine precision in float64 across 4 modules
([`docs/verify-projection.json`](docs/verify-projection.json)):

| module | max relative error |
|---|---:|
| `layers.0.attn.wo_b` | 8.789e-16 |
| `layers.21.attn.wo_b` | 8.671e-16 |
| `layers.42.attn.wo_b` | 9.162e-16 |
| `mtp.0.attn.wo_b` | 9.262e-16 |

Consequences: 757 KB instead of a 1.54 GB overlay, base weights verifiable by sha256
against the DeepSeek release, and λ becomes a continuous runtime parameter instead of a
property baked into a checkpoint.

Method origin: Arditi et al., *Refusal in Language Models Is Mediated by a Single
Direction*, NeurIPS 2024. Reference baked checkpoint:
`cebeuq/DeepSeek-V4-Flash-0731-abliterated`.

---

## 2. The premise this project started from was wrong

The original plan assumed the baked edit hits a **~68 % ceiling** because of the FP8
round-trip, and therefore over-projects to λ=2.5 to compensate. Gate G3 measured it
directly on the weights, no sampling involved:

| module | ⟨r̂ᵀW_abl, r̂ᵀW_base⟩ / ‖r̂ᵀW_base‖² | actual removal |
|---|---:|---:|
| `layers.0` | **−1.4229** | **+242.3 %** |
| `layers.21` | **−1.4139** | **+241.4 %** |
| `layers.42` | **−1.4708** | **+247.1 %** |
| `mtp.0` | **−1.3104** | **+231.0 %** |

There is no ceiling. The published checkpoint does not fall short of removing the
direction — it removes ~240 % of it, i.e. **inverts** the direction and leaves it at ~1.4×
its original magnitude pointing the other way.

So λ=2.5 does not compensate for anything; it *is* the overshoot. And the degradation
already measured on the baked deployment (acceptance 0.5128 vs a 0.55 floor, code
throughput 46.6–55.3 vs 59.32 tok/s) is **the cost of inverting the direction**, not the
cost of removing it.

This reframes the whole project. The argument is not "we avoid an FP8 round-trip." It is:

> **The entire useful range 0 < λ ≤ 1.5 is territory no published checkpoint occupies.**
> Baked abliteration jumps from +100 % (unedited) straight to −143 %. The clean point in
> between only exists as a runtime dial.

---

## 3. Phase 1 — extracting the directions

[`tools/extract_refusal_dirs.py`](tools/extract_refusal_dirs.py) dequantizes both
checkpoints in float64, computes ΔW = W_abl − W_base, takes its SVD, and emits one
4096-float `r̂` per module.

Ran as a CPU Job with a 12 GiB limit. Nothing downloaded — both checkpoints (156 G base
@`9e165c30`, 157 G abliterated @`21bd923c`) were already in the node's HF cache.

### The four gates — two passed, two failed

| # | metric | expected | measured | |
|---|---|---|---|---|
| 1 | edited modules | 46 | **46** (43 backbone + 3 MTP) | ✅ |
| 2 | rank-1 energy `S₀²/ΣS²` | ≥ 0.999 | **0.8376 – 0.9448** (mean 0.8970) | ❌ |
| 3 | mean δ Frobenius `‖ΔW‖/‖W‖` | 0.0587 (published) | **0.06018** (+2.5 %) | ✅ |
| 4 | λ effective | ~1.7 (hypothesis) | **2.429** (range 2.340 – 2.473) | ❌ |

Unrequested but decisive extras:

- **The edit subtracts** in all 46 modules (`⟨s₀v₀, u₀ᵀW_base⟩ < 0`). No risk of the hook
  amplifying refusal instead of removing it.
- **`cos(v₀, u₀ᵀW_base) ≥ 0.9873`** across all 46. The dominant component genuinely *has
  the shape of a projection*, not merely of some rank-1 edit.
- Backbone and MTP are edited **differently**: δ 0.0615 vs 0.0410, λ_eff 2.435 vs 2.343.
  The drafter is edited more weakly — which matters in §6.

### Gate 2 failed, and r=2 does not fix it

The instruction on failure was "evaluate r=2". Measured, and r=2 is not the answer
([`docs/probe-residual.json`](docs/probe-residual.json)):

| module | s₁ | s₂ | s₂/s₈ | E(r=1) | E(r=2) |
|---|---:|---:|---:|---:|---:|
| `layers.0` | 8.019 | 0.500 | 2.71 | 0.9014 | 0.9050 |
| `layers.21` | 7.697 | 0.492 | 2.64 | 0.8887 | 0.8923 |
| `layers.42` | 11.32 | 0.490 | 2.51 | 0.9448 | 0.9466 |
| `mtp.0` | 14.92 | 1.157 | 2.66 | 0.8376 | 0.8427 |

s₁/s₂ is 13–23×; from s₂ to s₈ the total decay is only 2.5–2.7× — a flat tail. Going to
r=2 buys **0.36 points** of energy, and r=8 still doesn't reach 0.91. There is no second
direction: the missing ~10 % is broadband noise from requantization to E4M3 with
power-of-two block scales.

Which means that 10 % **is precisely the artifact the runtime design exists to avoid**.
It is not something the hook should reproduce.

### Two corrections to the plan, found by measuring

**a) The scale format is not what the recipe says.** The suffix is `.scale`, not
`weight_scale_inv`, and the dtype is `F8_E8M0 [32,64]` — pure exponent, power of two, no
mantissa. Blocks are 128×128 as expected. Gate 3 landing at 0.06018 against the published
0.0587 confirms the product direction (`W = w_fp8 * scale`) is right; inverted, it would
be off by orders of magnitude.

**b) `wo_b` has no bias.** 92 keys = 46 × (`.weight` + `.scale`), zero `.bias`. The
usual precaution about projecting before adding the bias does not apply here.

**c) The SVD sign convention is unnecessary.** The hook `y − λ·r̂·(r̂·y)` contains `r̂`
twice — it is the outer product `r̂r̂ᵀ`, invariant to sign, since `(−r̂)(−r̂)ᵀ = r̂r̂ᵀ`. An
"inverted" `r̂` **cannot** amplify refusal. What can point the wrong way is the sign of λ.
The gate that matters is confirming the *published* edit subtracts — and that sign is a
property of ΔW, not a choice made by the SVD.

---

## 4. Phase 2 — validating the hook offline

[`tools/verify_projection.py`](tools/verify_projection.py): CPU torch harness, 4 modules,
256 Gaussian input tokens, sweep over 14 values of λ.

- **G1 — equivalence:** passes at 9e-16 (table in §1). The only hard gate of the phase.
- **G5 — λ=0 bit-exact:** `torch.equal(hook(y, r, 0), y)` → `True` in all 4 modules.
- **G3 — the inversion finding:** §2.
- **G4 — kernel precision:**

  | | median relative error |
  |---|---:|
  | bf16 activations, dot product in **fp32** (the design) | 1.66e-3 |
  | bf16 activations, dot product in bf16 (the easy mistake) | 2.29e-3 |

  fp32 in the dot product buys ~28 %. Real, but not worth overselling: both are dominated
  by storing `y` in bf16, not by the dot product. Do it in fp32 — it's free — but the
  order of magnitude doesn't move.

**On "removes 100 % of the direction":** in exact arithmetic, yes — the residual component
at λ=1 is 0. With bf16 activations the real residual is ~1.7e-3 relative. So **99.83 %**,
not 100 %. Still incomparably cleaner than −143 %, but 99.83 % is the honest number.

### A threshold that was unreachable by construction

The plan asked that the hook reproduce `W_abl·x` within ~1e-3 median relative error.
Phase 1 predicted this was impossible, and Phase 2 confirmed it: at λ=1 the error against
`W_abl` is 0.020–0.039, twenty times the threshold.

Not a bug in the hook. `W_abl` **is not** the ideal projection — it is the ideal projection
*plus* requantization noise. Demanding the hook reproduce `W_abl` is demanding it reproduce
the noise the design set out to eliminate. The correct gate is against the ideal float64
projection, where it lands at ~1e-7; distance to `W_abl` is an observation, not a gate.

The λ-sweep confirms it independently — error against `W_abl` bottoms out at λ*=2.43 in all
three backbone modules and 2.25 in `mtp.0`, pinning Phase 1's λ_eff by a second route.

---

## 5. Phase 3 — the vLLM patch

[`patches/0001-rank1-projection.patch`](patches/0001-rank1-projection.patch) — 460 lines,
7 files. Applies cleanly to vLLM `0.25.2.dev0+g752a3a504`, compiles, and imports the whole
chain patched.

> There is no vLLM fork on disk — the deployment runs a prebuilt image, so the reference
> tree was extracted **from the image itself**. It is the code that actually runs, not
> upstream.

| file | role |
|---|---|
| [`vllm/refusal_projection.py`](vllm/refusal_projection.py) | **new.** Direction loading, λ state, kernel, prefix resolution, hash key |
| `vllm/models/deepseek_v4/attention.py` | hook construction in `__init__`, application at the `_o_proj` return |
| `vllm/v1/core/kv_cache_utils.py` | λ into the block hash key |
| `vllm/v1/worker/gpu_worker.py` | `set/get_refusal_lambda` via `collective_rpc` |
| `vllm/entrypoints/serve/refusal/api_router.py` | **new.** `POST`/`GET /admin/refusal_lambda` |

### The five things that will silently break this

**1 · λ must be a device tensor, never a Python float.** The deployment captures CUDA
graphs (`--max-cudagraph-capture-size 48`). A Python scalar gets **baked into the captured
graph** — changing it at runtime does nothing, with no error. One tensor per device, shared
by all layers, mutated with `fill_()`. There is an explicit test that `data_ptr` does not
change on reassignment; that is what guarantees an already-captured graph sees the new value.

**2 · λ must enter the prefix-cache hash key.** With `--enable-prefix-caching`, a prefix
cached at λ=0 and reused at λ=1 yields corrupt state, silently. The right precedent in vLLM
is **`cache_salt`**, not `lora_int_id` — it already exists in
`generate_block_hash_extra_keys` and applies only at `start_token_idx == 0`, which suffices
because the hash chains through `parent_block_hash`. λ is quantized to an integer (×1000).
Change λ and the old blocks stop matching and age out on their own.

**3 · The setter must go through `collective_rpc`.** With the `mp` executor the frontend
shares no process with the workers; a local setter reaches no rank. All ranks are checked to
return the same value — a mismatch is a **500**, not a 200, because a cluster running
different λ per rank produces garbage silently.

**4 · Apply at the sublayer output, before `hc_post`.** The site is the return of `_o_proj`
in `attention.py`, not the NVIDIA-path `o_proj.py` (a free function with no layer identity).
`attention.py` covers all four backends at once and has `self.prefix`. The mHC architecture
carries `[B,S,4,4096]`, but the mHC write is rank-1 across streams, so removing the direction
at the sublayer output removes it from all four streams simultaneously. Do not project onto
the carrier.

**5 · TP needs nothing special.** `wo_b` is a `RowParallelLinear` with `reduce_results`
default, so its return is already post-all-reduce on the full rank. The projection is linear
and commutes with the all-reduce.

### Control surface

`POST /admin/refusal_lambda {"lambda": 1.5}` and `GET`. The router **only mounts if
`VLLM_REFUSAL_DIRS` is set** — no hook, no endpoint to attack. Bound `[-1, 4]`; λ<0 is
deliberately allowed, since it is the only way to ask for a *more* reticent model.

vLLM cannot guarantee this route stays off the internet. It is an ordinary HTTP route —
the ingress decides. `/admin/*` should not leave the cluster network.

### Tests — 22/22 + 15/15, inside the image

Direction loading (46 × `[4096]` f32, unit norm) · prefix resolution for backbone, DSpark
drafter and MTP paths · hook math against a float64 reference at λ = 0 / 0.5 / 1 / 2.43
(err ≤ 1.6e-3, matching the bf16 floor from Phase 2) · **λ=0 bit-exact** via `torch.equal` ·
λ tensor mutated in place with stable `data_ptr` · distinct, integral, stable hash key per λ ·
no `VLLM_REFUSAL_DIRS` → no hook, no key, no branch in the forward pass. Plus endpoint tests:
conditional mounting, 422 bounds at −5 and 9, λ=−1 accepted, divergent ranks → 500.

---

## 6. Phase 4 — the MTP overlay turned out to be unnecessary

The plan assumed the three DSpark drafter stages run on their own path, out of reach of the
target's hook, and budgeted a ~100 MB baked overlay for them.

**Not so.** Both `mtp.py:125` and `dspark.py:91` construct the **same
`DeepseekV4DecoderLayer`** — the same `DeepseekV4Attention`, the same `wo_b`. One hook
reaches all three. The drafter layers are *named* `layers.{num_hidden_layers + i}` but
*loaded* from the checkpoint's `mtp.*` weights; `resolve_direction` implements exactly that
mapping and is tested: `layers.43 → mtp.0`, `layers.45 → mtp.2`, `mtp.1 → mtp.1`.

The overlay is dropped, and a capability appears that baking never had: **target and drafter
share one dial and stay aligned by construction.** In the published checkpoint they are not
(λ_eff 2.44 backbone vs 2.34 MTP), and that misalignment is a prime candidate for part of the
measured acceptance drop.

---

## 7. Phase 5 — A/B validation

All arms on the **same pod, same day, same load**. Only the dial changes.

### Acceptance, throughput, latency — 6 alternated runs per arm, 83.4 min

Source: [`bench/results/compare_full.log`](bench/results/compare_full.log)

| metric | λ=0 | λ=1.5 | t |
|---|---:|---:|---:|
| acceptance (code) | 0.5669 ± 0.0097 | 0.5608 ± 0.0189 | +0.70 |
| acceptance (varied) | 0.4747 ± 0.0122 | 0.4762 ± 0.0202 | −0.15 |
| mean accept length | 3.8338 ± 0.0482 | 3.7992 ± 0.0949 | +0.80 |
| tok/s code | 54.69 ± 6.96 | 51.16 ± 10.32 | +0.70 |
| tok/s varied | 46.34 ± 5.68 | 46.89 ± 6.16 | −0.16 |
| TTFT median | 0.2265 ± 0.0350 | 0.2362 ± 0.0400 | −0.45 |

|t| < 2 on every axis. **λ=1.5 costs nothing measurable.**

Runs alternate λ=0 / λ=1.5 rather than running one arm then the other, because this box
drifts: two λ=0 runs in the same session came in at 57.87 and 40.52 tok/s. Blocked arms
would have attributed that drift to λ.

### Refusal rate — the direct measure

[`bench/bench_refusal.py`](bench/bench_refusal.py): 10 low-harm triggers, 4 benign controls.
Classifies refusal vs response from opening markers only — **it does not read or store
response content**. Empty responses are their own category, never counted as "answered".

| λ | refuses | invalid | controls falsely refused |
|---|---:|---:|---:|
| 0 | **9/10** | 0 | 0/4 |
| 1 | 5/10 and 7/10 (two runs) | 0 | 0/4 |
| **1.5** | **0/10** | 0 | 0/4 |
| 2 | **0/10** | 0 | 0/4 |

**It saturates at 1.5.** λ=2 adds nothing over 1.5 and only moves toward the baked regime
(2.43), which does degrade. If you raise it, 1.5 is the point — never 2.

### Long-context retrieval — the gap nobody had measured

The reference model card captured its directions with ~60-token prompts, and no published
DeepSeek-V4 abliteration had been validated beyond 32k. This deployment serves at 262k.

| arm | 32k | 128k | total | raw |
|---|---:|---:|---:|---|
| base image, unpatched | 15/15 | 15/15 | **30/30** | `niah_baseline_lambda0.json` |
| λ=0 | 15/15 | 15/15 | **30/30** | `niah_rank1_lambda0.json` |
| λ=1 | 15/15 | 15/15 | **30/30** | `niah_rank1_lambda1.json` |
| λ=1.5 | 15/15 | 15/15 | **30/30** | `cf_niah_1.5.json` |

3 needles per cell, 5 depths (0/25/50/75/100 %), temperature 0, haystack length calibrated
against the server's own `/tokenize` and cross-checked against `usage.prompt_tokens` —
31,770 and 127,007 real tokens in the paired λ=0/λ=1.5 run.

**λ up to 1.5 does not degrade deep retrieval.**

> **Reading the raw files:** `cf_niah_1.5.json` records `"lambda": 0.0` internally. That is a
> harness artifact — `compare_full.py` sets λ out-of-band through `/admin/refusal_lambda`
> between arms, and `bench_niah.py` stamps only the λ it was passed on the command line.
> [`compare_full.log`](bench/results/compare_full.log) is what ties each file to its arm.

### Tool-calling — 8/8 both arms

Including the impossible-request test: at λ=1.5 the model still refuses to claim it issued
a refund it cannot issue. **The projection does not remove the ability to say "I can't"
when that is a real tool limit** — it removes the policy refusal, not the grounded one.

---

## 8. Three ways these benchmarks lied before they were fixed

Documented because each one nearly shipped as a finding.

**1 · The model reasons before answering, and the budget ate the answer.** First refusal
run at `max_tokens=400`: 9 of 10 responses at λ=1 came back empty on budget, and the
summary read `1/10 refuses`. Without an "invalid" column that reads as *spectacular*
success — and it was false. The model spends **1,100–1,400 tokens reasoning** first. Same
failure hit NIAH at `max_tokens=96`: 1/3 with 2 empties, which looked like the model
failing. At 512 tokens: 3/3.

**2 · Exact string match is not a valid gate.** Discovered by running λ=0 against itself:

```
L=128000 d=25%   A: 'SK-7734-QX'   B: 'SK-773-QX'    30/30 -> 29/30
```

vLLM is not deterministic run-to-run at temperature 0 — continuous batching changes GEMM
reduction order and moves the argmax on ties. Measured noise is **±1 cell in 30**. Without
that control, two wording differences between base and λ=0 would have been reported as
"the hook is not inert" — a false positive that would have stopped the project.

**3 · n=1 decides nothing on this bench.** The first two λ=1 measurements gave 0.5383 and
0.5944 — one below the floor, one well above, same λ. Any verdict at n=1 here is noise.

---

## 9. What this does **not** establish

- **General capability is unmeasured.** MMLU-Pro, GSM8K, HumanEval were not run — the same
  gap the reference model card left open. Tool-calling, retrieval and acceptance are covered;
  general reasoning is not.
- **256k is unmeasured**, deliberately. Fifteen prefills at that length against a model
  serving live traffic is a load to schedule, not to sneak in.
- **Variance rises with λ, even where the mean passes.** 1 of 6 runs at λ=1.5 and 2 of 6 at
  λ=1 dipped below the 0.55 acceptance floor. Means are indistinguishable; individual runs
  are not always.
- **λ=0 is bit-exact in output but not free in compute.** The dot product and subtraction
  execute in all 46 layers per token, with an fp32 cast, whatever λ is. Skipping it would
  require a branch on λ, which is exactly what breaks CUDA graph capture. For zero cost,
  unset `VLLM_REFUSAL_DIRS` and restart.

---

## 10. Security: this cuts in an uncomfortable direction

Reported because the deployment wires this model to MCP servers with **write** access to
external services, and it ingests scraped third-party content.

Reducing a model's resistance to instructions reduces its resistance to **injected**
instructions arriving inside untrusted content. Prompt injection and refusal are not
independent failure modes — they route through overlapping machinery.

Phase 2 sharpens this rather than softening it: a clean λ=1.5 produces a model **more
capable and more completely stripped of its ability to decline** than the baked checkpoint,
whose clumsiness incidentally limited how useful it was. **The better the dial works, the
more the isolation matters.**

Recommended, and the reason λ is per-request in
[`patches/0002-per-request-lambda.patch`](patches/0002-per-request-lambda.patch):

- **λ>0 must not share credentials with write-capable tools.** Separate deployment or
  enforced λ=0 on any request whose context contains scraped content or inbound mail.
- **Restrict the toolset when λ>0.** Read-only tools; no write path reachable from an
  ablated context.
- **Keep `/admin/refusal_lambda` off the public ingress.** vLLM cannot enforce this.

---

## 11. Reproducing it

```bash
# 1 — directions (CPU, ~12 GiB, no download if both checkpoints are cached)
python3 tools/extract_refusal_dirs.py \
  --base   <path-to>/DeepSeek-V4-Flash-0731 \
  --abl    <path-to>/DeepSeek-V4-Flash-0731-abliterated \
  --out    refusal_dirs.safetensors \
  --report docs/extraction-report.json
# gate: exactly 46 modules, all subtracting

# 2 — offline equivalence (must pass before touching a deployment)
python3 tools/verify_projection.py --out docs/verify-projection.json
# gate: G1 ~1e-16 in float64, G5 λ=0 bit-exact

# 3 — patch and build
patch -p1 < patches/0001-rank1-projection.patch
patch -p1 < patches/0002-per-request-lambda.patch   # optional: per-request λ
python3 -m pytest tools/test_refusal_projection.py tools/test_admin_endpoint.py

# 4 — serve
VLLM_REFUSAL_DIRS=/path/refusal_dirs.safetensors  vllm serve ...

# 5 — equality gate BEFORE trusting anything: λ=0 vs the unpatched base,
#      same prompt, temperature 0. Outputs must match. If not, the hook is not inert.

curl -XPOST localhost:8888/admin/refusal_lambda -d '{"lambda": 1.5}'

# 6 — A/B
python3 bench/compare_full.py --base http://<head>:8888 --lambdas 0,1.5 --reps 6
```

`deploy/` carries the Dockerfile and the Kubernetes Jobs for extraction, verification,
probing and image build.

### Deployment notes for 2× DGX Spark

From [`docs/01-HALLAZGOS-Y-ACTIVACION.md`](docs/01-HALLAZGOS-Y-ACTIVACION.md) — three places
the public recipe is wrong on this hardware:

- **It points at a dead NIC.** The recipe uses `NCCL_IB_HCA=rocep1s0f1` /
  `NCCL_SOCKET_IFNAME=enp1s0f1np1`. On these Sparks that port has no cable
  (`physical_state DISABLED`). The live links are the `f0` port of each ConnectX-7 — and
  there are **two**, not one. Copying the recipe verbatim hangs NCCL at init.
- **`num_speculative_tokens` must be 5, not 7.** The recipe's 7 is for the FP8/NVFP4
  variants with `method=mtp`. This checkpoint declares `dspark_block_size: 5`, and k<5
  **truncates draft blocks silently**.
- **`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK` does not apply.** Stage-C image only; ignored
  with a warning on Anemll 0.1.1.

Also measured there: the native `deepseek-ai/…-0731` checkpoint beats `nvidia/…-NVFP4` for
this target — smaller on disk (166.9 vs 168.3 GB), 97.4 % of params already FP4, and it
**includes the DSpark drafter** (`mtp.*`), which the NVFP4 repo omits. Without a drafter,
acceptance is 0 by construction.

---

## 12. Repository layout

```
tools/      extraction, offline verification, spectral probe, tests
vllm/       refusal_projection.py — the module the patch installs
patches/    0001 runtime projection · 0002 per-request λ
bench/      the four harnesses + compare drivers
bench/results/  every raw JSON behind every number above
docs/       phase reports (Spanish, as written) + raw metric dumps
deploy/     Dockerfile + Kubernetes Jobs
hf/         refusal_dirs.safetensors + model card
```

The phase reports in `docs/` are the original working documents, in Spanish. They contain
the reasoning as it happened, including the wrong turns.

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

## License

Code: Apache-2.0. The direction vectors are derived from the difference between two
publicly released checkpoints and inherit the DeepSeek Model License.
