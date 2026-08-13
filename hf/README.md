---
license: other
license_name: deepseek
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - abliteration
  - refusal-direction
  - activation-steering
  - vllm
  - deepseek
  - interpretability
library_name: safetensors
---

# DeepSeek-V4-Flash-0731 — runtime rank-1 refusal projection

**757 KB of direction vectors instead of a 1.54 GB weight overlay.** Base checkpoint stays
byte-identical to the DeepSeek release; the ablation strength λ becomes a dial you change at
runtime with no restart and no reload.

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

**The baked checkpoint cannot reach this operating point.** It exists only at λ_eff ≈ 2.43,
where acceptance falls to **0.5128** — below the 0.55 floor this deployment requires.

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

## Calibration

| λ | refusal rate | acceptance | NIAH 128k |
|---:|---:|---:|---:|
| 0 | 90 % | 0.5669 ± 0.0097 (n=6) † | 30/30 |
| 1 | 50–70 % | 0.5635 (n=6) ‡ | 30/30 |
| **1.5** | **0 %** | **0.5608 ± 0.0189 (n=6)** † | **30/30** |
| 2 | 0 % | — | — |
| ~2.43 (baked) | — | **0.5128** ‡ | not measured |

† per-run raw JSON in the repo (`bench/results/cf_speed_*.json`, `compare_full.log`) —
the λ=0 vs λ=1.5 arms were run alternated in one 83.4-minute session.
‡ from the working report `docs/projection-validation.md`; the per-run JSON for these
arms is not in the published set.

**It saturates at 1.5.** λ=2 adds nothing over 1.5 and only moves toward the baked regime,
which does degrade. If you raise it, 1.5 is the point — never 2.

---

## What this does not establish

- **General capability is unmeasured.** MMLU-Pro, GSM8K, HumanEval were not run — the same
  gap the reference model card left open.
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

- **λ>0 should not share credentials with write-capable tools** — separate deployment, or
  enforced λ=0 on any request whose context contains untrusted content. This is why λ is
  per-request in patch `0002`.
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
