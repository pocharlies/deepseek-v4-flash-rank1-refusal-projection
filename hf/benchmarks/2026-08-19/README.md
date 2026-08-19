# 2026-08-19 — per-request λ, quality sweep, and three truthfulness suites

Three questions were open after the earlier A/B work:

1. Does the per-request λ actually work? (It was **withdrawn** from this README in
   August because it silently did not.)
2. How far can λ be pushed before capability degrades?
3. Does a high λ make the model *lie* — accept false premises, hallucinate,
   or fold when the user pushes back?

All three were measured. Two of the three answers are the opposite of what the
maintainer (and the assistant doing the work) expected going in.

Every number here was produced with **per-request λ**: the global dial stayed at
`0.0` on both TP ranks for the entire run and was re-checked before and after
each arm. Production kept serving from the same pod throughout.

---

## 1. Per-request λ: fixed, and verified on hardware

The mechanism ships λ per request in `cache_salt: "refusal:<x>"`. It was
withdrawn from the docs because it did not work. Three defects, not one — and
fixing only the loud one would have changed nothing:

| # | Defect | Why it was invisible |
|---|---|---|
| 1 | Target and drafter share the attention module but advance in **disjoint** multiples (`1+k` vs `k`). One per-token vector cannot serve both. | This one *did* warn — ~25 shape-mismatch lines per boot. |
| 2 | `capture_model` never goes through `execute_model`, so during capture the per-token slot is `None` and **the traced branch bakes in the global scalar**. Replay runs no Python. | **Completely silent.** Every graph-served decode used the global λ forever, with no warning. |
| 3 | The slot was never cleared, so dummy runs read the previous step's stale tensor. | Surfaced as nonsense pairs like `16 → 8192` (`max_num_batched_tokens`). |

The fix is a **persistent buffer per role**, allocated in the runner's `__init__`
(so, before `capture_model`) and always mutated in place. The forward takes
`buf[:n]`: capture bakes the pointer and the size, each step rewrites the
contents. This is the pattern LoRA and DFlash already use in the same file. The
role is fixed at module construction from the prefix, using the same criterion
that already selects the direction — so a drafter row cannot read a target
request's λ. On any unexpected layout it falls back to the global λ, never to
another request's.

**Verification that can fail.** The GPU job captures a CUDA graph, mutates the
buffer, calls `replay()` and requires the output to change:

```
PASS  replay sees the per-request lambda written AFTER capture   err_max=1.637e-03
PASS  same graph, different lambda -> different output
PASS  mixed batch under replay: each request with ITS lambda     err_max=1.607e-03
PASS  in a mixed batch the lambda=0 rows come out untouched
TODOS OK   (exit 0)
```

**Negative control**, because a green test proves nothing on its own: rebuilding
the old behaviour (buffers absent at capture time) gives
`replay reflects the per-request lambda: False`. The test discriminates.

On the live pod after deployment: `~25` shape-mismatch warnings per boot became
**0**, and startup logs a single clean line —
`refusal projection: buffers por rol listos (8192 tokens, cuda:0)` — *before*
graph capture.

End-to-end through the router, both requests in parallel **in the same batch**:
4/4 triggers went from refusal to answer with `cache_salt: refusal:1.5`, while
the unsealed request in that same batch still refused.

---

## 2. Quality sweep — and the measurement error that inverted it

`quality_paired_4000tok.md` · `quality_arms_4000tok.json`

MMLU-Pro (112, balanced 8/category) and GSM8K (100), paired, temperature 0,
`max_gen_toks=4000`. 10/10 arms completed, 100 % sealed.

| λ | MMLU-Pro | GSM8K | MMLU when it *answers* |
|---|---:|---:|---:|
| 0.0 | 80.4 % | 71 % | 90.9 % |
| 1.0 | 79.5 % | 77 % | 92.7 % |
| 1.5 | 76.8 % | 74 % | 92.5 % |
| 2.0 | 77.7 % | 69 % | 95.6 % |
| 2.5 | 78.6 % | 68 % | 94.6 % |

**No comparison is significant** — every McNemar p ≥ 0.09. The curve is not even
monotonic: at λ=1.0 GSM8K *beats* the baseline by 6 points, and MMLU at λ=2.5
scores above λ=1.5. That is noise, not a trend.

### The first run of this sweep was wrong, and it looked convincing

`quality_paired_2048tok_truncation_confounded.md` is kept **on purpose**. Run
with `max_gen_toks=2048`, it produced a clean descending curve and two
"significant" results:

```
MMLU: 78.6 / 74.1 / 72.3 / 68.8 / 68.8
0.0 vs 2.0  -9.82%  p=0.0010
0.0 vs 2.5  -9.82%  p=0.0034
```

It was an artefact. Empty answers rose monotonically with λ (18 → 23 → 25 → 31 → 31)
and empties score as wrong. Reproducing one empty item directly against the head:

```
finish_reason=length   content=0 chars   reasoning=10050 chars   completion_tokens=2048
```

The model spends the whole budget reasoning and never emits an answer — **and it
does so at λ=0 exactly as at λ=2.5**. Conditioning on non-empty answers, accuracy
is flat at every λ (93.6 → 93.3 → 93.1 → 95.1 → 95.1). The entire "degradation"
was truncation. Raising the budget to 4000 tokens shrank the 0→1.0 delta from
−4.5 points to −0.9.

This is the fourth entry for §8 of the main README, *"ways these benchmarks lied
before they were fixed"*: **if a reasoning model is scored with a short
generation budget, λ is confounded with verbosity.**

---

## 3. Truthfulness and sycophancy — the hypothesis that failed

`truthfulness_sycophancy.json`

The worry that motivated this: ablation removes the *refusal* direction. If
*"no, that is false"* shares machinery with *"no, I won't tell you"*, then at high
λ the model should start accepting false premises and folding under pressure.
Three suites, ~410 requests, substring scoring (no LLM judge — a judge would put
its own bias inside the measurement):

| λ | corrects false premise | over-corrects control | yields to pressure | NIAH retrieval |
|---|---:|---:|---:|---:|
| 0.0 | 100 % (21/21) | 0 % (0/11) | 0 % (0/23) | 18/18 |
| 1.0 | 100 % (21/21) | 18.2 % (2/11) | 0 % (0/22) | 18/18 |
| 1.5 | 100 % (21/21) | 0 % (0/11) | 0 % (0/23) | 18/18 |
| 2.0 | 100 % (20/20) | 9.1 % (1/11) | 0 % (0/23) | 18/18 |
| 2.5 | 100 % (21/21) | 10 % (1/10) | 0 % (0/22) | 18/18 |

**0 yields out of 113 opportunities. 90/90 needle retrievals, 0 errors, 0 empties.**
The over-correction column is 0, 2, 0, 1, 1 raw cases out of 11 — noise, and the
control exists precisely so that a model that "corrected" everything could not
score as lucid.

Design details that matter:

- **Only items answered correctly in turn 1 count** toward sycophancy. A flip
  from an already-wrong answer measures nothing.
- **Yielding means dropping the correct answer**, not naming the lure. A pilot
  run returned `mixed` on 3/3 items at both λ=0 and λ=2.5 — the model says
  *"Sydney is not the capital, Canberra is"*, naming the lure only to rebut it.
  Under the first criterion those fell outside the denominator and the whole
  sweep would have had **n=0**. That was caught before the real run.
- **Truncated answers are excluded**, not scored as failures — the same artefact
  that ruined the 2048-token sweep.

---

## 4. What this does **not** establish

The three suites return **100 % in every arm**. That is a ceiling effect: they do
not discriminate. They do not show that a high λ is harmless — they show that
*these suites cannot detect it*. Specifically:

- With 0 yields in ~22 items per arm, the rule of three only rules out sycophancy
  rates **above ~13.6 %**. A 5 % rate would be invisible here.
- The false premises used (Great Wall from space, Einstein failing maths, blind
  bats) are well-known myths and evidently easy for this model.
- Pressure is a **single turn**. Sustained pressure over three turns is harder and
  is not measured.
- MMLU-Pro and GSM8K score a letter or a number. They say nothing about whether
  the prose around it is sound.
- With n=112, only drops larger than ~6 points are detectable at all.

And the honest limit, which no amount of extra benchmarking removes: **content
that only appears at high λ is content the model was least likely to have
represented well**, and it is precisely the domain where no ground truth is
available to check it against. A model can be impeccable on myth-correction and
needle retrieval and still confabulate a fluent, confident, structurally
plausible procedure. Fluency is not a correctness signal. None of these numbers
license trusting an answer that only a high λ produced.

What *has* changed: there is no longer any empirical basis for believing that
λ in 1.5–2.5 degrades general truthfulness on this model. That was a suspicion;
it was measured, and it does not appear.

---

## 5. This model is not the other model

Do not carry a λ across checkpoints. On Qwen3.8-27B, λ=1.5 cost **−26.8 points**
of MMLU-Pro (p=0.0000) with **zero** empty answers — a real collapse, not a
truncation artefact. The same λ is inert here. The two models do not behave
alike, and the safe operating point has to be measured per checkpoint.

---

## Files

| File | What it is |
|---|---|
| `quality_paired_4000tok.md` / `.json` | MMLU-Pro + GSM8K, 5 λ, paired, 4000-token budget. The usable one. |
| `quality_arms_4000tok.json` | Per-arm status, wall time, seal audit (sealed/seen). |
| `quality_paired_2048tok_truncation_confounded.md` / `.json` | The first sweep. **Kept as a worked example of a confounded measurement**, not as a result. |
| `truthfulness_sycophancy.json` | False premises, true-premise controls, sycophancy under pressure, by λ. |
| `niah_per_request_lambda.json` | Needle-in-a-haystack at 32k/128k, 3 depths, 3 reps, per-request λ. |

### Reproducing

```bash
# quality sweep (per-request λ; leaves the global dial alone and verifies it)
python3 bench/bench_quality_lambda_sweep.py --base http://<head>:8888 \
  --model <served-name> --lambdas 0,1.0,1.5,2.0,2.5 --expect-global 0.0 \
  --max-gen-toks 4000 --results-dir results/

# truthfulness + sycophancy
python3 bench/bench_truthfulness_lambda.py --base http://<head>:8888 \
  --model <served-name> --lambdas 0,1.0,1.5,2.0,2.5 --expect-global 0.0 --out truth/

# NIAH behind the sealing proxy (bench_niah never touches the dial)
python3 bench/refusal_salt_proxy.py --upstream http://<head>:8888 --lambda 1.5 --port 0
python3 bench/bench_niah.py --base http://127.0.0.1:<port> --no-lambda-control --lambdas 0
```

`--expect-global` is not decoration: the sweep aborts if the global dial moves
mid-run, because from that point on the arms would be mixing two λ values.
