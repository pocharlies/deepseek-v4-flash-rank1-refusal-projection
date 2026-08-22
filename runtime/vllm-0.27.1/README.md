# DeepSeek V4 DSpark on vLLM 0.27.1

This directory publishes the exact compatibility changes used to move the existing
DeepSeek-V4-Flash-0731 DGX Spark runtime from vLLM 0.25.2 to 0.27.1 while keeping
the serving configuration unchanged:

- TP=2 across two GB10/sm_121 nodes;
- DSpark probabilistic decoding with `k=5`;
- `nvfp4_ds_mla` KV cache and 393216 context;
- B12X MXFP4 experts, 4 sequences, 8192 batched tokens and CUDA graph size 24;
- vision, `min_p=0.01` and the per-request rank-1 refusal dial.

## Why this is more than changing the image tag

The server arguments do not change, but three binary/source compatibility fixes are
required:

1. `anemll-dspark-vllm0271.patch` ports the stable NVFP4 and sparse-MLA DSpark paths
   to the exact vLLM 0.27.1 tree. `patch_vllm_0271_deepseek.py` restores the DeepSeek
   rank-1 hook, DSpark draft-buffer layout and B12X MXFP4 oracle registration.
2. vLLM 0.27.1 ships DeepGEMM 2.6.1, whose scale-layout path rejects the `(1,32)`
   DeepSeek V4 block on sm_121. `Dockerfile.deepgemm-sm121` recompiles the exact
   DSpark DeepGEMM 2.5.0 commit `a6b593d2826719dcf4892609af7b84ee23aaf32a`
   against the PyTorch 2.13 ABI in the new image. Copying the old `_C.so` is invalid.
3. The historical vision processor tokenizes the image marker but does not expand it.
   vLLM 0.27.1 assumes otherwise. `patch_dsv4_vision_0271.py` opts into vLLM's prompt
   updates with the same override used by upstream multimodal processors.

The `min_p` speculative-decoding hotfix is retained because vLLM 0.27.1 does not yet
contain [vllm-project/vllm#42802](https://github.com/vllm-project/vllm/pull/42802).
Its tests keep `logit_bias` blocked and cover the per-request/per-position mask.

## Reproducible three-stage build

The first stage needs two operator-controlled inputs:

- `LEGACY`: the already validated 0.25.2 DSpark image carrying vision, B12X and rank-1;
- `BASE`: the vLLM 0.27.1 ARM64 image carrying the generic per-request rank-1 plumbing.
  The companion Qwen repository publishes that fail-closed base patch under
  [`runtime/vllm-0.27.1/`](https://github.com/pocharlies/qwen38-27b-rank1-refusal-projection/tree/main/runtime/vllm-0.27.1).

Build from the repository root and pin both inputs by digest:

```sh
docker buildx build --platform linux/arm64 \
  --build-arg LEGACY="$LEGACY_IMAGE_AT_DIGEST" \
  --build-arg BASE="$VLLM0271_RANK1_BASE_AT_DIGEST" \
  -f runtime/vllm-0.27.1/Dockerfile \
  -t your-registry/deepseek-v4:v0.27.1-core --push .

docker buildx build --platform linux/arm64 \
  --build-arg BASE=your-registry/deepseek-v4:v0.27.1-core \
  --build-context deepgemm-src='https://github.com/deepseek-ai/DeepGEMM.git#a6b593d2826719dcf4892609af7b84ee23aaf32a' \
  -f runtime/vllm-0.27.1/Dockerfile.deepgemm-sm121 \
  -t your-registry/deepseek-v4:v0.27.1-deepgemm-sm121 --push .

docker buildx build --platform linux/arm64 \
  --build-arg BASE=your-registry/deepseek-v4:v0.27.1-deepgemm-sm121 \
  -f runtime/vllm-0.27.1/Dockerfile.vision \
  -t your-registry/deepseek-v4:v0.27.1-dspark --push .
```

## Measured result

On 2026-08-22 the final image completed weight loading, multimodal profiling, Sparse
MLA autotuning, full/piecewise CUDA graphs and DSpark graph capture with zero restarts.
Health, `min_p`, tool calling and image input all passed.

| single-stream workload | decode | TTFT | DSpark acceptance |
|---|---:|---:|---:|
| Python code, 3 × 256 tokens | **67.41 tok/s median** | 0.35 s median | 67.39% |
| technical prose, 3 × 256 tokens | 40.80 tok/s median | 0.36 s median | 32.89% |

The previous clean code-heavy reference was 58–66 tok/s on the same hardware, though
it was not the exact same prompt. The result supports 0.27.1 for this DeepSeek stack;
the workload spread also shows that DSpark acceptance, not a fixed runtime ceiling,
dominates single-stream decode.

Machine-readable evidence is in
[`hf/benchmarks/2026-08-22/vllm-0.27.1-dspark.json`](../../hf/benchmarks/2026-08-22/vllm-0.27.1-dspark.json).
