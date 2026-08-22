# syntax=docker/dockerfile:1.7
# DeepSeek V4 keeps the exact production overlays and serving configuration;
# only the vLLM runtime underneath them moves from 0.25.2.dev0 to 0.27.1.
# Both inputs are deliberately supplied by the builder. LEGACY is the operator's
# known-good 0.25.2 DSpark image with vision/B12X/rank-1; BASE is the 0.27.1 ARM64
# image carrying the generic per-request rank-1 plumbing. See README.md.
ARG LEGACY
ARG BASE
FROM ${LEGACY} AS legacy

# This image is the digest-pinned official vLLM 0.27.1 ARM64 runtime already
# exercised by Qwen38.  Its generic per-request rank-1 plumbing is model-neutral.
FROM ${BASE}

ARG SITE=/usr/local/lib/python3.12/dist-packages

# Preserve the production DeepSeek-only capabilities byte-for-byte.  They are
# copied from the pinned stable image, not rebuilt from mutable package indexes.
COPY --from=legacy ${SITE}/dsv4_vision_vllm/ \
                   ${SITE}/dsv4_vision_vllm/
COPY --from=legacy ${SITE}/dsv4_vision_vllm-0.1.0.dist-info/ \
                   ${SITE}/dsv4_vision_vllm-0.1.0.dist-info/
COPY --from=legacy ${SITE}/vllm/refusal_projection.py \
                   ${SITE}/vllm/refusal_projection.py
COPY --from=legacy ${SITE}/vllm/v1/worker/gpu/refusal_utils.py \
                   ${SITE}/vllm/v1/worker/gpu/refusal_utils.py
COPY --from=legacy ${SITE}/b12x/ \
                   ${SITE}/b12x/
COPY --from=legacy ${SITE}/b12x-0.15.3.dist-info/ \
                   ${SITE}/b12x-0.15.3.dist-info/
COPY --from=legacy ${SITE}/vllm/model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py \
                   ${SITE}/vllm/model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py
# DeepGEMM is deliberately not copied from the legacy image: its pybind
# extension is linked against the old libtorch ABI.  Dockerfile.deepgemm-sm121
# recompiles the exact DSpark source revision against this image's PyTorch.
COPY --from=legacy /opt/refusal/refusal_dirs.safetensors \
                   /opt/refusal/refusal_dirs.safetensors

COPY runtime/vllm-0.27.1/anemll-dspark-vllm0271.patch /opt/deepseek-v4/anemll-dspark-vllm0271.patch
COPY runtime/vllm-0.27.1/patch_vllm_0271_deepseek.py /opt/deepseek-v4/patch_vllm_0271_deepseek.py
COPY runtime/vllm-0.27.1/payload/minp_spec_decode/vllm_minp_spec_decode.py \
     ${SITE}/vllm_minp_spec_decode.py
COPY runtime/vllm-0.27.1/payload/minp_spec_decode/vllm_minp_spec_decode.pth \
     ${SITE}/vllm_minp_spec_decode.pth
COPY runtime/vllm-0.27.1/test_minp_spec_decode_hotfix.py \
     /opt/deepseek-v4/test_minp_spec_decode_hotfix.py

RUN cd ${SITE} \
    && patch --batch --forward -p1 < /opt/deepseek-v4/anemll-dspark-vllm0271.patch \
    && python3 /opt/deepseek-v4/patch_vllm_0271_deepseek.py --site ${SITE}/vllm \
    && python3 -m py_compile \
      ${SITE}/vllm/models/deepseek_v4/attention.py \
      ${SITE}/vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py \
      ${SITE}/vllm/models/deepseek_v4/sparse_mla.py \
      ${SITE}/vllm/model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py \
      ${SITE}/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py \
      ${SITE}/vllm/v1/worker/gpu/model_runner.py \
      ${SITE}/vllm/v1/attention/backends/mla/sparse_swa.py \
      ${SITE}/vllm/v1/kv_cache_interface.py \
      ${SITE}/vllm/refusal_projection.py \
      ${SITE}/vllm/v1/worker/gpu/refusal_utils.py \
      ${SITE}/vllm_minp_spec_decode.py \
      ${SITE}/dsv4_vision_vllm/model.py \
    && python3 /opt/deepseek-v4/test_minp_spec_decode_hotfix.py \
    && python3 -c 'import importlib.metadata as m; assert m.version("vllm") == "0.27.1"; assert m.version("b12x") == "0.15.3"; from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend, map_mxfp4_backend; assert map_mxfp4_backend("flashinfer_b12x") == [Mxfp4MoeBackend.B12X_MXFP4]; import dsv4_vision_vllm.model'

ENV VLLM_REFUSAL_LAMBDA_INIT=0.0

LABEL org.opencontainers.image.title="vllm-deepseek-v4-rank1" \
      org.opencontainers.image.description="vLLM 0.27.1 ARM64 for DeepSeek-V4-Flash-0731 with DSpark, vision and per-request rank-1 projection" \
      org.opencontainers.image.base.name="vllm/vllm-openai:v0.27.1-aarch64-ubuntu2404" \
      org.opencontainers.image.version="v0.27.1-rank1g-vision-dsparkfix-b12x-minp" \
      org.opencontainers.image.source="https://github.com/pocharlies/deepseek-v4-flash-rank1-refusal-projection" \
      ai.e-dani.model="deepseek-ai/DeepSeek-V4-Flash-0731@9e165c30" \
      ai.e-dani.arch="linux/arm64 (sm121)"
