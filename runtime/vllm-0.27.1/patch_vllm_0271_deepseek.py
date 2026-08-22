#!/usr/bin/env python3
"""Port the production DeepSeek V4 hooks to the exact vLLM 0.27.1 tree.

The base image already contains the generic, tested vLLM 0.27.1 rank-1 request
plumbing used by Qwen38.  This fail-closed patch adds only the DeepSeek attention
hook and changes the draft-buffer filling from native MTP to DSpark's layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace(path: Path, old: str, new: str, *, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise SystemExit(
            f"[deepseek-v4-vllm-0.27.1] {path}: anchor count {found}, expected {count}"
        )
    path.write_text(text.replace(old, new, count), encoding="utf-8")


def patch_attention(site: Path) -> None:
    path = site / "models/deepseek_v4/attention.py"
    replace(
        path,
        """from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.triton_utils import tl, triton
""",
        """from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.refusal_projection import (
    RefusalProjection,
    is_enabled as refusal_is_enabled,
    resolve as refusal_resolve,
)
from vllm.triton_utils import tl, triton
""",
    )
    replace(
        path,
        """        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

        # Initialize rotary embedding before the indexer/compressor consume it.
""",
        """        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

        self.refusal_proj: nn.Module | None = None
        if refusal_is_enabled():
            direction, role = refusal_resolve(prefix, config.num_hidden_layers)
            if direction is not None:
                self.refusal_proj = RefusalProjection(direction, role)

        # Initialize rotary embedding before the indexer/compressor consume it.
""",
    )
    replace(
        path,
        """        # Inverse-RoPE + wo_a + wo_b output projection (platform-specific).
        return self._o_proj(o, positions)
""",
        """        # Inverse-RoPE + wo_a + wo_b output projection (platform-specific).
        out = self._o_proj(o, positions)
        if self.refusal_proj is not None:
            out = self.refusal_proj(out)
        return out
""",
    )


def patch_dspark_buffers(site: Path) -> None:
    path = site / "v1/worker/gpu/model_runner.py"
    replace(
        path,
        """                self.refusal_state.fill_draft_neutral(global_lambda)
""",
        """                # DSpark's draft rows are filled immediately before propose().
""",
    )
    replace(
        path,
        """            draft_tokens = self.speculator.propose(
""",
        """            if refusal_projection.is_enabled():
                num_query_per_req = getattr(
                    self.speculator, "num_query_per_req", None
                )
                if num_query_per_req is not None:
                    self.refusal_state.fill_draft(
                        input_batch.idx_mapping_np,
                        int(num_query_per_req),
                        refusal_projection.get_lambda(),
                    )
            draft_tokens = self.speculator.propose(
""",
    )


def patch_b12x_envs(site: Path) -> None:
    path = site / "envs.py"
    replace(
        path,
        """    VLLM_USE_FLASHINFER_SAMPLER: bool = True
    VLLM_PP_LAYER_PARTITION: str | None = None
""",
        """    VLLM_USE_FLASHINFER_SAMPLER: bool = True
    VLLM_USE_B12X_MOE: bool = False
    VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM: int = 0
    VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M: int = 16
    VLLM_B12X_W4A16_FORCE_TILE_CONFIG: str = ""
    VLLM_PP_LAYER_PARTITION: str | None = None
""",
    )
    replace(
        path,
        """    "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": lambda: int(
        os.getenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    ),
    # If set, the OpenAI API server will stay alive even after the underlying
""",
        """    "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": lambda: int(
        os.getenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    ),
    "VLLM_USE_B12X_MOE": lambda: bool(int(os.getenv("VLLM_USE_B12X_MOE", "0"))),
    "VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM": lambda: int(
        os.getenv("VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM", "0")
    ),
    "VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M": lambda: int(
        os.getenv("VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M", "16")
    ),
    "VLLM_B12X_W4A16_FORCE_TILE_CONFIG": lambda: os.getenv(
        "VLLM_B12X_W4A16_FORCE_TILE_CONFIG", ""
    ),
    # If set, the OpenAI API server will stay alive even after the underlying
""",
    )


def patch_b12x_mxfp4_oracle(site: Path) -> None:
    """Restore Anemll's B12X adapter for DeepSeek-native MXFP4 experts."""

    path = site / "model_executor/layers/fused_moe/oracle/mxfp4.py"
    replace(
        path,
        """    FLASHINFER_CUTLASS_MXFP4_MXFP8 = "FLASHINFER_CUTLASS_MXFP4_MXFP8"
    FLASHINFER_CUTLASS_MXFP4_BF16 = "FLASHINFER_CUTLASS_MXFP4_BF16"
    # Marlin
""",
        """    FLASHINFER_CUTLASS_MXFP4_MXFP8 = "FLASHINFER_CUTLASS_MXFP4_MXFP8"
    FLASHINFER_CUTLASS_MXFP4_BF16 = "FLASHINFER_CUTLASS_MXFP4_BF16"
    # B12X backend for DeepSeek V4 native MXFP4 weights on SM120/GB10
    B12X_MXFP4 = "B12X_MXFP4"
    # Marlin
""",
    )
    replace(
        path,
        """        return [FlashInferExperts]

    elif backend == Mxfp4MoeBackend.TRITON:
""",
        """        return [FlashInferExperts]

    elif backend == Mxfp4MoeBackend.B12X_MXFP4:
        from vllm.model_executor.layers.fused_moe.experts.b12x_mxfp4_moe import (
            B12xExperts,
        )

        return [B12xExperts]

    elif backend == Mxfp4MoeBackend.TRITON:
""",
    )
    replace(
        path,
        """        "flashinfer_cutlass_afp8": [Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8],
        "triton": [Mxfp4MoeBackend.TRITON],
""",
        """        "flashinfer_cutlass_afp8": [Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8],
        "flashinfer_b12x": [Mxfp4MoeBackend.B12X_MXFP4],
        "triton": [Mxfp4MoeBackend.TRITON],
""",
    )
    replace(
        path,
        """    elif backend in (
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
    ):
        intermediate_size = round_up(intermediate_size, 128)
        hidden_size = round_up(hidden_size, 128)
""",
        """    elif backend in (
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
        Mxfp4MoeBackend.B12X_MXFP4,
    ):
        intermediate_size = round_up(intermediate_size, 128)
        hidden_size = round_up(hidden_size, 128)
""",
    )
    replace(
        path,
        """        return (
            w13_weight.data,
            w2_weight.data,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
        )

    if mxfp4_backend == Mxfp4MoeBackend.HUMMING:
""",
        """        return (
            w13_weight.data,
            w2_weight.data,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
        )

    if mxfp4_backend == Mxfp4MoeBackend.B12X_MXFP4:
        # B12X prepares its W4A16 packed representation from the original
        # native MXFP4 tensors during expert post-load setup.
        return (
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
        )

    if mxfp4_backend == Mxfp4MoeBackend.HUMMING:
""",
    )
    replace(
        path,
        """        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
        Mxfp4MoeBackend.AITER_MXFP4_BF16,
""",
        """        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
        Mxfp4MoeBackend.B12X_MXFP4,
        Mxfp4MoeBackend.AITER_MXFP4_BF16,
""",
    )
    replace(
        path,
        """        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            **extra_kwargs,
        )

    kernel = mk.FusedMoEKernel(
""",
        """        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            **extra_kwargs,
        )

    if mxfp4_backend == Mxfp4MoeBackend.B12X_MXFP4:
        assert layer is not None
        experts.process_weights_after_loading(layer)

    kernel = mk.FusedMoEKernel(
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()
    site = args.site.resolve()
    required = (
        site / "models/deepseek_v4/attention.py",
        site / "v1/worker/gpu/model_runner.py",
    )
    if not all(path.exists() for path in required):
        raise SystemExit(f"[deepseek-v4-vllm-0.27.1] unexpected base at {site}")
    patch_b12x_envs(site)
    patch_b12x_mxfp4_oracle(site)
    patch_attention(site)
    patch_dspark_buffers(site)
    print("[deepseek-v4-vllm-0.27.1] all anchors applied")


if __name__ == "__main__":
    main()
