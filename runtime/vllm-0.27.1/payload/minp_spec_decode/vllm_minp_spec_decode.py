"""Backport min-p support for vLLM V1 speculative decoding.

Based on vllm-project/vllm#42802 (commit 1694a211e82ff047924bdfe807ad38b26da8da49).
The upstream PR fixes target-token filtering but intentionally leaves the request
validator locked. This compatibility shim applies that fix and only relaxes the
min_p half of the validator; logit_bias remains rejected.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Callable
from types import ModuleType


_PATCH_TARGETS = {
    "vllm.sampling_params",
    "vllm.v1.sample.logits_processor.builtin",
    "vllm.v1.sample.rejection_sampler",
}


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader, patch: Callable[[ModuleType], None]):
        self.loader = loader
        self.patch = patch

    def create_module(self, spec):
        create_module = getattr(self.loader, "create_module", None)
        return create_module(spec) if create_module else None

    def exec_module(self, module: ModuleType) -> None:
        self.loader.exec_module(module)
        self.patch(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        if fullname not in _PATCH_TARGETS:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None

        patch = {
            "vllm.sampling_params": _patch_sampling_params,
            "vllm.v1.sample.logits_processor.builtin": _patch_builtin,
            "vllm.v1.sample.rejection_sampler": _patch_rejection_sampler,
        }[fullname]
        spec.loader = _PatchLoader(spec.loader, patch)
        return spec


def _patch_sampling_params(module: ModuleType) -> None:
    original = module.SamplingParams._validate_spec_decode

    def validate_spec_decode(self, speculative_config) -> None:
        if speculative_config is None:
            return
        if self.logit_bias:
            raise ValueError(
                "The logit_bias sampling parameter is not yet supported "
                "with speculative decoding."
            )

    validate_spec_decode.__name__ = original.__name__
    validate_spec_decode.__qualname__ = original.__qualname__
    module.SamplingParams._validate_spec_decode = validate_spec_decode


def _patch_builtin(module: ModuleType) -> None:
    import torch

    def apply_with_spec_decode(self, logits, num_draft_tokens):
        if not self.min_p_count:
            return logits

        counts = torch.tensor(num_draft_tokens, device="cpu")
        request_indices = torch.arange(len(num_draft_tokens), device="cpu")
        row_indices = request_indices.repeat_interleave(counts).to(
            device=logits.device, non_blocking=True
        )
        min_p_per_row = self.min_p[row_indices]

        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        max_probabilities = torch.amax(probabilities, dim=-1, keepdim=True)
        threshold = max_probabilities.mul_(min_p_per_row)
        logits.masked_fill_(probabilities < threshold, -float("inf"))
        return logits

    module.MinPLogitsProcessor.apply_with_spec_decode = apply_with_spec_decode


def _patch_rejection_sampler(module: ModuleType) -> None:
    original = module.RejectionSampler.apply_logits_processors

    def apply_logits_processors(self, logits, sampling_metadata, metadata):
        logits = original(self, logits, sampling_metadata, metadata)
        min_p_type = sys.modules[
            "vllm.v1.sample.logits_processor.builtin"
        ].MinPLogitsProcessor
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            if isinstance(processor, min_p_type):
                logits = processor.apply_with_spec_decode(
                    logits, metadata.num_draft_tokens
                )
        return logits

    module.RejectionSampler.apply_logits_processors = apply_logits_processors


sys.meta_path.insert(0, _PatchFinder())
