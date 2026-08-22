#!/usr/bin/env python3
"""Adapt the legacy DSV4 vision processor to vLLM 0.27.1 prompt updates."""

from pathlib import Path
import sys


if len(sys.argv) != 2:
    raise SystemExit("usage: patch_dsv4_vision_0271.py <dsv4_vision_vllm/model.py>")

path = Path(sys.argv[1])
source = path.read_text()
marker = "    def _hf_processor_applies_updates(\n"
if marker in source:
    raise SystemExit("vision compatibility override already present")

anchor = """class DSV4VisionMultiModalProcessor(
    BaseMultiModalProcessor[DSV4VisionProcessingInfo]
):
    def _call_hf_processor(
"""
replacement = """class DSV4VisionMultiModalProcessor(
    BaseMultiModalProcessor[DSV4VisionProcessingInfo]
):
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # This custom processor tokenizes the input placeholder unchanged.
        # vLLM must apply `_get_prompt_updates` after `_call_hf_processor`.
        return False

    def _call_hf_processor(
"""
if source.count(anchor) != 1:
    raise SystemExit("DSV4 vision processor anchor not found exactly once")

path.write_text(source.replace(anchor, replacement))
