from types import SimpleNamespace

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor
from vllm.v1.sample.rejection_sampler import RejectionSampler


def test_validator_allows_min_p_and_keeps_logit_bias_blocked():
    SamplingParams(min_p=0.01)._validate_spec_decode(object())

    try:
        SamplingParams(logit_bias={1: 1.0})._validate_spec_decode(object())
    except ValueError as error:
        assert "logit_bias" in str(error)
    else:
        raise AssertionError("logit_bias must remain blocked under spec decode")


def test_min_p_matches_reference_for_mixed_request_rows():
    processor = object.__new__(MinPLogitsProcessor)
    processor.min_p_count = 2
    processor.min_p = torch.tensor([[0.01], [0.20]])
    logits = torch.tensor(
        [[4.0, 1.0, 0.0], [2.0, 1.0, -2.0], [2.0, 0.0, -1.0]],
        dtype=torch.float32,
    )

    expected = logits.clone()
    probabilities = torch.softmax(expected, dim=-1)
    thresholds = probabilities.amax(dim=-1, keepdim=True) * torch.tensor(
        [[0.01], [0.01], [0.20]]
    )
    expected.masked_fill_(probabilities < thresholds, -float("inf"))

    actual = processor.apply_with_spec_decode(logits.clone(), [2, 1])
    assert torch.equal(actual, expected)


def test_rejection_sampler_applies_min_p_after_existing_processors():
    processor = object.__new__(MinPLogitsProcessor)
    processor.min_p_count = 1
    processor.min_p = torch.tensor([[0.20]])
    metadata = SimpleNamespace(num_draft_tokens=[1])
    sampling_metadata = SimpleNamespace(
        no_penalties=True,
        bad_words_token_ids=None,
        thinking_budget_state_holder=None,
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        spec_token_ids=None,
        logitsprocs=SimpleNamespace(
            non_argmax_invariant=[], argmax_invariant=[processor]
        ),
    )
    sampler = object.__new__(RejectionSampler)
    logits = torch.tensor([[2.0, 0.0, -1.0]])

    actual = sampler.apply_logits_processors(logits, sampling_metadata, metadata)
    assert torch.isfinite(actual[0, 0])
    assert torch.isneginf(actual[0, 1:]).all()


if __name__ == "__main__":
    test_validator_allows_min_p_and_keeps_logit_bias_blocked()
    test_min_p_matches_reference_for_mixed_request_rows()
    test_rejection_sampler_applies_min_p_after_existing_processors()
    print("3 min_p speculative-decoding hotfix tests passed")
