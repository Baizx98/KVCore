from __future__ import annotations

import torch

from kvcore.sample import LogitsProcessor, Sampler
from kvcore.sample.sampler import apply_top_k_top_p
from kvcore.utils.sampling_params import SamplingParams


def test_logits_processor_projects_hidden_states() -> None:
    lm_head = torch.nn.Linear(2, 3, bias=False)
    lm_head.weight.data.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    hidden_states = torch.tensor([[2.0, 3.0]])

    logits = LogitsProcessor(scale=0.5)(hidden_states, lm_head)

    assert torch.allclose(logits, torch.tensor([[1.0, 1.5, 2.5]]))


def test_logits_processor_can_use_logits_as_input() -> None:
    logits = torch.tensor([[2.0, -2.0]])
    processor = LogitsProcessor(logits_as_input=True, soft_cap=1.0)

    processed = processor(logits)

    assert torch.all(processed.abs() <= 1.0)


def test_sampler_greedy_uses_argmax() -> None:
    logits = torch.tensor([[0.0, 2.0, 1.0], [4.0, 3.0, 2.0]])
    sampler = Sampler()

    output = sampler(logits, SamplingParams(max_tokens=1, temperature=0.0))

    assert output.sampled_token_ids.tolist() == [[1], [0]]


def test_sampler_seeded_random_sampling_is_reproducible() -> None:
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    sampler = Sampler()
    params = SamplingParams(max_tokens=1, seed=123)

    first = sampler.sample(logits, params)
    second = sampler.sample(logits, params)

    assert first.tolist() == second.tolist()


def test_top_k_masks_tokens_outside_the_k_best() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    filtered = apply_top_k_top_p(logits, torch.tensor([2]), None)

    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 1])
    assert torch.isfinite(filtered[0, 2])
    assert torch.isfinite(filtered[0, 3])


def test_top_p_keeps_minimal_high_probability_prefix() -> None:
    logits = torch.tensor([[10.0, 9.0, 0.0, -1.0]])

    filtered = apply_top_k_top_p(logits, None, torch.tensor([0.8]))

    assert torch.isfinite(filtered[0, 0])
    assert torch.isfinite(filtered[0, 1])
    assert torch.isneginf(filtered[0, 2])
    assert torch.isneginf(filtered[0, 3])


def test_sampler_accepts_per_row_sampling_params() -> None:
    logits = torch.tensor([[0.0, 4.0], [5.0, 1.0]])
    sampler = Sampler()
    params = [
        SamplingParams(max_tokens=1, temperature=0.0),
        SamplingParams(max_tokens=1, top_k=1, seed=7),
    ]

    sampled = sampler.sample(logits, params)

    assert sampled.tolist() == [1, 0]
