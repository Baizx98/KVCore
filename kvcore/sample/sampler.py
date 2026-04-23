from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from kvcore.utils.sampling_params import SamplingParams

_SAMPLING_EPS = 1e-5


@dataclass(frozen=True, slots=True)
class SamplerOutput:
    sampled_token_ids: torch.Tensor


@dataclass(frozen=True, slots=True)
class SamplingMetadata:
    temperatures: torch.Tensor
    top_p: torch.Tensor | None
    top_k: torch.Tensor | None
    generators: dict[int, torch.Generator]
    all_greedy: bool
    all_random: bool

    @classmethod
    def from_params(
        cls,
        sampling_params: SamplingParams | Sequence[SamplingParams],
        *,
        batch_size: int,
        vocab_size: int,
        device: torch.device,
    ) -> SamplingMetadata:
        params = _normalize_params(sampling_params, batch_size)
        temperatures = torch.tensor(
            [p.temperature for p in params],
            dtype=torch.float32,
            device=device,
        )

        top_k_values = [vocab_size if p.top_k is None else min(p.top_k, vocab_size) for p in params]
        top_p_values = [p.top_p for p in params]
        top_k = None
        top_p = None
        if any(value < vocab_size for value in top_k_values):
            top_k = torch.tensor(top_k_values, dtype=torch.long, device=device)
        if any(value < 1.0 for value in top_p_values):
            top_p = torch.tensor(top_p_values, dtype=torch.float32, device=device)

        generators = {
            row_idx: _make_generator(p.seed, device)
            for row_idx, p in enumerate(params)
            if p.seed is not None
        }
        greedy_mask = temperatures < _SAMPLING_EPS
        return cls(
            temperatures=temperatures,
            top_p=top_p,
            top_k=top_k,
            generators=generators,
            all_greedy=bool(greedy_mask.all().item()),
            all_random=bool((~greedy_mask).all().item()),
        )


class Sampler(nn.Module):
    """Sample next tokens from logits.

    Supported first-version features:
    greedy sampling, random sampling, temperature, top-k, and top-p. More
    specialized vLLM branches such as speculative decoding, penalties, bad
    words, and logprobs are intentionally left out for now.
    """

    def forward(
        self,
        logits: torch.Tensor,
        sampling_params: SamplingParams | Sequence[SamplingParams],
    ) -> SamplerOutput:
        sampled = self.sample(logits, sampling_params)
        return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1).to(torch.int32))

    def sample(
        self,
        logits: torch.Tensor,
        sampling_params: SamplingParams | Sequence[SamplingParams],
    ) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"logits must be rank-2 [batch, vocab], got {tuple(logits.shape)}")

        logits = logits.to(torch.float32)
        metadata = SamplingMetadata.from_params(
            sampling_params,
            batch_size=logits.shape[0],
            vocab_size=logits.shape[1],
            device=logits.device,
        )

        greedy_sampled = None if metadata.all_random else greedy_sample(logits)
        if metadata.all_greedy:
            assert greedy_sampled is not None
            return greedy_sampled

        logits = apply_temperature(logits, metadata.temperatures, metadata.all_random)
        logits = apply_top_k_top_p(logits, metadata.top_k, metadata.top_p)
        random_sampled = random_sample(
            logits.softmax(dim=-1, dtype=torch.float32),
            metadata.generators,
        )
        if greedy_sampled is None:
            return random_sampled
        return torch.where(metadata.temperatures < _SAMPLING_EPS, greedy_sampled, random_sampled)


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1).view(-1)


def apply_temperature(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    all_random: bool,
) -> torch.Tensor:
    if not all_random:
        temperature = torch.where(temperature < _SAMPLING_EPS, 1.0, temperature)
    return logits.div(temperature.unsqueeze(dim=1))


def apply_top_k_top_p(
    logits: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
) -> torch.Tensor:
    if top_k is None and top_p is None:
        return logits

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if top_k is not None:
        top_k = top_k.clamp(max=logits.shape[-1]).to(torch.long)
        top_k_threshold_idx = logits_sort.size(1) - top_k
        top_k_threshold = logits_sort.gather(1, top_k_threshold_idx.unsqueeze(dim=1))
        logits_sort = logits_sort.masked_fill(logits_sort < top_k_threshold, -float("inf"))

    if top_p is not None:
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = probs_sum <= 1 - top_p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        logits_sort = logits_sort.masked_fill(top_p_mask, -float("inf"))

    return logits.scatter(dim=-1, index=logits_idx, src=logits_sort)


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    q = torch.empty_like(probs)
    if len(generators) != probs.shape[0]:
        q.exponential_()
    for row_idx, generator in generators.items():
        q[row_idx].exponential_(generator=generator)
    return probs.div(q).argmax(dim=-1).view(-1)


def _normalize_params(
    sampling_params: SamplingParams | Sequence[SamplingParams],
    batch_size: int,
) -> list[SamplingParams]:
    if isinstance(sampling_params, SamplingParams):
        return [sampling_params] * batch_size
    params = list(sampling_params)
    if len(params) != batch_size:
        raise ValueError(
            "sampling_params length must match logits batch size, "
            f"got {len(params)} params for batch size {batch_size}"
        )
    return params


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


__all__ = [
    "Sampler",
    "SamplerOutput",
    "SamplingMetadata",
    "apply_temperature",
    "apply_top_k_top_p",
    "greedy_sample",
    "random_sample",
]
