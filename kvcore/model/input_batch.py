from __future__ import annotations

from dataclasses import dataclass

import torch

from kvcore.kv.block_table import MultiGroupBlockTable
from kvcore.utils.log import get_logger
from kvcore.utils.sampling_params import SamplingParams

logger = get_logger(__name__)


@dataclass(slots=True)
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    block_ids: tuple[tuple[int, ...], ...]
    num_computed_tokens: int
    token_ids: list[int] | None = None
    num_prompt_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.token_ids is None:
            self.token_ids = list(self.prompt_token_ids)
        if self.num_prompt_tokens is None:
            self.num_prompt_tokens = len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        assert self.token_ids is not None
        return len(self.token_ids)


class InputBatch:
    """Persistent runner-side request batch.

    This is intentionally shaped like vLLM v1's InputBatch: requests occupy
    stable rows, token/computed-token state lives in staged CPU buffers, and
    per-step model inputs are prepared by ModelRunner from these rows.
    """

    def __init__(
        self,
        *,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        block_sizes: list[int],
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.pin_memory = pin_memory and torch.cuda.is_available()

        self._req_ids: list[str | None] = []
        self.req_id_to_index: dict[str, int] = {}
        self.requests: list[CachedRequestState | None] = []

        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            dtype=torch.long,
            device="cpu",
            pin_memory=False,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        self.num_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.num_tokens = self.num_tokens_cpu_tensor.numpy()
        self.num_prompt_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.num_prompt_tokens = self.num_prompt_tokens_cpu_tensor.numpy()
        self.num_computed_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.num_computed_tokens_cpu = self.num_computed_tokens_cpu_tensor.numpy()

        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=self.pin_memory,
            device=device,
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
        )

    @property
    def req_ids(self) -> list[str]:
        return [req_id for req_id in self._req_ids if req_id is not None]

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    def add_request(self, request: CachedRequestState) -> int:
        if request.req_id in self.req_id_to_index:
            req_index = self.req_id_to_index[request.req_id]
        else:
            if len(self._req_ids) >= self.max_num_reqs:
                raise ValueError(
                    f"InputBatch capacity exceeded: max_num_reqs={self.max_num_reqs}"
                )
            req_index = len(self._req_ids)
            self._req_ids.append(request.req_id)
            self.requests.append(None)
            self.req_id_to_index[request.req_id] = req_index

        self._req_ids[req_index] = request.req_id
        self.requests[req_index] = request
        self._write_request_row(req_index, request)
        self.block_table.add_row(_to_block_table_row(request.block_ids), req_index)
        logger.debug(
            "InputBatch added request req_id=%s row=%d prompt_tokens=%d",
            request.req_id,
            req_index,
            request.num_prompt_tokens,
        )
        return req_index

    def update_cached_request(
        self,
        *,
        req_id: str,
        new_token_ids: tuple[int, ...],
        block_ids: tuple[tuple[int, ...], ...],
        num_computed_tokens: int,
    ) -> int:
        req_index = self.req_id_to_index.get(req_id)
        if req_index is None:
            raise KeyError(f"Request {req_id!r} is not present in InputBatch")
        request = self.requests[req_index]
        if request is None:
            raise KeyError(f"Request {req_id!r} is not present in InputBatch")

        start = int(self.num_tokens[req_index])
        if num_computed_tokens >= start:
            end = start + len(new_token_ids)
            self._check_token_capacity(req_id, end)
            if new_token_ids:
                self.token_ids_cpu[req_index, start:end] = new_token_ids
                self.num_tokens[req_index] = end
                assert request.token_ids is not None
                request.token_ids.extend(new_token_ids)
        request.block_ids = block_ids
        request.num_computed_tokens = num_computed_tokens
        self.num_computed_tokens_cpu[req_index] = num_computed_tokens
        self.block_table.add_row(_to_block_table_row(block_ids), req_index)
        logger.debug(
            "InputBatch updated cached request req_id=%s row=%d new_tokens=%d "
            "num_computed_tokens=%d",
            req_id,
            req_index,
            len(new_token_ids),
            num_computed_tokens,
        )
        return req_index

    def remove_request(self, req_id: str) -> int | None:
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self._req_ids[req_index] = None
        self.requests[req_index] = None
        self._clear_row(req_index)
        return req_index

    def remove_requests(self, req_ids: tuple[str, ...] | list[str]) -> None:
        removed = 0
        for req_id in req_ids:
            if self.remove_request(req_id) is not None:
                removed += 1
        if removed:
            self.condense()
            logger.debug("InputBatch removed requests count=%d", removed)

    def condense(self) -> None:
        active: list[tuple[int, CachedRequestState]] = [
            (index, request)
            for index, request in enumerate(self.requests)
            if request is not None
        ]
        for new_index, (old_index, request) in enumerate(active):
            if new_index != old_index:
                self._move_row(old_index, new_index)
                self.block_table.move_row(old_index, new_index)
            self._req_ids[new_index] = request.req_id
            self.requests[new_index] = request

        active_count = len(active)
        for index in range(active_count, len(self.requests)):
            self._clear_row(index)
        del self._req_ids[active_count:]
        del self.requests[active_count:]
        self.req_id_to_index = {
            request.req_id: index for index, request in enumerate(self.requests)
        }

    def record_sampled_tokens(
        self,
        req_ids: tuple[str, ...],
        token_ids: tuple[int, ...],
    ) -> None:
        for req_id, token_id in zip(req_ids, token_ids, strict=True):
            req_index = self.req_id_to_index[req_id]
            start = int(self.num_tokens[req_index])
            end = start + 1
            self._check_token_capacity(req_id, end)
            self.token_ids_cpu[req_index, start] = token_id
            self.num_tokens[req_index] = end
            request = self.requests[req_index]
            if request is not None:
                assert request.token_ids is not None
                request.token_ids.append(token_id)
        if req_ids:
            logger.debug("InputBatch recorded sampled tokens count=%d", len(req_ids))

    def get_request(self, req_id: str) -> CachedRequestState:
        try:
            req_index = self.req_id_to_index[req_id]
            request = self.requests[req_index]
        except KeyError as exc:
            raise KeyError(f"Request {req_id!r} is not present in InputBatch") from exc
        if request is None:
            raise KeyError(f"Request {req_id!r} is not present in InputBatch")
        return request

    def _write_request_row(self, req_index: int, request: CachedRequestState) -> None:
        num_tokens = request.num_tokens
        num_prompt_tokens = int(request.num_prompt_tokens or 0)
        self._check_token_capacity(request.req_id, num_tokens)
        if num_tokens:
            assert request.token_ids is not None
            self.token_ids_cpu[req_index, :num_tokens] = request.token_ids
        self.num_tokens[req_index] = num_tokens
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens

    def _move_row(self, old_index: int, new_index: int) -> None:
        num_tokens = int(self.num_tokens[old_index])
        if num_tokens:
            self.token_ids_cpu[new_index, :num_tokens] = self.token_ids_cpu[
                old_index,
                :num_tokens,
            ]
        self.num_tokens[new_index] = self.num_tokens[old_index]
        self.num_prompt_tokens[new_index] = self.num_prompt_tokens[old_index]
        self.num_computed_tokens_cpu[new_index] = self.num_computed_tokens_cpu[
            old_index
        ]

    def _clear_row(self, req_index: int) -> None:
        num_tokens = int(self.num_tokens[req_index])
        if num_tokens:
            self.token_ids_cpu[req_index, :num_tokens] = 0
        self.num_tokens[req_index] = 0
        self.num_prompt_tokens[req_index] = 0
        self.num_computed_tokens_cpu[req_index] = 0
        self.block_table.clear_row(req_index)

    def _check_token_capacity(self, req_id: str, num_tokens: int) -> None:
        if num_tokens > self.max_model_len:
            raise ValueError(
                f"Request {req_id!r} has {num_tokens} tokens, "
                f"but InputBatch max_model_len is {self.max_model_len}"
            )


def _to_block_table_row(
    block_ids: tuple[tuple[int, ...], ...],
) -> tuple[list[int], ...]:
    return tuple(list(layer_block_ids) for layer_block_ids in block_ids)


__all__ = [
    "CachedRequestState",
    "InputBatch",
]
