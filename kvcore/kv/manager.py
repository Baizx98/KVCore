"""Top-level KV manager."""

from __future__ import annotations

from dataclasses import dataclass

from kvcore.kv.block_pool import BlockPool
from kvcore.kv.single_type_manager import SingleTypeKVManager


@dataclass(slots=True)
class KVManager:
    """Top-level KV owner coordinating block pool and per-type managers."""

    num_layers: int
    block_size: int
    device: str
    block_pool: BlockPool
    single_type_manager: SingleTypeKVManager

    @classmethod
    def from_model_config(cls, *, num_layers: int, block_size: int, device: str) -> KVManager:
        block_pool = BlockPool()
        single_type_manager = SingleTypeKVManager(
            num_layers=num_layers,
            block_size=block_size,
            block_pool=block_pool,
        )
        return cls(
            num_layers=num_layers,
            block_size=block_size,
            device=device,
            block_pool=block_pool,
            single_type_manager=single_type_manager,
        )

    def register_request(self, *, request_id: str, total_tokens: int):
        return self.single_type_manager.register_request(
            request_id=request_id,
            total_tokens=total_tokens,
        )

    def advance_request(self, *, request_id: str, total_tokens: int):
        return self.single_type_manager.advance_request(
            request_id=request_id,
            total_tokens=total_tokens,
        )

    def get_request_view(self, request_id: str):
        return self.single_type_manager.get_request_view(request_id)

    def release_request(self, request_id: str) -> None:
        self.single_type_manager.release_request(request_id)
