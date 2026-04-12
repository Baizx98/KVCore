"""Single-type KV manager for one homogeneous KV cache specification."""

from __future__ import annotations

from dataclasses import dataclass, field

from kvcore.kv.block_pool import BlockPool
from kvcore.kv.metadata import RequestKVView


@dataclass(slots=True)
class SingleTypeKVManager:
    """Manage request block tables and per-layer views for one KV spec."""

    num_layers: int
    block_size: int
    block_pool: BlockPool
    request_views: dict[str, RequestKVView] = field(default_factory=dict)

    def register_request(self, *, request_id: str, total_tokens: int) -> RequestKVView:
        kv_view = RequestKVView.from_token_count(
            seq_id=request_id,
            total_tokens=total_tokens,
            num_layers=self.num_layers,
            block_size=self.block_size,
            allocator=self.block_pool,
            canonical_tensor_name="continuous_kv_cache",
        )
        self.request_views[request_id] = kv_view
        return kv_view

    def advance_request(self, *, request_id: str, total_tokens: int) -> RequestKVView:
        kv_view = self.request_views[request_id]
        kv_view.update_total_tokens(total_tokens, self.block_pool)
        return kv_view

    def get_request_view(self, request_id: str) -> RequestKVView:
        return self.request_views[request_id]

    def release_request(self, request_id: str) -> None:
        kv_view = self.request_views.pop(request_id)
        kv_view.release(self.block_pool)
