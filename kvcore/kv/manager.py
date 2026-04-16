"""Top-level KV manager without a coordinator layer."""

from __future__ import annotations

from dataclasses import dataclass, field

from kvcore.kv.block_pool import BlockPool
from kvcore.kv.metadata import CanonicalKVRef, CanonicalLayerKVState, RequestKVView
from kvcore.kv.single_type_manager import SingleTypeKVManager


@dataclass(slots=True)
class KVManager:
    """Own one SingleTypeKVManager per layer and aggregate request views."""

    num_layers: int
    block_size: int
    device: str
    block_pool: BlockPool
    layer_managers: dict[int, SingleTypeKVManager]
    request_views: dict[str, RequestKVView] = field(default_factory=dict)

    @classmethod
    def from_model_config(cls, *, num_layers: int, block_size: int, device: str) -> KVManager:
        block_pool = BlockPool()
        layer_managers = {
            layer_id: SingleTypeKVManager(
                layer_id=layer_id,
                block_size=block_size,
                block_pool=block_pool,
            )
            for layer_id in range(num_layers)
        }
        return cls(
            num_layers=num_layers,
            block_size=block_size,
            device=device,
            block_pool=block_pool,
            layer_managers=layer_managers,
        )

    def register_request(self, *, request_id: str, total_tokens: int) -> RequestKVView:
        layer_states = {
            layer_id: manager.register_request(
                request_id=request_id,
                total_tokens=total_tokens,
            )
            for layer_id, manager in self.layer_managers.items()
        }
        kv_view = RequestKVView(
            seq_id=request_id,
            total_tokens=total_tokens,
            block_size=self.block_size,
            layer_states=layer_states,
            canonical_layer_states=self._build_canonical_layer_states(),
        )
        self.request_views[request_id] = kv_view
        return kv_view

    def advance_request(self, *, request_id: str, total_tokens: int) -> RequestKVView:
        kv_view = self.request_views[request_id]
        if total_tokens < kv_view.total_tokens:
            raise ValueError("total_tokens must not decrease")
        if total_tokens == kv_view.total_tokens:
            return kv_view

        for layer_id, manager in self.layer_managers.items():
            kv_view.layer_states[layer_id] = manager.advance_request(
                request_id=request_id,
                total_tokens=total_tokens,
            )
        kv_view.total_tokens = total_tokens
        return kv_view

    def get_request_view(self, request_id: str) -> RequestKVView:
        return self.request_views[request_id]

    def cache_request_blocks(
        self,
        *,
        request_id: str,
        token_ids: list[int],
        extra_keys: tuple[object, ...] = (),
    ) -> None:
        for manager in self.layer_managers.values():
            manager.cache_blocks(
                request_id=request_id,
                token_ids=token_ids,
                extra_keys=extra_keys,
            )

    def find_longest_cache_hit(
        self,
        *,
        token_ids: list[int],
        max_length: int,
        extra_keys: tuple[object, ...] = (),
    ) -> int:
        if not self.layer_managers:
            return 0
        hits = [
            manager.find_longest_cache_hit(
                token_ids=token_ids,
                max_length=max_length,
                extra_keys=extra_keys,
            )
            for manager in self.layer_managers.values()
        ]
        return min(hits)

    def release_request(self, request_id: str) -> None:
        for manager in self.layer_managers.values():
            manager.release_request(request_id)
        kv_view = self.request_views.pop(request_id)
        kv_view.total_tokens = 0
        for state in kv_view.layer_states.values():
            state.blocks.clear()

    def _build_canonical_layer_states(self) -> dict[int, CanonicalLayerKVState]:
        return {
            layer_id: CanonicalLayerKVState(
                layer_id=layer_id,
                ref=CanonicalKVRef(
                    tensor_name=f"layer_{layer_id}_kv_cache",
                    layer_id=layer_id,
                    block_size_factor=1,
                ),
            )
            for layer_id in self.layer_managers
        }
