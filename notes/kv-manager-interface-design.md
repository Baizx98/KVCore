# KVManager Interface Design

Date: 2026-04-22 23:10:38 CST

## Problem Statement

KVCore needs a vLLM-inspired KV management boundary that keeps block lifecycle
logic separate from model execution. The key design choice is:

- `KVManager` owns request/layer logical block tables and KV block allocation.
- `ModelRunner` owns KV cache tensors, slot mappings, attention metadata, and
  model/backend inputs.

This mirrors vLLM's split where core KV managers decide block allocation, while
GPU model runners initialize KV tensors and build attention metadata.

## Design Summary

`KVManager` removes vLLM's coordinator layer. Instead, it owns one
`SingleTypeKVManager` per model layer, with all layers sharing one global
`BlockPool`. The first runnable path supports full attention. Sliding-window
attention is represented in the type system and manager factory, but cache-hit
logic is intentionally left for later work.

The public KV lifecycle API is intentionally tensor-free:

- `get_computed_blocks(request)`
- `can_fit(request, num_new_tokens, new_computed_blocks=None)`
- `allocate_slots(request, num_new_tokens, new_computed_blocks=None)`
- `cache_blocks(request, num_computed_tokens)`
- `free(request)`
- `get_blocks(request_id)`
- `get_block_ids(request_id)`
- `take_new_block_ids()`

`KVCacheBlocks.blocks[layer_idx][logical_block_idx]` is the handoff object used
by `ModelRunner` to build runtime metadata.

## Permanent Eviction

Permanent eviction is a KV lifecycle operation and belongs in `KVManager`.

The API uses logical block indices instead of physical block ids:

```python
LayerBlockSelection(
    request_id="req",
    layer_idx=0,
    block_indices={0, 3, 7},
)
```

For each selected logical block, `SingleTypeKVManager` releases the request's
reference to the physical `KVBlock`, replaces that position with `null_block`,
and records the index in `permanently_evicted_blocks`. The operation is local
to the selected request and layer. It does not globally remove prefix-cache
entries for the same physical block, because other requests may still share it.

Repeated, already-null, or out-of-range indices are skipped and reported in
`EvictionResult`.

## Compute Sparsity

Compute sparsity is not a KV lifecycle operation. It is a per-forward metadata
choice handled by `ModelRunner`.

`SparseComputePlan` accepts the same logical block selection shape, but it only
filters active blocks while building block tables and slot mappings. It does not
modify `KVManager`, physical KV tensors, ref counts, or prefix-cache state.

This keeps the two mechanisms separate:

- Permanent eviction changes future logical block tables.
- Compute sparsity only changes the current forward's attention metadata.

## ModelRunner Responsibilities

`ModelRunner.initialize_kv_cache()` now creates the `KVManager`, initializes KV
cache tensors, wraps each tensor in a lightweight runtime cache object, and binds
those cache objects to model `Attention` modules.

The first tensor layout is:

```text
[2, num_blocks, block_size, num_kv_heads, head_size]
```

`ModelRunner.build_attention_metadata(request_ids, sparse_plan=None)` reads
`KVManager.get_blocks()` and emits:

- vLLM-style `MultiGroupBlockTable`
- per-layer slot mappings
- the optional sparse plan used for this forward

This method is the only place where logical KV blocks become model/backend
runtime inputs.

## BlockTable Boundary

KVCore aligns the public `BlockTable` shape with vLLM's
`vllm/v1/worker/block_table.py`: it is a runner-side CPU/GPU staged table with
row operations (`append_row`, `add_row`, `clear_row`, `move_row`, `swap_row`),
explicit `commit_block_table`, and `compute_slot_mapping`.

vLLM's `vllm/v1/worker/gpu/block_table.py` serves a more GPU-specialized role:
it stores multiple group tables with staged writes/UVA-backed metadata and
gather kernels for GPU input batches. KVCore does not need that path yet, but
the current `MultiGroupBlockTable` naming and group-oriented shape leave room
for that optimization later.

`BlockTable` is intentionally owned by `ModelRunner`, not `KVManager`.
`KVManager` only produces logical `KVBlock` lists and block ids.

## Validation Plan

Current tests cover:

- shared global block pool allocation across multiple layers
- permanent eviction of random non-contiguous logical blocks
- idempotent repeated eviction and skipped out-of-range indices
- ModelRunner KV tensor initialization and binding
- sparse compute metadata filtering without mutating KVManager
- dense metadata recovery when no sparse plan is passed

## Limits

The current implementation deliberately omits:

- real paged KV tensor write/read kernels
- sliding-window cache-hit logic
- external KV transfer
- speculative decoding and lookahead slots
- prefix cache events
