from __future__ import annotations

import torch

from kvcore.config import KVCoreConfig, ModelConfig, SchedulerConfig
from kvcore.kv.kv_manager import KVManagerConfig
from kvcore.kv.kv_metrics import KVCacheMetricsCollector
from kvcore.kv.single_type_kv_manager import KVLayerSpec
from kvcore.sched.scheduler import Scheduler
from kvcore.utils.request import Request, RequestStatus
from kvcore.utils.sampling_params import SamplingParams


def make_kv_config() -> KVManagerConfig:
    return KVManagerConfig(
        num_gpu_blocks=16,
        max_model_len=16,
        layer_specs=(
            KVLayerSpec(
                layer_idx=0,
                block_size=2,
                num_kv_heads=2,
                head_size=8,
                dtype=torch.float16,
            ),
        ),
    )


def make_config(scheduler_config: SchedulerConfig | None = None) -> KVCoreConfig:
    return KVCoreConfig(
        model_config=ModelConfig(model="unused", attn_backend="torch_paged"),
        scheduler_config=scheduler_config or SchedulerConfig(),
    )


def make_request(request_id: str, token_ids: list[int], max_tokens: int = 2) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )


def test_scheduler_initializes_and_owns_kv_manager() -> None:
    scheduler = Scheduler(make_config(), make_kv_config())

    assert scheduler.kv_manager.config.num_gpu_blocks == 16
    assert scheduler.kv_manager.num_layers == 1


def test_scheduler_passes_metrics_collector_to_kv_manager() -> None:
    metrics_collector = KVCacheMetricsCollector(sample_rate=1.0)
    scheduler = Scheduler(make_config(), make_kv_config(), metrics_collector=metrics_collector)

    assert scheduler.kv_manager.block_pool.metrics_collector is metrics_collector


def test_scheduler_chunked_prefill_emits_new_request_data() -> None:
    scheduler = Scheduler(
        make_config(
            SchedulerConfig(
                max_num_seqs=2,
                max_num_scheduled_tokens=4,
                max_num_partial_prefills=2,
            )
        ),
        make_kv_config(),
    )
    request = make_request("req", [1, 2, 3, 4, 5, 6], max_tokens=1)
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 4
    assert output.num_scheduled_tokens == {"req": 4}
    assert output.scheduled_new_reqs[0].req_id == "req"
    assert output.scheduled_new_reqs[0].prompt_token_ids == (1, 2, 3, 4, 5, 6)
    assert output.scheduled_new_reqs[0].num_computed_tokens == 0
    assert output.scheduled_cached_reqs.req_ids == ()
    assert output.new_block_ids_to_zero


def test_scheduler_chunked_prefill_continuation_samples_on_last_chunk() -> None:
    scheduler = Scheduler(
        make_config(SchedulerConfig(max_num_seqs=1, max_num_scheduled_tokens=4)),
        make_kv_config(),
    )
    request = make_request("req", [1, 2, 3, 4, 5, 6], max_tokens=2)
    scheduler.add_request(request)

    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"req": 4}
    scheduler.update_from_outputs(first, {})

    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {"req": 2}
    assert second.scheduled_new_reqs == ()
    assert second.scheduled_cached_reqs.req_ids == ("req",)
    assert second.scheduled_cached_reqs.new_token_ids[0] == (5, 6)
    assert second.scheduled_cached_reqs.num_computed_tokens[0] == 4
    scheduler.update_from_outputs(second, {"req": 9})

    third = scheduler.schedule()
    assert third.num_scheduled_tokens == {"req": 1}
    assert third.scheduled_cached_reqs.new_token_ids[0] == (9,)
    assert third.scheduled_cached_reqs.num_output_tokens[0] == 1


def test_scheduler_limits_partial_prefill_continuations() -> None:
    scheduler = Scheduler(
        make_config(
            SchedulerConfig(
                max_num_seqs=2,
                max_num_scheduled_tokens=4,
                max_num_partial_prefills=1,
            )
        ),
        make_kv_config(),
    )
    scheduler.add_request(make_request("req-1", [1, 2, 3, 4, 5, 6], max_tokens=1))
    scheduler.add_request(make_request("req-2", [7, 8, 9, 10, 11, 12], max_tokens=1))

    output = scheduler.schedule()

    assert tuple(output.num_scheduled_tokens) == ("req-1",)


def test_scheduler_does_not_reuse_prefix_blocks_within_same_step() -> None:
    scheduler = Scheduler(
        make_config(
            SchedulerConfig(
                max_num_seqs=2,
                max_num_scheduled_tokens=8,
                max_num_partial_prefills=2,
            )
        ),
        make_kv_config(),
    )
    scheduler.add_request(make_request("req-1", [1, 2, 3, 4], max_tokens=1))
    scheduler.add_request(make_request("req-2", [1, 2, 5, 6], max_tokens=1))

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"req-1": 4, "req-2": 4}


def test_scheduler_prioritizes_running_requests_before_waiting_fcfs() -> None:
    scheduler = Scheduler(
        make_config(
            SchedulerConfig(
                max_num_seqs=2,
                max_num_scheduled_tokens=4,
                max_num_partial_prefills=2,
            )
        ),
        make_kv_config(),
    )
    scheduler.add_request(make_request("running", [1, 2, 3, 4, 5, 6], max_tokens=1))
    first = scheduler.schedule()
    scheduler.update_from_outputs(first, {})
    scheduler.add_request(make_request("waiting", [7, 8], max_tokens=1))

    second = scheduler.schedule()

    assert tuple(second.num_scheduled_tokens) == ("running", "waiting")
    assert second.num_scheduled_tokens["running"] == 2
    assert second.num_scheduled_tokens["waiting"] == 2


def test_scheduler_decode_progression_and_finish() -> None:
    scheduler = Scheduler(
        make_config(SchedulerConfig(max_num_seqs=2, max_num_scheduled_tokens=4)),
        make_kv_config(),
    )
    request = make_request("req", [1, 2, 3, 4], max_tokens=2)
    scheduler.add_request(request)

    first = scheduler.schedule()
    update = scheduler.update_from_outputs(first, {"req": 9})
    assert update.step_outputs[0].sampled_token_id == 9
    assert not update.step_outputs[0].finished

    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {"req": 1}
    assert second.scheduled_cached_reqs.new_token_ids[0] == (9,)

    final_update = scheduler.update_from_outputs(second, {"req": 10})
    assert final_update.step_outputs[0].sampled_token_id == 10
    assert final_update.step_outputs[0].finished
    assert final_update.finished_requests[0].request_id == "req"


def test_scheduler_finish_requests_clears_waiting_and_running() -> None:
    scheduler = Scheduler(
        make_config(SchedulerConfig(max_num_seqs=1, max_num_scheduled_tokens=2)),
        make_kv_config(),
    )
    scheduler.add_request(make_request("running", [1, 2, 3, 4], max_tokens=1))
    scheduler.add_request(make_request("waiting", [5, 6], max_tokens=1))
    first = scheduler.schedule()
    scheduler.update_from_outputs(first, {})

    finished = scheduler.finish_requests(
        ("running", "waiting"),
        RequestStatus.FINISHED_ABORTED,
    )

    assert {request.request_id for request in finished} == {"running", "waiting"}
    assert scheduler.get_request_counts() == (0, 0)
    assert scheduler.has_finished_requests()
