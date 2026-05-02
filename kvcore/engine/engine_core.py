from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from kvcore.config import KVCoreConfig
from kvcore.kv.kv_manager import KVManagerConfig
from kvcore.kv.single_type_kv_manager import KVLayerSpec
from kvcore.model.model_runner import KVCacheProfileResult, ModelRunner
from kvcore.sched.scheduler import Scheduler
from kvcore.sched.utils import RequestStepOutput
from kvcore.utils.log import get_logger
from kvcore.utils.request import FinishReason, Request
from kvcore.utils.sampling_params import SamplingParams
from kvcore.utils.tokenizer import TokenizerManager

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EngineCoreOutputs:
    request_outputs: tuple[RequestStepOutput, ...]

    @classmethod
    def empty(cls) -> EngineCoreOutputs:
        return cls(request_outputs=())


@dataclass(frozen=True, slots=True)
class FinishedRequestOutput:
    request_id: str
    output_token_ids: tuple[int, ...]
    finish_reason: FinishReason | None


class EngineCore:
    def __init__(
        self,
        config: KVCoreConfig,
    ) -> None:
        self.config = config
        logger.info(
            "Initializing EngineCore model=%s device=%s attn_backend=%s",
            self.config.model_config.model,
            self.config.device_config.device,
            self.config.model_config.attn_backend,
        )
        self.model_runner = ModelRunner(config)
        if self.model_runner.model is None:
            self.model_runner.load_model()
        self.tokenizer_manager = TokenizerManager.from_model_source(
            model=self.config.model_config.tokenizer or self.config.model_config.model,
            revision=self.config.load_config.revision,
            trust_remote_code=self.config.model_config.trust_remote_code,
            local_files_only=self.config.load_config.local_files_only,
        )
        self.stop_token_ids = self._resolve_stop_token_ids()

        kv_manager_config = self._build_kv_manager_config()
        self.scheduler = Scheduler(
            self.config,
            kv_manager_config,
        )
        self.model_runner.initialize_kv_cache(kv_manager_config)
        self.finished_outputs: dict[str, FinishedRequestOutput] = {}
        logger.info(
            "EngineCore ready num_gpu_blocks=%d block_size=%d max_model_len=%d "
            "num_layers=%d stop_tokens=%d",
            kv_manager_config.num_gpu_blocks,
            self.config.cache_config.block_size,
            kv_manager_config.max_model_len,
            len(kv_manager_config.layer_specs),
            len(self.stop_token_ids),
        )

    def add_request(
        self,
        request_id: str,
        messages: Sequence[Mapping[str, Any]],
        sampling_params: SamplingParams,
    ) -> None:
        prompt_token_ids = self.tokenizer_manager.encode_messages(messages)
        request = Request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
        self.scheduler.add_request(request)
        logger.info(
            "Added request request_id=%s prompt_tokens=%d max_tokens=%d",
            request_id,
            len(prompt_token_ids),
            sampling_params.max_tokens,
        )

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests()

    def step(self) -> EngineCoreOutputs:
        scheduler_output = self.scheduler.schedule()
        if scheduler_output.is_empty:
            logger.debug("Engine step skipped: scheduler output is empty")
            return EngineCoreOutputs.empty()
        logger.debug(
            "Engine step scheduled reqs=%d tokens=%d",
            len(scheduler_output.num_scheduled_tokens),
            scheduler_output.total_num_scheduled_tokens,
        )

        model_step_output = self.model_runner.execute_model(
            scheduler_output=scheduler_output,
        )
        sampled_token_map = {
            req_id: token_ids[0]
            for req_id, token_ids in zip(
                model_step_output.req_ids,
                model_step_output.sampled_token_ids,
                strict=True,
            )
            if token_ids
        }
        update_result = self.scheduler.update_from_outputs(
            scheduler_output,
            sampled_token_map,
            stop_token_ids=self.stop_token_ids,
        )
        for finished_request in update_result.finished_requests:
            self.finished_outputs[finished_request.request_id] = FinishedRequestOutput(
                request_id=finished_request.request_id,
                output_token_ids=finished_request.output_token_ids,
                finish_reason=finished_request.finish_reason,
            )
        self.model_runner.remove_requests(
            [finished_request.request_id for finished_request in update_result.finished_requests]
        )
        logger.info(
            "Engine step completed outputs=%d sampled=%d finished=%d",
            len(update_result.step_outputs),
            len(sampled_token_map),
            len(update_result.finished_requests),
        )
        return EngineCoreOutputs(request_outputs=update_result.step_outputs)

    def take_finished_output(self, request_id: str) -> FinishedRequestOutput | None:
        return self.finished_outputs.pop(request_id, None)

    def _build_kv_manager_config(self) -> KVManagerConfig:
        model = self.model_runner.model
        if model is None:
            raise RuntimeError("ModelRunner must own a loaded model before KV init")
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("Loaded model does not expose a Hugging Face config")

        num_hidden_layers = config.num_hidden_layers
        num_attention_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_attention_heads)
        hidden_size = config.hidden_size
        head_size = getattr(config, "head_dim", hidden_size // num_attention_heads)
        max_model_len = (
            self.config.model_config.max_model_len
            or getattr(config, "max_position_embeddings", None)
            or getattr(config, "max_model_len", None)
        )
        if max_model_len is None:
            raise ValueError("Unable to resolve max_model_len from engine/model config")

        param_dtype = next(model.parameters()).dtype
        layer_specs = tuple(
            KVLayerSpec(
                layer_idx=layer_idx,
                block_size=self.config.cache_config.block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=param_dtype,
            )
            for layer_idx in range(num_hidden_layers)
        )
        self.kv_cache_profile = self.model_runner.profile_run(
            layer_specs=layer_specs,
            block_size=self.config.cache_config.block_size,
            max_model_len=max_model_len,
            requested_num_gpu_blocks=self.config.cache_config.num_gpu_blocks,
            should_profile=self.config.cache_config.profile_kv_cache,
            gpu_memory_utilization=self.config.cache_config.gpu_memory_utilization,
        )
        logger.info(
            "KV cache profile num_gpu_blocks=%d bytes_per_block=%d "
            "max_tokens_per_sequence=%d",
            self.kv_cache_profile.num_gpu_blocks,
            self.kv_cache_profile.bytes_per_block,
            self.kv_cache_profile.max_tokens_per_sequence,
        )

        return KVManagerConfig(
            num_gpu_blocks=self.kv_cache_profile.num_gpu_blocks,
            max_model_len=max_model_len,
            layer_specs=layer_specs,
            enable_caching=self.config.cache_config.enable_caching,
        )

    def _resolve_stop_token_ids(self) -> set[int]:
        stop_token_ids = set(self.tokenizer_manager.get_stop_token_ids())
        config = getattr(self.model_runner.model, "config", None)
        eos_token_id = getattr(config, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            stop_token_ids.add(eos_token_id)
        elif isinstance(eos_token_id, (list, tuple)):
            stop_token_ids.update(int(token_id) for token_id in eos_token_id)
        return stop_token_ids

__all__ = [
    "EngineCore",
    "EngineCoreOutputs",
    "FinishedRequestOutput",
    "KVCacheProfileResult",
]
