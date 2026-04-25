from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from kvcore.config import KVCoreConfig, ModelConfig, RuntimeConfig
from kvcore.kv.kv_manager import KVManagerConfig
from kvcore.kv.single_type_kv_manager import KVLayerSpec
from kvcore.model.model_loader import ModelLoadConfig
from kvcore.model.model_runner import KVCacheProfileResult, ModelRunner
from kvcore.sched.scheduler import Scheduler
from kvcore.sched.utils import RequestStepOutput, SchedulerConfig
from kvcore.utils.request import FinishReason, Request
from kvcore.utils.sampling_params import SamplingParams
from kvcore.utils.tokenizer import TokenizerManager


@dataclass(frozen=True, slots=True)
class EngineConfig:
    block_size: int = 16
    num_gpu_blocks: int | None = 2048
    max_num_seqs: int = 8
    max_num_scheduled_tokens: int = 512
    max_model_len: int | None = None
    profile_kv_cache: bool = False
    gpu_memory_utilization: float = 0.9

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.num_gpu_blocks is not None and self.num_gpu_blocks <= 0:
            raise ValueError(f"num_gpu_blocks must be positive, got {self.num_gpu_blocks}")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                "gpu_memory_utilization must be in (0, 1], "
                f"got {self.gpu_memory_utilization}"
            )


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
        load_config: ModelLoadConfig | KVCoreConfig | None = None,
        engine_config: EngineConfig | None = None,
        *,
        config: KVCoreConfig | None = None,
        model_runner: ModelRunner | None = None,
        tokenizer_manager: TokenizerManager | None = None,
    ) -> None:
        self.config = self._normalize_config(
            config=config,
            load_config=load_config,
            engine_config=engine_config,
        )
        load_config = self.config.model.to_load_config()
        if load_config.attn_backend is None:
            load_config.attn_backend = self._default_attn_backend(load_config.device)
        self.load_config = load_config
        self.engine_config = self.config.runtime
        self.model_runner = model_runner or ModelRunner(load_config)
        if self.model_runner.model is None:
            self.model_runner.load_model()
        self.tokenizer_manager = tokenizer_manager or TokenizerManager.from_model_source(
            model=load_config.model,
            revision=load_config.revision,
            trust_remote_code=load_config.trust_remote_code,
            local_files_only=load_config.local_files_only,
        )
        self.stop_token_ids = self._resolve_stop_token_ids()

        kv_manager_config = self._build_kv_manager_config()
        self.scheduler = Scheduler(
            kv_manager_config,
            scheduler_config=self.config.scheduler,
        )
        self.model_runner.initialize_kv_cache(kv_manager_config)
        self.finished_outputs: dict[str, FinishedRequestOutput] = {}

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

    def abort_request(self, request_id: str) -> None:
        request = self.scheduler.abort_request(request_id)
        if request is None:
            return
        self.finished_outputs[request_id] = FinishedRequestOutput(
            request_id=request_id,
            output_token_ids=request.output_token_ids,
            finish_reason=request.get_finished_reason(),
        )

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests()

    def step(self) -> EngineCoreOutputs:
        scheduler_output = self.scheduler.schedule()
        if scheduler_output.is_empty:
            return EngineCoreOutputs.empty()

        model_step_output = self.model_runner.execute_model(
            scheduler_output=scheduler_output,
            kv_manager=self.scheduler.kv_manager,
        )
        sampled_token_map = dict(
            zip(
                model_step_output.sampled_request_ids,
                model_step_output.sampled_token_ids,
                strict=True,
            )
        )
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
            self.config.runtime.max_model_len
            or getattr(config, "max_position_embeddings", None)
            or getattr(config, "max_model_len", None)
        )
        if max_model_len is None:
            raise ValueError("Unable to resolve max_model_len from engine/model config")

        param_dtype = next(model.parameters()).dtype
        layer_specs = tuple(
            KVLayerSpec(
                layer_idx=layer_idx,
                block_size=self.config.runtime.block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=param_dtype,
            )
            for layer_idx in range(num_hidden_layers)
        )
        self.kv_cache_profile = self.model_runner.profile_run(
            layer_specs=layer_specs,
            block_size=self.config.runtime.block_size,
            max_model_len=max_model_len,
            requested_num_gpu_blocks=self.config.runtime.num_gpu_blocks,
            should_profile=self.config.runtime.profile_kv_cache,
            gpu_memory_utilization=self.config.runtime.gpu_memory_utilization,
        )

        return KVManagerConfig(
            num_gpu_blocks=self.kv_cache_profile.num_gpu_blocks,
            max_model_len=max_model_len,
            layer_specs=layer_specs,
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

    @classmethod
    def _normalize_config(
        cls,
        *,
        config: KVCoreConfig | None,
        load_config: ModelLoadConfig | KVCoreConfig | None,
        engine_config: EngineConfig | None,
    ) -> KVCoreConfig:
        if isinstance(load_config, KVCoreConfig):
            if config is not None:
                raise ValueError("KVCoreConfig was provided twice")
            config = load_config
            load_config = None
        if config is not None:
            if load_config is not None or engine_config is not None:
                raise ValueError(
                    "Pass either KVCoreConfig or legacy load_config/engine_config, not both."
                )
            return config
        if load_config is None:
            raise ValueError("load_config is required when KVCoreConfig is not provided")

        legacy_engine_config = engine_config or EngineConfig()
        return KVCoreConfig(
            model=ModelConfig.from_load_config(load_config),
            runtime=RuntimeConfig(
                block_size=legacy_engine_config.block_size,
                num_gpu_blocks=legacy_engine_config.num_gpu_blocks,
                max_model_len=legacy_engine_config.max_model_len,
                profile_kv_cache=legacy_engine_config.profile_kv_cache,
                gpu_memory_utilization=legacy_engine_config.gpu_memory_utilization,
            ),
            scheduler=SchedulerConfig(
                max_num_seqs=legacy_engine_config.max_num_seqs,
                max_num_scheduled_tokens=legacy_engine_config.max_num_scheduled_tokens,
            ),
        )

    @staticmethod
    def _default_attn_backend(device: str | None) -> str:
        if (device is None and torch.cuda.is_available()) or (
            device is not None and torch.device(device).type == "cuda"
        ):
            return "triton_paged"
        return "torch_paged"


__all__ = [
    "EngineConfig",
    "EngineCore",
    "EngineCoreOutputs",
    "FinishedRequestOutput",
    "KVCacheProfileResult",
]
