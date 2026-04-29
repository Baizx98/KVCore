from __future__ import annotations

import argparse
import hashlib
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from kvcore.config import KVCoreConfig, ModelConfig
from kvcore.engine.engine_core import (
    EngineConfig,
    EngineCore,
    EngineCoreOutputs,
    FinishedRequestOutput,
)
from kvcore.entry.llm_engine import GenerationRequest
from kvcore.model.model_loader import ModelLoadConfig
from kvcore.model.model_runner import ModelRunner
from kvcore.model.models.llama3 import Llama3ForCausalLM
from kvcore.utils.log import configure_logging
from kvcore.utils.sampling_params import SamplingParams


@dataclass(frozen=True, slots=True)
class DebugRequest:
    request: GenerationRequest
    prompt_token_ids: tuple[int, ...]
    prompt_text: str


EMBEDDED_CHINESE_REQUESTS = (
    "请分析 KVCore 当前的 Scheduler 如何在连续批处理中处理长 prompt 请求，重点说明 chunked prefill "
    "和 decode 请求混合时 token budget 如何分配，并给出可能影响吞吐量和尾延迟的因素。",
    "我正在设计 KV cache 块稀疏压缩实验，请帮我提出一个最小可复现的实验方案，要求包含随机块驱逐 "
    "作为 baseline、固定 seed、正确性检查、显存占用统计和生成质量观察。",
    "请比较 vLLM 的 GPUModelRunner 和 KVCore 当前 ModelRunner 的职责边界，重点讨论 InputBatch、"
    "attention metadata、block table、slot mapping 和 sampler 之间的数据流。",
    "假设一个在线服务场景中同时到达多个中文问答请求，请说明异步前端如何将请求提交给 EngineCore，"
    "以及 run_until_idle 循环如何在没有真正异步 GPU executor 的情况下模拟在线批处理。",
    "请为 KV cache CPU offload 设计一个第一阶段原型，要求区分权重 offload 和 KV block offload，"
    "并描述 CPU storage、GPU block location、prefetch plan 和 ModelRunner copy hook 的接口。",
    "请解释为什么 chunked prefill 的中间 chunk 不应该采样，最后一个 prefill chunk "
    "才采样首个 decode token，并用一个长 prompt 的执行过程举例说明 num_computed_tokens 如何推进。",
    "我需要写一段系统论文中的设计动机，请从 KV cache 内存压力、长上下文请求、"
    "连续批处理和研究可扩展性 "
    "四个角度说明为什么 KVCore 需要清晰划分 Scheduler、KVManager 和 ModelRunner。",
    "请检查一个推理系统的 step 级别日志，说明应该关注哪些指标来判断性能瓶颈，例如 prepare time、"
    "metadata time、forward time、sample time、scheduled tokens、prefill 数量和 decode 数量。",
    "请构造一个多请求混合 batch 的例子，其中包含两个正在 decode 的请求、一个 prefill continuation "
    "和一个新的 waiting request，并解释 scheduler output 中 new request 和 cached request 的区别。",
    "请给出下一阶段优化 KVCore 的建议，但不要引入完整 vLLM serving stack，"
    "只考虑单机研究内核中最有价值的 "
    "改进，例如持久化 InputBatch buffer、减少 metadata 构造开销和增加 offload 观测指标。",
)


class DebugTokenizerManager:
    def __init__(self, *, vocab_size: int, seed: int) -> None:
        self.vocab_size = vocab_size
        self.seed = seed
        self.id_to_text: dict[int, str] = {}

    def encode_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        del add_generation_prompt
        text = " ".join(str(message["content"]) for message in messages)
        tokens: list[int] = []
        for index, word in enumerate(text.split()):
            digest = hashlib.blake2b(
                f"{self.seed}:{index}:{word}".encode(),
                digest_size=4,
            ).digest()
            token_id = (int.from_bytes(digest, "little") % (self.vocab_size - 8)) + 4
            tokens.append(token_id)
            self.id_to_text.setdefault(token_id, word)
        return tokens

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        pieces = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id < 4:
                pieces.append(f"<special_{token_id}>")
            else:
                pieces.append(self.id_to_text.get(token_id, f"<id:{token_id}>"))
        return " ".join(pieces)

    def get_stop_token_ids(self) -> set[int]:
        return set()


def make_tiny_model_runner(
    *,
    vocab_size: int,
    max_model_len: int,
    seed: int,
    attn_backend: str,
) -> ModelRunner:
    torch.manual_seed(seed)
    runner = ModelRunner(
        ModelLoadConfig(model="unused", device="cpu", attn_backend=attn_backend)
    )
    hf_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max_model_len,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
    )
    runner.model = Llama3ForCausalLM(
        KVCoreConfig(
            model=ModelConfig(
                model="unused",
                device="cpu",
                attn_backend=attn_backend,
                hf_config=hf_config,
            )
        )
    )
    return runner


def make_random_requests(
    *,
    num_requests: int,
    min_prompt_tokens: int,
    max_prompt_tokens: int,
    min_new_tokens: int,
    max_new_tokens: int,
    vocab_size: int,
    seed: int,
) -> list[DebugRequest]:
    rng = random.Random(seed)
    tokenizer = DebugTokenizerManager(vocab_size=vocab_size, seed=seed)
    requests: list[DebugRequest] = []
    for request_index in range(num_requests):
        content = EMBEDDED_CHINESE_REQUESTS[
            request_index % len(EMBEDDED_CHINESE_REQUESTS)
        ]
        words = content.split()
        target_prompt_len = rng.randint(min_prompt_tokens, max_prompt_tokens)
        if len(words) < target_prompt_len:
            repeats = (target_prompt_len + len(words) - 1) // len(words)
            words = (words * repeats)[:target_prompt_len]
            content = " ".join(words)
        elif len(words) > max_prompt_tokens:
            words = words[:max_prompt_tokens]
            content = " ".join(words)
        max_tokens = rng.randint(min_new_tokens, max_new_tokens)
        messages = ({"role": "user", "content": content},)
        prompt_token_ids = tuple(tokenizer.encode_messages(messages))
        requests.append(
            DebugRequest(
                request=GenerationRequest(
                    request_id=f"offline-{request_index:02d}",
                    messages=messages,
                    sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
                ),
                prompt_token_ids=prompt_token_ids,
                prompt_text=content,
            )
        )
    return requests


def print_request_summary(requests: list[DebugRequest]) -> None:
    print("=== Requests ===")
    for item in requests:
        print(
            f"{item.request.request_id}: "
            f"prompt_tokens={len(item.prompt_token_ids)} "
            f"max_new_tokens={item.request.sampling_params.max_tokens} "
            f"prompt_head={item.prompt_token_ids[:8]} "
            f"prompt_text={item.prompt_text[:48]}"
        )


def print_scheduler_state(core: EngineCore) -> None:
    scheduler = core.scheduler
    print(f"waiting={tuple(scheduler.waiting)} running={tuple(scheduler.running)}")
    for request_id in tuple(scheduler.running) + tuple(scheduler.waiting):
        request = scheduler.requests[request_id]
        print(
            f"  state {request_id}: computed={request.num_computed_tokens} "
            f"tokens={request.num_tokens} prompt={request.num_prompt_tokens} "
            f"output={request.num_output_tokens} status={request.status.value}"
        )


def debug_step(core: EngineCore, step_index: int) -> EngineCoreOutputs:
    print(f"\n=== Step {step_index} ===")
    print_scheduler_state(core)

    scheduler_output = core.scheduler.schedule()
    if scheduler_output.is_empty:
        print("scheduler_output: empty")
        return EngineCoreOutputs.empty()

    print(
        "scheduler_output: "
        f"total_tokens={scheduler_output.total_num_scheduled_tokens} "
        f"prefill_reqs={scheduler_output.num_prefill_reqs} "
        f"decode_reqs={scheduler_output.num_decode_reqs} "
        f"new_reqs={[req.req_id for req in scheduler_output.scheduled_new_reqs]} "
        f"cached_reqs={list(scheduler_output.scheduled_cached_reqs.req_ids)} "
        f"zero_blocks={scheduler_output.new_block_ids_to_zero}"
    )
    for scheduled_request in scheduler_output.scheduled_requests:
        layer0_blocks = scheduled_request.block_ids[0] if scheduled_request.block_ids else ()
        print(
            f"  scheduled {scheduled_request.request_id}: "
            f"prefill={scheduled_request.is_prefill} "
            f"context={scheduled_request.context_len} "
            f"query={scheduled_request.query_len} "
            f"sample={scheduled_request.should_sample} "
            f"sample_index={scheduled_request.sample_index} "
            f"layer0_blocks={layer0_blocks}"
        )

    model_step_output = core.model_runner.execute_model(scheduler_output)
    sampled_token_map = dict(
        zip(
            model_step_output.sampled_request_ids,
            model_step_output.sampled_token_ids,
            strict=True,
        )
    )
    sampled_text = {
        request_id: core.tokenizer_manager.decode((token_id,))
        for request_id, token_id in sampled_token_map.items()
    }
    print(f"model_output: sampled={sampled_token_map} sampled_text={sampled_text}")
    stats = core.model_runner.last_step_stats
    if stats is not None:
        print(
            "model_stats: "
            f"reqs={stats.num_reqs} tokens={stats.num_scheduled_tokens} "
            f"prefill={stats.num_prefill_reqs} decode={stats.num_decode_reqs} "
            f"zeroed={stats.num_zeroed_blocks} "
            f"prepare={stats.prepare_time_sec:.6f}s "
            f"metadata={stats.metadata_time_sec:.6f}s "
            f"forward={stats.forward_time_sec:.6f}s "
            f"sample={stats.sample_time_sec:.6f}s"
        )

    update_result = core.scheduler.update_from_outputs(
        scheduler_output,
        sampled_token_map,
        stop_token_ids=core.stop_token_ids,
    )
    for step_output in update_result.step_outputs:
        print(
            f"  update {step_output.request_id}: "
            f"sampled={step_output.sampled_token_id} "
            f"finished={step_output.finished} "
            f"finish_reason={step_output.finish_reason}"
        )
    for finished_request in update_result.finished_requests:
        core.finished_outputs[finished_request.request_id] = FinishedRequestOutput(
            request_id=finished_request.request_id,
            output_token_ids=finished_request.output_token_ids,
            finish_reason=finished_request.finish_reason,
        )
    core.model_runner.remove_requests(
        [request.request_id for request in update_result.finished_requests]
    )
    return EngineCoreOutputs(request_outputs=update_result.step_outputs)


def build_engine(args: argparse.Namespace) -> EngineCore:
    config = EngineConfig(
        block_size=args.block_size,
        num_gpu_blocks=args.num_gpu_blocks,
        max_num_seqs=args.max_num_seqs,
        max_num_scheduled_tokens=args.max_num_scheduled_tokens,
        max_num_partial_prefills=args.max_num_partial_prefills,
        max_long_partial_prefills=args.max_long_partial_prefills,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        max_model_len=args.max_model_len,
    )
    return EngineCore(
        load_config=ModelLoadConfig(
            model="unused",
            device="cpu",
            attn_backend=args.attn_backend,
        ),
        engine_config=config,
        model_runner=make_tiny_model_runner(
            vocab_size=args.vocab_size,
            max_model_len=args.max_model_len,
            seed=args.seed,
            attn_backend=args.attn_backend,
        ),
        tokenizer_manager=DebugTokenizerManager(
            vocab_size=args.vocab_size,
            seed=args.seed,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a verbose offline batch generation debug test."
    )
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--min-prompt-tokens", type=int, default=24)
    parser.add_argument("--max-prompt-tokens", type=int, default=40)
    parser.add_argument("--min-new-tokens", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=18)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=96)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--num-gpu-blocks", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-scheduled-tokens", type=int, default=16)
    parser.add_argument("--max-num-partial-prefills", type=int, default=2)
    parser.add_argument("--max-long-partial-prefills", type=int, default=1)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=32)
    parser.add_argument("--attn-backend", default="torch_paged")
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    requests = make_random_requests(
        num_requests=args.num_requests,
        min_prompt_tokens=args.min_prompt_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    print_request_summary(requests)
    core = build_engine(args)
    for item in requests:
        core.add_request(
            request_id=item.request.request_id,
            messages=item.request.messages,
            sampling_params=item.request.sampling_params,
        )

    start_time = time.perf_counter()
    step_index = 0
    while core.has_unfinished_requests():
        if step_index >= args.max_steps:
            raise RuntimeError(f"Exceeded max_steps={args.max_steps}")
        step_output = debug_step(core, step_index)
        if not step_output.request_outputs and core.has_unfinished_requests():
            no_progress_state = core.scheduler.last_no_progress_state
            if no_progress_state is not None:
                raise RuntimeError(no_progress_state.format_message())
            raise RuntimeError("No progress while requests remain unfinished")
        step_index += 1

    elapsed = time.perf_counter() - start_time
    print("\n=== Final Outputs ===")
    for item in requests:
        finished_output = core.take_finished_output(item.request.request_id)
        if finished_output is None:
            print(f"{item.request.request_id}: missing finished output")
            continue
        decoded = core.tokenizer_manager.decode(finished_output.output_token_ids)
        print(
            f"{item.request.request_id}: "
            f"tokens={finished_output.output_token_ids} "
            f"text={decoded!r} "
            f"finish_reason={finished_output.finish_reason}"
        )
    print(f"\ncompleted_steps={step_index} elapsed_sec={elapsed:.3f}")


if __name__ == "__main__":
    main()
