from __future__ import annotations

import argparse
import contextlib
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from kvcore.config import KVCoreConfig, ModelConfig, RuntimeConfig
from kvcore.entry.llm_engine import GenerationRequest, LLMEngine
from kvcore.sched.utils import SchedulerConfig
from kvcore.utils.log import configure_logging
from kvcore.utils.sampling_params import SamplingParams

DEFAULT_MODEL = "/Tan/model/Llama-3.1-8B-Instruct"
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_PREFIX = "llm_engine_offline_batch_llama31_8b"

# Fill this list with the offline batch requests you want to test.
# Each request can override sampling params with a plain dict.
OFFLINE_REQUESTS: list[dict[str, Any]] = [
    {
        "request_id": "offline-000",
        "messages": [
            {
                "role": "user",
                "content": "中国的首都是什么？请用一句话回答。",
            }
        ],
        "sampling_params": {
            "max_tokens": 32,
            "temperature": 0.0,
        },
    },
    {
        "request_id": "offline-001",
        "messages": [
            {
                "role": "user",
                "content": "请简要说明 KV cache 在大模型推理中的作用。",
            }
        ],
        "sampling_params": {
            "max_tokens": 64,
            "temperature": 0.0,
        },
    },
    {
        "request_id": "offline-002",
        "messages": [
            {
                "role": "user",
                "content": "请简要说明 KV cache 在大模型推理中的作用。",
            }
        ],
        "sampling_params": {
            "max_tokens": 64,
            "temperature": 0.0,
        },
    },
    {
        "request_id": "offline-003",
        "messages": [
            {
                "role": "user",
                "content": "请简要说明因果掩码在大模型推理中的作用。",
            }
        ],
        "sampling_params": {
            "max_tokens": 64,
            "temperature": 0.0,
        },
    },
    {
        "request_id": "offline-004",
        "messages": [
            {
                "role": "user",
                "content": "请简要说明位置编码在大模型推理中的作用。",
            }
        ],
        "sampling_params": {
            "max_tokens": 64,
            "temperature": 0.0,
        },
    },
    {
        "request_id": "offline-005",
        "messages": [
            {
                "role": "user",
                "content": "请简要说明prefill在大模型推理中的作用。",
            }
        ],
        "sampling_params": {
            "max_tokens": 64,
            "temperature": 0.0,
        },
    },
    {
        "request_id": "offline-006",
        "messages": [
            {
                "role": "user",
                "content": "请简要说明 KV cache 稀疏原理。",
            }
        ],
        "sampling_params": {
            "max_tokens": 64,
            "temperature": 0.0,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline batch generation through LLMEngine."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-backend", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-gpu-blocks", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-scheduled-tokens", type=int, default=128)
    parser.add_argument("--max-num-partial-prefills", type=int, default=2)
    parser.add_argument("--max-long-partial-prefills", type=int, default=2)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=0)
    parser.add_argument("--reserve-full-prompt-blocks", action="store_true")
    parser.add_argument("--log-level", default="DEBUG")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path for teeing terminal output. Defaults to logs/<script>_<timestamp>.log.",
    )
    return parser.parse_args()


def main() -> None:
    program_start = time.perf_counter()
    args = parse_args()
    log_path = resolve_log_path(args.log_file)

    with tee_terminal_output(log_path):
        configure_logging(args.log_level, force=True)
        wall_start = datetime.now().astimezone()
        print("=== Run Metadata ===")
        print(f"start_time={wall_start.isoformat(timespec='seconds')}")
        print(f"log_file={log_path}")
        print(f"model={Path(args.model).expanduser()}")
        print(f"device={args.device}")
        print(f"local_files_only={args.local_files_only}")
        print(f"max_num_seqs={args.max_num_seqs}")
        print(f"max_num_scheduled_tokens={args.max_num_scheduled_tokens}")

        requests = build_generation_requests(OFFLINE_REQUESTS)
        if not requests:
            raise ValueError("OFFLINE_REQUESTS is empty; add at least one request.")

        engine_start = time.perf_counter()
        engine = LLMEngine(config=build_config(args))
        engine_elapsed = time.perf_counter() - engine_start
        print(f"engine_init_sec={engine_elapsed:.3f}")
        print_request_summary(requests)

        generate_start = time.perf_counter()
        outputs = engine.generate(requests)
        generate_elapsed = time.perf_counter() - generate_start

        print("\n=== Detokenized Outputs ===")
        for output in outputs:
            print(f"\n[{output.request_id}] finish_reason={output.finish_reason}")
            print(f"token_ids={output.output_token_ids}")
            print(output.output_text)

        total_elapsed = time.perf_counter() - program_start
        wall_end = datetime.now().astimezone()
        print("\n=== Timing Summary ===")
        print(f"end_time={wall_end.isoformat(timespec='seconds')}")
        print(f"num_requests={len(requests)}")
        print(f"engine_init_sec={engine_elapsed:.3f}")
        print(f"generate_sec={generate_elapsed:.3f}")
        print(f"total_elapsed_sec={total_elapsed:.3f}")


def build_config(args: argparse.Namespace) -> KVCoreConfig:
    return KVCoreConfig(
        model=ModelConfig(
            model=str(Path(args.model).expanduser()),
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            attn_backend=args.attn_backend,
            device=args.device,
        ),
        runtime=RuntimeConfig(
            block_size=args.block_size,
            num_gpu_blocks=args.num_gpu_blocks,
            max_model_len=args.max_model_len,
        ),
        scheduler=SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            max_num_scheduled_tokens=args.max_num_scheduled_tokens,
            max_num_partial_prefills=args.max_num_partial_prefills,
            max_long_partial_prefills=args.max_long_partial_prefills,
            long_prefill_token_threshold=args.long_prefill_token_threshold,
            reserve_full_prompt_blocks=args.reserve_full_prompt_blocks,
        ),
    )


def build_generation_requests(
    request_specs: Sequence[Mapping[str, Any]],
) -> list[GenerationRequest]:
    return [
        GenerationRequest(
            request_id=str(request_spec["request_id"]),
            messages=tuple(request_spec["messages"]),
            sampling_params=build_sampling_params(request_spec.get("sampling_params", {})),
        )
        for request_spec in request_specs
    ]


def build_sampling_params(params: SamplingParams | Mapping[str, Any]) -> SamplingParams:
    if isinstance(params, SamplingParams):
        return params
    return SamplingParams(**dict(params))


def print_request_summary(requests: Sequence[GenerationRequest]) -> None:
    print("=== Offline Requests ===")
    for request in requests:
        prompt = " ".join(str(message["content"]) for message in request.messages)
        print(
            f"{request.request_id}: messages={len(request.messages)} "
            f"max_tokens={request.sampling_params.max_tokens} "
            f"temperature={request.sampling_params.temperature} "
            f"prompt={prompt[:80]}"
        )


def resolve_log_path(log_file: str | None) -> Path:
    if log_file is not None:
        return Path(log_file).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return DEFAULT_LOG_DIR / f"{DEFAULT_LOG_PREFIX}_{timestamp}.log"


class TeeTextIO:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            if "\n" in data:
                stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(stream.isatty() for stream in self.streams)


class tee_terminal_output(contextlib.AbstractContextManager[Path]):
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_file: TextIO | None = None
        self.stdout: TextIO | None = None
        self.stderr: TextIO | None = None

    def __enter__(self) -> Path:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = TeeTextIO(self.stdout, self.log_file)  # type: ignore[assignment]
        sys.stderr = TeeTextIO(self.stderr, self.log_file)  # type: ignore[assignment]
        return self.log_path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self.stdout is not None:
            sys.stdout = self.stdout
        if self.stderr is not None:
            sys.stderr = self.stderr
        if self.log_file is not None:
            self.log_file.close()
        return None


if __name__ == "__main__":
    main()
