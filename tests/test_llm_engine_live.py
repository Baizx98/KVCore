from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kvcore.engine.engine_core import EngineConfig
from kvcore.entry.llm_engine import LLMEngine
from kvcore.model.model_loader import ModelLoadConfig
from kvcore.utils.sampling_params import SamplingParams

LIVE_MODEL_PATH = Path("/Tan/model/Llama-3.1-8B-Instruct")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not LIVE_MODEL_PATH.exists(),
    reason="Local Llama-3.1-8B-Instruct CUDA test environment not available",
)
def test_llm_engine_live_llama31_generation() -> None:
    engine = LLMEngine(
        load_config=ModelLoadConfig(
            model=str(LIVE_MODEL_PATH),
            device="cuda",
            local_files_only=True,
        ),
        engine_config=EngineConfig(
            block_size=16,
            num_gpu_blocks=256,
            max_num_seqs=2,
            max_num_scheduled_tokens=128,
            max_model_len=512,
        ),
    )

    outputs = engine.generate(
        [
            {
                "request_id": "live-req",
                "messages": [
                    {"role": "system", "content": "You are a concise assistant."},
                    {"role": "user", "content": "Reply with one short word."},
                ],
                "sampling_params": SamplingParams(max_tokens=1, temperature=0.0),
            }
        ]
    )

    print(outputs)

    assert len(outputs) == 1
    assert outputs[0].request_id == "live-req"
    assert len(outputs[0].output_token_ids) == 1
    assert outputs[0].finish_reason == "length"

if __name__ == "__main__":
    test_llm_engine_live_llama31_generation()
