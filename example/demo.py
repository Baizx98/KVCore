from kvcore.api import EngineConfig, GenerationConfig, LLMEngine, Request

config = EngineConfig(
    model_name_or_path="/Tan/model/Llama-3.2-1B-Instruct",
    device="cuda:1",
    dtype="bfloat16",
    max_new_tokens=16,
    block_size=16,
    log_level="INFO",
)

engine = LLMEngine.from_pretrained(config)

result = engine.generate(
    Request(
        prompt="Hello, please briefly introduce yourself in Chinese.",
        request_id="manual-smoke",
    ),
    GenerationConfig(max_new_tokens=1024),
)

print("finish_reason =", result.finish_reason)
print("num_prompt_tokens =", result.num_prompt_tokens)
print("num_generated_tokens =", result.num_generated_tokens)
print("kv_block_count =", result.kv_block_count)
print("kv_total_tokens =", result.kv_total_tokens)
print("generated_text =", repr(result.text))
print("generated_token_ids =", result.generated_token_ids)
