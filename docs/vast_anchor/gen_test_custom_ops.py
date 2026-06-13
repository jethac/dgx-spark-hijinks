import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


TOKEN = os.environ["HF_TOKEN"]
MODEL = os.environ.get("MODEL", "google/gemma-4-12B-it")
CUSTOM_OPS = os.environ.get("CUSTOM_OPS", "all")

llm_kwargs = {
    "model": MODEL,
    "kv_cache_dtype": os.environ.get("KV_CACHE_DTYPE", "bfloat16"),
    "max_model_len": int(os.environ.get("MAX_MODEL_LEN", "8192")),
    "enforce_eager": True,
    "gpu_memory_utilization": float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9")),
    "trust_remote_code": True,
}

if CUSTOM_OPS != "all":
    llm_kwargs["compilation_config"] = {"custom_ops": [CUSTOM_OPS]}

if os.environ.get("NATIVE_RMS", "0") == "1":
    llm_kwargs["kernel_config"] = {
        "ir_op_priority": {
            "rms_norm": ["native"],
            "fused_add_rms_norm": ["native"],
        }
    }

print("MODEL", MODEL)
print("CUSTOM_OPS", CUSTOM_OPS)
print("NATIVE_RMS", os.environ.get("NATIVE_RMS", "0"))
print("LLM_KWARGS", {k: v for k, v in llm_kwargs.items() if k != "model"})

tok = AutoTokenizer.from_pretrained(MODEL, token=TOKEN)
llm = LLM(**llm_kwargs)

g = llm.generate(
    ["The capital of France is"],
    SamplingParams(max_tokens=8, temperature=0.0),
)
print("GEN:", repr(g[0].outputs[0].text))

ids = tok("The quick brown fox jumps over the lazy dog")["input_ids"]
o = llm.generate(
    [{"prompt_token_ids": ids}],
    SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0),
)
pls = o[0].prompt_logprobs
for i in range(1, len(pls)):
    d = pls[i]
    top = max(d.items(), key=lambda kv: kv[1].logprob)
    actual = d.get(ids[i])
    print(
        f"pos{i} actual={tok.decode([ids[i]])!r} nll={-actual.logprob:.2f} "
        f"| top1={tok.decode([top[0]])!r} lp={top[1].logprob:.2f}"
    )
