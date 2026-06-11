# SGLang Gemma 4 E4B Chat Compare

Run: `sglang_gemma4_e4b_chat_compare_20260611T175952JST`

## Raw `/generate`

```json
{
  "text": "\n\n---\n\n---\n\n---\n\n---\n\n---\n\n---\n\n---\n\n---",
  "output_ids": [
    108,
    7243,
    108,
    7243,
    108,
    7243,
    108,
    7243,
    108,
    7243,
    108,
    7243,
    108,
    7243,
    108,
    7243
  ],
  "meta_info": {
    "id": "393fd63750b740e2a750c407c8b90205",
    "finish_reason": {
      "type": "length",
      "length": 16
    },
    "prompt_tokens": 11,
    "weight_version": "default",
    "num_retractions": 0,
    "reasoning_tokens": 0,
    "completion_tokens": 16,
    "cached_tokens": 0,
    "cached_tokens_details": null,
    "dp_rank": null,
    "e2e_latency": 72.81032070200308,
    "response_sent_to_client_ts": 1781168601.0862772
  }
}
```

## OpenAI `/v1/chat/completions`

```json
{
  "id": "ba818d0e9d4d487d9a4cd81d5a44c949",
  "object": "chat.completion",
  "created": 1781168601,
  "model": "google/gemma-4-E4B-it",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of Japan is Tokyo.",
        "reasoning_content": null,
        "tool_calls": null
      },
      "logprobs": null,
      "finish_reason": "stop",
      "matched_stop": 106
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "total_tokens": 28,
    "completion_tokens": 8,
    "prompt_tokens_details": null,
    "reasoning_tokens": 0
  },
  "metadata": {
    "weight_version": "default"
  }
}
```

Raw contains Tokyo: `False`
Chat contains Tokyo: `True`
