# llama.cpp Source Loglikelihood Audit

Source: `B:\workshop\dgx-spark-hijinks\third_party\llama.cpp`
Commit: `19bba67c1f4db723c60a0d421aa0788bf4ddc699`

Stock server contract capable: `False`
Endpoint patch needed: `True`

## Findings

- No server-source evidence of prompt-token token_logprobs arrays or a supplied-token loglikelihood endpoint.
- Server logprob evidence is generated-token/top-N oriented via n_probs/probs_output.
- Server prompt evaluation does not appear to retain full prompt-token logits for supplied-token scoring.
- Non-server tools already compute supplied-token logprobs from logits and token ids.

## Server Logprob Evidence

- `tools/server/server-common.cpp:830` throw std::runtime_error("Only no echo is supported");
- `tools/server/server-common.cpp:1139` // Handle "logprobs" field
- `tools/server/server-common.cpp:1141` if (json_value(body, "logprobs", false)) {
- `tools/server/server-common.cpp:1143` throw std::invalid_argument("logprobs is not supported with tools + stream");
- `tools/server/server-common.cpp:1145` llama_params["n_probs"] = json_value(body, "top_logprobs", 20);
- `tools/server/server-common.cpp:1146` } else if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
- `tools/server/server-common.cpp:1147` throw std::invalid_argument("top_logprobs requires logprobs to be set to true");
- `tools/server/server-task.cpp:333` // Use OpenAI API logprobs only if n_probs wasn't provided
- `tools/server/server-task.cpp:334` if (data.contains("logprobs") && params.sampling.n_probs == defaults.sampling.n_probs){
- `tools/server/server-task.cpp:335` params.sampling.n_probs = json_value(data, "logprobs", defaults.sampling.n_probs);
- `tools/server/server-task.cpp:708` post_sampling_probs ? "top_probs" : "top_logprobs",
- `tools/server/server-task.cpp:771` if (!stream && !probs_output.empty()) {
- `tools/server/server-task.cpp:772` res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
- `tools/server/server-task.cpp:788` json logprobs = json(nullptr); // OAI default to null

## Prompt-Eval Logits Evidence

- `tools/server/server-context.cpp:115` std::vector<completion_token_output> generated_token_probs;
- `tools/server/server-context.cpp:221` generated_token_probs.clear();
- `tools/server/server-context.cpp:320` generated_token_probs.push_back(token);
- `tools/server/server-context.cpp:1686` void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
- `tools/server/server-context.cpp:1868` size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
- `tools/server/server-context.cpp:1870` slot.generated_token_probs.begin(),
- `tools/server/server-context.cpp:1871` slot.generated_token_probs.end() - safe_offset);
- `tools/server/server-context.cpp:1874` slot.generated_token_probs.begin(),
- `tools/server/server-context.cpp:1875` slot.generated_token_probs.end());
- `tools/server/server-context.cpp:3078` // extract the logits only for the last token
- `tools/server/server-context.cpp:3397` result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs
- `tools/server/server-context.cpp:3400` populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);

## Prompt-Token Server Evidence

- none

## Reusable Scoring Evidence

- `tools/perplexity/perplexity.cpp:34` struct results_log_softmax {
- `tools/perplexity/perplexity.cpp:35` double log_softmax;
- `tools/perplexity/perplexity.cpp:40` static std::vector<float> softmax(const std::vector<float>& logits) {
- `tools/perplexity/perplexity.cpp:60` static results_log_softmax log_softmax(int n_vocab, const float * logits, int tok) {
- `tools/perplexity/perplexity.cpp:79` static double log_softmax(int n_vocab, const float * logits, uint16_t * log_prob, int tok) {
- `tools/perplexity/perplexity.cpp:126` const results_log_softmax results = log_softmax(n_vocab, logits + size_t(i)*n_vocab, tokens[i+1]);
- `tools/perplexity/perplexity.cpp:127` const double v = -results.log_softmax;
- `tools/perplexity/perplexity.cpp:160` const double v = log_softmax(n_vocab, logits + size_t(i)*n_vocab, log_probs.data() + size_t(i)*nv, tokens[i+1]);
- `tools/perplexity/perplexity.cpp:191` static std::pair<double, float> log_softmax(int n_vocab, const float * logits, const uint16_t * base_log_prob, int tok, kl_divergence_result & kld) {
- `tools/perplexity/perplexity.cpp:282` std::pair<double, float> v = log_softmax(n_vocab, logits + size_t(i)*n_vocab, base_log_probs.data() + size_t(i)*nv, tokens[i+1], local_kld);
- `tools/perplexity/perplexity.cpp:395` const auto * batch_logits = llama_get_logits(ctx);
- `tools/perplexity/perplexity.cpp:425` const float prob = softmax(tok_logits)[tokens[start + j + 1]];
- `tools/perplexity/perplexity.cpp:595` const auto * batch_logits = llama_get_logits(ctx);
- `tools/perplexity/perplexity.cpp:615` const float * all_logits = num_batches > 1 ? logits.data() : llama_get_logits_ith(ctx, seq*n_ctx + first);

## Recommended Patch Shape

- Add a server endpoint or request mode that accepts context plus continuation.
- Tokenize context and continuation separately, preserving normal BOS/chat-template behavior in the response.
- Decode context+continuation with logits for every continuation prediction position.
- For each continuation token id, compute log_softmax(logits)[token_id] directly, independent of top-N rank.
- Return continuation_token_ids, continuation_token_logprobs, target_logprob_sum, all_tokens_greedy, and lm_eval_loglikelihood_tuple.
- Keep generated-token n_probs/top_logprobs output separate from this supplied-token scoring contract.
