# Forks, Submodules, And Worktrees

Policy: if this campaign needs changes to an upstream open-source library, the changes must live in a `jethac` fork and be added to this repo as a git submodule.

No loose long-lived patch directories.

## Rules

- Fork upstream under `jethac`.
- Add the fork under `third_party/<repo-name>`.
- Create one branch per GitHub Issue.
- Use a separate git worktree for each active branch.
- Link every fork branch back to the hijinks Issue that requires it.
- Record upstream base commit, local branch, test command, and upstreaming plan.

## Naming

Submodules:

| upstream | fork | submodule path |
|---|---|---|
| `vllm-project/vllm` | `jethac/vllm` | `third_party/vllm` |
| `sgl-project/sglang` | `jethac/sglang` | `third_party/sglang` |
| `flashinfer-ai/flashinfer` | `jethac/flashinfer` | `third_party/flashinfer` |
| `ggml-org/llama.cpp` | `jethac/llama.cpp` | `third_party/llama.cpp` |
| `google-ai-edge/LiteRT-LM` | `jethac/LiteRT-LM` | `third_party/LiteRT-LM` |

Branches:

```text
spark/hijinks-<issue-number>-short-topic
```

Examples:

```text
spark/hijinks-001-sm121-build
spark/hijinks-003-gemma4-unified
spark/hijinks-004-backend-observability
spark/hijinks-014-sglang-runtime-sm121
spark/hijinks-018-sglang-nvfp4-kv-sm121
```

Worktrees:

```text
B:/workshop/worktrees/<repo-name>/<branch-slug>
```

Example:

```bash
git -C third_party/vllm worktree add B:/workshop/worktrees/vllm/spark-hijinks-003-gemma4-unified \
  -b spark/hijinks-003-gemma4-unified upstream/main
```

## Fork Command Pattern

```bash
gh repo fork vllm-project/vllm --org jethac --clone=false
git submodule add git@github.com:jethac/vllm.git third_party/vllm
git -C third_party/vllm remote add upstream https://github.com/vllm-project/vllm.git
git -C third_party/vllm fetch upstream
```

Use the same pattern for SGLang, FlashInfer, llama.cpp, and LiteRT-LM when a real code change is needed.

## Reference Repositories

Reference repos are allowed, but their names must not imply Spark validation.

For example, `hikarioyama/sglang-nvfp4-kv-sm120` should be treated as SM120 source context, not as a Spark fork. If cloned, name it something like:

```text
references/sglang-nvfp4-kv-sm120-reference
```

Active Spark work should use architecture-accurate names:

```text
sglang-runtime-sm121
sglang-nvfp4-kv-sm121
flashinfer-nvfp4-kv-sm121
vllm-gemma4-sm121
```

## Current State

No upstream forks are added as submodules yet. We have not made code changes to upstream libraries in this repo.

The first submodule should be added when an issue has a concrete patch target and reproduction command.

The vLLM subagent checked that `jethac/vllm` did not exist at the time of this note. Create forks only when the corresponding issue is ready to carry code.
