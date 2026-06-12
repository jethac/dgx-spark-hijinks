# SGLang Spark Base Probe

Date: 2026-06-12 JST

Scope: base-image OS/glibc compatibility for self-contained SGLang Spark images. No model load, no GPU serving, no compile.

## Findings

| image | platform | OS | glibc | verdict |
|---|---:|---|---|---|
| `nvcr.io/nvidia/sglang:26.05-py3` | `linux/arm64` | Ubuntu 24.04.4 | 2.39 | RED for Spark artifact under Ubuntu 22/glibc 2.35 invariant |
| `nvcr.io/nvidia/pytorch:25.04-py3` | `linux/arm64` | Ubuntu 24.04.2 | not separately probed | RED for Spark artifact |
| `nvcr.io/nvidia/pytorch:25.03-py3` | `linux/arm64` | Ubuntu 24.04.1 | not separately probed | RED for Spark artifact |
| `nvidia/cuda:13.0.2-devel-ubuntu22.04` | `linux/arm64` | Ubuntu 22.04.5 | expected 2.35 family | GREEN as OS base only; not yet a SGLang/PyTorch runtime |

## Consequence

The current SGLang source-stack workflow must not publish a Spark serving image from `nvcr.io/nvidia/sglang:26.05-py3`. It now has an in-container `/etc/os-release` and compiled-extension GLIBC ceiling gate, so that base fails early by design.

Next image work needs either:

1. a proven SGLang/PyTorch arm64 base that is already Ubuntu 22.04 / glibc <= 2.35, or
2. a new self-contained stack derived from `nvidia/cuda:13.0.2-devel-ubuntu22.04` with Python/PyTorch/SGLang installed and validated.

The second path is larger than a base swap because `scripts/install_sglang_source_stack.sh` assumes an existing Python 3.12 + torch runtime.

## Raw probe excerpts

`nvcr.io/nvidia/sglang:26.05-py3`:

```text
PRETTY_NAME="Ubuntu 24.04.4 LTS"
VERSION_ID="24.04"
ldd (Ubuntu GLIBC 2.39-0ubuntu8.7) 2.39
```

`nvcr.io/nvidia/pytorch:25.04-py3`:

```text
PRETTY_NAME="Ubuntu 24.04.2 LTS"
VERSION_ID="24.04"
```

`nvcr.io/nvidia/pytorch:25.03-py3`:

```text
PRETTY_NAME="Ubuntu 24.04.1 LTS"
VERSION_ID="24.04"
```

`nvidia/cuda:13.0.2-devel-ubuntu22.04`:

```text
PRETTY_NAME="Ubuntu 22.04.5 LTS"
VERSION_ID="22.04"
```
