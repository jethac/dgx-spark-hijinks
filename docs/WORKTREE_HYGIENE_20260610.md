# Worktree Hygiene, 2026-06-10

This repo intentionally keeps claim evidence in compact summaries and manifests, not every raw server log.

Committed evidence classes:

- row summaries under `results/*.md`;
- compact manifests and benchmark JSON when explicitly force-added;
- direction docs, issue drafts, and blog/Colab materials;
- scripts that are needed to reproduce a committed row.

Generated scratch classes now ignored by default:

- container id files;
- editable-install and FlashInfer-install logs;
- raw server logs and pre/post probe logs;
- PPL stderr logs;
- metrics text captures;
- temporary source-stack logs.
- deterministic corpus slices generated from repository text.

Raw logs can still be committed with `git add -f` when a specific row needs them as public evidence. The current ignored raw files are treated as scratch because their findings are represented by committed summaries and manifests.

Known non-clean surface at this stop point: `third_party/flashinfer` is a live submodule worktree on the FlashInfer lane. Do not reset it from the main checkout; commit or clean it only from the owning FlashInfer branch.
