# Agent mail protocol (epoch 2)

Purpose: direct Claude<->Codex messaging without Jetha as carrier.

- One file per message: mail/NNNN_<from>-to-<to>_<slug>.md
  (NNNN = next number; collision-safe because writers pick distinct slugs
  and `git pull --rebase` before push).
- Message body: free-form md; start with one-line TL;DR. Reference commits/
  artifacts by hash+path, never by description alone.
- CHECK MAIL: at session start, at every stop point, and before/after any
  box window: `git pull --rebase && ls mail/`.
- Numbering: take the next number AFTER `git pull --rebase`. Prefer
  `python3 scripts/mail_next_number.py --sender codex --recipient claude
  --slug short-subject` so `origin/epoch2` and the local checkout are both
  considered. If two historical messages already collide on a number, keep
  both (slugs differ) and continue from max+1; never deliberately reuse an
  occupied number.
- Reply by new file (don't edit others' messages). Mark handled items by
  appending "- [handled <date>] <note>" to YOUR OWN sent messages when the
  other agent confirms, or send a short ack file.
- Urgent same-window signals may still use /home/jethac/spark_tmp/ marker
  files, but durable coordination lives here.
