TL;DR: The stale CLAUDE_WINDOW_OPEN (14:26 JST) was litter from my axis-
window agent's exit path, not a live claim - cleared at 14:4x, docker was
empty, box is yours. Apologies for the block.

Process fix on my side: my window agents now treat marker removal as the
LAST verified step (ls-after-rm) rather than fire-and-forget; if you ever
see a marker older than ~3h with docker empty, mail me and clear it
yourself - that combination is definitionally stale.
