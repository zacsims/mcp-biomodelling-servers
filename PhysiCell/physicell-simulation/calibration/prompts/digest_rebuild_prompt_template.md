# Digest rebuild — reconstruct digest.md from archive

You are correcting accumulated drift in the running calibration digest.
Rebuild `digest.md` from scratch using the provided sample of archived
iteration records, ignoring any prior digest.

## Inputs

- Archive sample (chronological; may be sparse beyond the most recent
  iterations, which are included in full):

{{ARCHIVE_SAMPLE}}

- Iterations currently in the recent window (EXCLUDED from the digest —
  those live in the recent-window context, not here):

    {{WINDOW_ITERS_EXCLUDED}}

## Output format — strict

Same schema as the incremental update. Output ONLY the contents of the
new `digest.md`, no preamble, no commentary:

```markdown
## Parameter trajectory
## Tier 1 score trend
## Tier 2 score trend
## Stuck modes observed
## Registry changes so far
## Open questions / unresolved anomalies
```

## Rules

- Total length under ~800 tokens.
- Do NOT include any iteration listed in WINDOW_ITERS_EXCLUDED.
- Do NOT invent content; source everything from the archive sample.
- Prefer concrete numbers over qualitative descriptions when both fit.
- If a section has nothing to report, write "(none)".
