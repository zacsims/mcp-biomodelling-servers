# Digest update — fold iteration {{FALLING_OFF_ITER}} into the running digest

You are compressing calibration history for a long-running autoregressive
loop. Your ONLY job is to produce an updated `digest.md` by folding the
record that just fell off the recent-window into the previous digest.

## Inputs

- Previous digest (may be empty on the first call):

```markdown
{{PREVIOUS_DIGEST}}
```

- Record that is falling out of the recent window (iteration {{FALLING_OFF_ITER}}):

```json
{{FALLING_OFF_ITER_RECORD}}
```

- Human-readable delta report for the same iteration:

```markdown
{{FALLING_OFF_DELTA_REPORT}}
```

## Output format — strict

Output ONLY the contents of the new `digest.md`. No preamble, no commentary.
The file has EXACTLY these H2 sections, in this order, and nothing else:

```markdown
## Parameter trajectory
<one bullet per currently-tracked parameter: "<name>: <brief trajectory>"
 e.g. "tumor.motility_speed: 0.3 → 0.15 → 0.1, bounded below by adhesion">

## Tier 1 score trend
<one bullet per metric family (CLQ, Ripley's H, Gi*, assortativity, etc.)
 summarizing pass/fail movement across iterations>

## Tier 2 score trend
<one or two bullets on population count and proportion trends>

## Stuck modes observed
<bullets for metrics that have plateaued, oscillated, or regressed;
 include iteration ranges. "(none)" if clean.>

## Registry changes so far
<chronological bullets of add/modify/remove with brief reasons. Compress
 aggressively — the full log lives in registry.jsonl.>

## Open questions / unresolved anomalies
<bullets for things the next iteration should investigate; "(none)" if clean.>
```

## Rules

- Keep total length under ~800 tokens. If you approach that limit, compress
  the oldest entries in each section first.
- Do NOT invent content. Only carry forward what the previous digest or
  the falling-off record assert.
- Do NOT include iteration numbers past {{FALLING_OFF_ITER}} — iterations
  in the current window are NOT your responsibility.
- If the previous digest is empty, this is the first digest update; produce
  all sections with content sourced only from the falling-off record.
- Prefer concrete numbers over qualitative descriptions when both fit.
