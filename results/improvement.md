# Improvement case study

## Baseline

The first implementation used a simple two-step retrieval and verification loop with a single verdict category and minimal chunk selection. It retrieved a handful of semantically similar chunks, but it lacked explicit handling for version conflicts, recency precedence, and unsupported questions.

## Optimization driven by evaluation

We improved the pipeline by:

1. Adding a stronger verifier rubric with `supported`, `partially_supported`, `conflicting_evidence`, and `insufficient_evidence`.
2. Sorting candidate chunks by recency using the `published` metadata before finalizing evidence.
3. Deduplicating results and preferring newer versions for the same doc.
4. Requiring explicit refusal when there is not enough direct evidence.
5. Adding a follow-up-aware planner prompt for multi-turn questions.

## Result

The refined workflow is more faithful to the corpus and more conservative when evidence is weak. This is the correct behavior for a product documentation assistant because a refusal is safer than a made-up answer.

## Evidence from the new evaluation set

The evaluation dataset now covers single-hop, multi-hop, conflicting-evidence, unsupported, and multi-turn follow-up cases. This gives a stronger signal than the earlier three-question benchmark, which was too narrow to capture real failure modes.
