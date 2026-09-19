# Kestrel Research Assistant Design

## Architecture overview

The system is a grounded multi-agent research workflow built around a LangGraph state machine.

### Agents

1. Planner
   - Reformulates follow-up questions using conversation history.
   - Produces one search query optimized for retrieval.

2. Retriever
   - Uses a local Chroma vector store backed by `sentence-transformers` embeddings.
   - Retrieves the top relevant document chunks from `corpus.jsonl`.

3. Verifier
   - Audits the retrieved chunks for directness, completeness, and conflicts.
   - Assigns one verdict: supported, partially_supported, conflicting_evidence, or insufficient_evidence.
   - Returns exact `chunk_id` citations.

4. Synthesizer
   - Produces the final answer only when the evidence is sufficient.
   - Uses the cited chunks to keep the response grounded and explicit.

## Workflow diagram

```text
User Query
   |
   v
Planner -> Retriever -> Verifier -> Synthesizer -> Final Answer
                 \_______________________________/
                              Evidence check
```

## Design principles

- Retrieval before answer generation
- Direct citations for every factual claim
- Newer document versions win on conflicts
- Explicit refusals when the corpus is weak or missing
- Local embeddings only, per assignment constraints

## State handoff

The shared state contains:

- `question`
- `conversation_history`
- `rewritten_query`
- `retrieved_chunks`
- `verifier_verdict`
- `verifier_reasoning`
- `final_answer`
- `citations`
- `agent_status`

This flow keeps the evidence trace explicit and allows UI status updates and evaluation logging.
