import json
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from src.indexer import get_vector_store
from src.state import AgentState

try:
    from langsmith import traceable
except Exception:  # pragma: no cover
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return decorator

load_dotenv()


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing!")

    env_model = os.getenv("GROQ_MODEL")
    candidate_models = []
    if env_model:
        candidate_models.append(env_model)

    candidate_models.extend([
        "qwen/qwen3.8-27b",
        "groq/compound-mini",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ])

    deduped_models = []
    seen = set()
    for model_name in candidate_models:
        if model_name and model_name not in seen:
            deduped_models.append(model_name)
            seen.add(model_name)

    last_error = None
    for model_name in deduped_models:
        try:
            llm = ChatGroq(
                model_name=model_name,
                temperature=0.0,
                groq_api_key=api_key,
            )
            llm.invoke("ping")
            return llm
        except Exception as exc:  # pragma: no cover
            last_error = exc
            print(f"Warning: Groq model '{model_name}' failed. Trying next available model. Error: {type(exc).__name__}: {exc}")
            continue

    if last_error:
        print(f"Warning: all Groq models failed; using final fallback. Error: {last_error}")

    return ChatGroq(
        model_name=deduped_models[0] if deduped_models else "qwen/qwen3.8-27b",
        temperature=0.0,
        groq_api_key=api_key,
    )


def _safe_llm_response(prompt: str, *, stage: str, fallback: str = "I could not complete this step because the LLM service was unavailable.") -> str:
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as exc:  # pragma: no cover - environment-dependent path
        print(f"Warning: LLM call failed during {stage}: {type(exc).__name__}: {exc}")
        return fallback


def _coalesce_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(
        chunks,
        key=lambda c: (c.get("published") or "", c.get("version") or ""),
        reverse=True,
    )
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for chunk in ordered:
        key = chunk.get("chunk_id") or f"{chunk.get('doc_id')}:{chunk.get('title')}:{chunk.get('text')[:80]}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


@traceable(name="planner_node")
def planner_node(state: AgentState) -> Dict[str, Any]:
    history = state.get("conversation_history", [])
    question = state["question"]
    status = state.get("agent_status", []) + ["Planner: Decomposing Query & Resolving Context"]

    if history and len(history) > 0:
        llm = get_llm()
        prompt = f"""Given the conversation history and follow-up question, output one self-contained documentation search query.

History:
{json.dumps(history, indent=2)}

Follow-up: {question}

Search Query (output ONLY the string):"""
        res = llm.invoke(prompt)
        rewritten = res.content.strip().strip('"').strip("'")
    else:
        rewritten = question

    return {"rewritten_query": rewritten, "agent_status": status}


@traceable(name="retriever_node")
def retriever_node(state: AgentState) -> Dict[str, Any]:
    query = state["rewritten_query"]
    status = state.get("agent_status", []) + [f"Retriever: Searching ChromaDB for '{query}'"]

    vectorstore = get_vector_store()
    results = vectorstore.similarity_search(query, k=8)

    chunks = []
    for doc in results:
        chunks.append({
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "doc_id": doc.metadata.get("doc_id", ""),
            "title": doc.metadata.get("title", ""),
            "category": doc.metadata.get("category", ""),
            "published": doc.metadata.get("published", ""),
            "version": doc.metadata.get("version", ""),
            "text": doc.page_content,
        })

    ordered = _coalesce_chunks(chunks)
    return {"retrieved_chunks": ordered, "agent_status": status}


@traceable(name="verifier_node")
def verifier_node(state: AgentState) -> Dict[str, Any]:
    question = state["question"]
    chunks = state.get("retrieved_chunks", [])
    status = state.get("agent_status", []) + ["Verifier: Auditing retrieved chunks against question"]

    if not chunks:
        return {
            "verifier_verdict": "insufficient_evidence",
            "verifier_reasoning": "No relevant chunks were retrieved from the local corpus.",
            "citations": [],
            "agent_status": status,
        }

    context_text = ""
    for c in chunks:
        context_text += f"\n--- Chunk [{c['chunk_id']}] (Doc: {c['doc_id']}, Category: {c.get('category', 'N/A')}, Published: {c.get('published', 'N/A')}, Version: {c.get('version', 'N/A')}) ---\n{c['text']}\n"

    prompt = f"""You are a strict evidence auditor for an internal knowledge base.

User Question: {question}

Context:
{context_text}

Instructions:
1. Return verdict as one of: supported, partially_supported, conflicting_evidence, insufficient_evidence.
2. supported = the context directly answers the question with clear evidence.
3. partially_supported = the context gives a likely answer but misses important details or a direct confirmation.
4. conflicting_evidence = the context contains relevant but opposing statements, usually due to versioning or outdated docs.
5. insufficient_evidence = the context does not answer the question or is too weak.
6. Use exact chunk IDs from the context in citations.
7. Prefer the newest published records when the same fact appears in multiple versions.
8. If the answer is missing or the question is outside the corpus, return insufficient_evidence.
9. Reply with valid JSON only.

JSON schema:
{{
  "verdict": "supported | partially_supported | conflicting_evidence | insufficient_evidence",
  "reasoning": "brief justification",
  "citations": ["chunk_id_1", "chunk_id_2"]
}}"""

    raw = _safe_llm_response(
        prompt,
        stage="verifier",
        fallback='{"verdict": "insufficient_evidence", "reasoning": "The LLM provider was unavailable or rate-limited, so the answer could not be fully verified.", "citations": []}',
    )
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
    except Exception:
        data = {}

    verdict = str(data.get("verdict", "insufficient_evidence")).lower()
    if verdict not in {"supported", "partially_supported", "conflicting_evidence", "insufficient_evidence"}:
        verdict = "insufficient_evidence"

    reasoning = data.get("reasoning", "The retrieved context was reviewed for direct evidence and answer coverage.")
    citations = data.get("citations", [])
    if not citations and verdict == "supported":
        citations = [c["chunk_id"] for c in chunks[:2] if c.get("chunk_id")]

    return {
        "verifier_verdict": verdict,
        "verifier_reasoning": reasoning,
        "citations": citations,
        "agent_status": status,
    }


@traceable(name="synthesizer_node")
def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    verdict = state.get("verifier_verdict", "insufficient_evidence")
    question = state["question"]
    chunks = state.get("retrieved_chunks", [])
    citations = state.get("citations", [])
    status = state.get("agent_status", []) + ["Synthesizer: Generating final answer"]

    if verdict == "insufficient_evidence":
        final_answer = "I could not answer this reliably from the available Kestrel documentation. The corpus does not contain enough direct evidence, or the question is outside the documented scope."
        return {"final_answer": final_answer, "agent_status": status}

    if verdict == "conflicting_evidence":
        final_answer = "The available documentation contains conflicting or version-dependent information, so I cannot give a single definitive answer without reconciling newer sources first."
        return {"final_answer": final_answer, "agent_status": status}

    cited_chunks = [c for c in chunks if c.get("chunk_id") in citations] or chunks[:3]
    context = "\n\n".join([f"[{c['chunk_id']}]: {c['text']}" for c in cited_chunks])

    prompt = f"""Use only the provided evidence to answer the question. Cite each material claim with the exact chunk_id in the format [doc_id:chunk_id].

Question: {question}

Evidence:
{context}

Answer:"""

    answer = _safe_llm_response(
        prompt,
        stage="synthesizer",
        fallback="I could not generate a reliable final answer because the LLM service was unavailable or rate-limited.",
    )

    return {"final_answer": answer, "agent_status": status}


def build_research_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "verifier")
    workflow.add_edge("verifier", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()