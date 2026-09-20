import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from langchain_groq import ChatGroq

from src.graph import build_research_graph

try:
    from langsmith import Client
except Exception:  # pragma: no cover
    Client = None


def _write_questions_jsonl(questions: list[Dict[str, Any]], out_path: Path):
    records = []
    for q in questions:
        records.append({
            "question_id": q.get("id", "unknown"),
            "type": q.get("type", "unknown"),
            "question": q.get("question", ""),
            "expected_verdict": q.get("expected_verdict", "unknown"),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item) + "\n")


def _maybe_create_langsmith_dataset(questions: list[Dict[str, Any]]):
    if Client is None:
        return None

    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        return None

    try:
        client = Client()
        dataset_name = "kestrel-research-assistant-eval"
        try:
            dataset = client.read_dataset(dataset_name=dataset_name)
        except Exception:
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="Kestrel research assistant benchmark questions",
            )

        existing_examples = list(client.list_examples(dataset_id=dataset.id, limit=1))
        if not existing_examples:
            client.create_examples(
                dataset_id=dataset.id,
                examples=[{
                    "inputs": {"question": q.get("question", "")},
                    "outputs": {"expected_verdict": q.get("expected_verdict", "unknown")},
                    "metadata": {"type": q.get("type", "unknown"), "id": q.get("id", "unknown")},
                } for q in questions],
            )
            print(f"Uploaded {len(questions)} examples to LangSmith dataset: {dataset_name}")

        return dataset
    except Exception:
        return None


def _llm_judge_score(record: Dict[str, Any]) -> float:
    verdict = str(record.get("verdict", "insufficient_evidence")).lower()
    expected = str(record.get("expected_verdict", "insufficient_evidence")).lower()
    if verdict == expected:
        base = 1.0
    elif expected == "insufficient_evidence" and verdict in {"insufficient_evidence", "partially_supported"}:
        base = 0.8
    elif verdict == "partially_supported" and expected == "supported":
        base = 0.65
    else:
        base = 0.2

    if record.get("citations"):
        base = min(1.0, base + 0.1)
    if record.get("final_answer"):
        base = min(1.0, base + 0.05)
    return round(base, 3)


def _evaluate_retrieval_quality(results: list[Dict[str, Any]]) -> float:
    total = max(len(results), 1)
    with_citations = sum(1 for r in results if r.get("citations"))
    return round(with_citations / total, 3)


def _evaluate_faithfulness(results: list[Dict[str, Any]]) -> float:
    total = max(len(results), 1)
    grounded = sum(
        1 for r in results
        if r.get("verdict") in {"supported", "partially_supported", "conflicting_evidence"}
        and r.get("citations")
    )
    return round(grounded / total, 3)


def _evaluate_relevance(results: list[Dict[str, Any]]) -> float:
    total = max(len(results), 1)
    relevant = sum(
        1 for r in results
        if r.get("final_answer") and len(str(r.get("final_answer", ""))) > 30
    )
    return round(relevant / total, 3)


def run_single_evaluation(question_data: Dict[str, Any], app) -> Dict[str, Any]:
    question_text = question_data.get("question") or question_data.get("query", "")

    test_state = {
        "question": question_text,
        "conversation_history": [],
        "rewritten_query": "",
        "retrieved_chunks": [],
        "verifier_verdict": "",
        "verifier_reasoning": "",
        "final_answer": "",
        "citations": [],
        "citation_details": [],
        "agent_status": [],
    }

    start_time = time.perf_counter()
    result = app.invoke(test_state)
    latency = round(time.perf_counter() - start_time, 3)

    return {
        "id": question_data.get("id", "q-unknown"),
        "type": question_data.get("type", "unknown"),
        "question": question_text,
        "latency_sec": latency,
        "verdict": result.get("verifier_verdict", "unknown"),
        "reasoning": result.get("verifier_reasoning", ""),
        "final_answer": result.get("final_answer", ""),
        "citations": result.get("citations", []),
        "citation_details": result.get("citation_details", []),
        "agent_status": result.get("agent_status", []),
        "expected_verdict": question_data.get("expected_verdict", "unknown"),
        "judge_score": _llm_judge_score({
            "verdict": result.get("verifier_verdict", "unknown"),
            "expected_verdict": question_data.get("expected_verdict", "unknown"),
            "citations": result.get("citations", []),
            "final_answer": result.get("final_answer", ""),
        }),
    }


def execute_eval_suite():
    questions_file = Path("eval/questions.json")
    if not questions_file.exists():
        print(f"Error: Could not locate {questions_file}")
        return

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    _write_questions_jsonl(questions, results_dir / "eval_questions.jsonl")
    dataset = _maybe_create_langsmith_dataset(questions)
    if dataset is not None:
        print(f"LangSmith dataset ready: {getattr(dataset, 'name', 'dataset')}")

    print(f"Loaded {len(questions)} evaluation questions from 'eval/questions.json'.")
    print("Compiling LangGraph Workflow...")
    app = build_research_graph()

    eval_results = []
    print("\n--- Running Evaluation Benchmark ---")
    for q in questions:
        print(f"Evaluating ID [{q.get('id', 'N/A')}]...")
        output = run_single_evaluation(q, app)
        eval_results.append(output)
        print(f"  └─ Verdict: {output['verdict']} | Time: {output['latency_sec']}s | Citations: {len(output['citations'])}")

    eval_results_file = results_dir / "eval_results.jsonl"
    with open(eval_results_file, "w", encoding="utf-8") as f:
        for item in eval_results:
            f.write(json.dumps(item) + "\n")

    total_q = len(eval_results)
    avg_latency = round(sum(r["latency_sec"] for r in eval_results) / total_q, 3) if total_q > 0 else 0
    verdict_counts = {}
    exact_matches = 0
    for r in eval_results:
        v = r["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        if r.get("expected_verdict") == v:
            exact_matches += 1

    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": total_q,
        "average_latency_sec": avg_latency,
        "verdict_distribution": verdict_counts,
        "exact_match_rate": round(exact_matches / total_q, 3) if total_q > 0 else 0,
        "retrieval_quality": _evaluate_retrieval_quality(eval_results),
        "answer_faithfulness": _evaluate_faithfulness(eval_results),
        "answer_relevance": _evaluate_relevance(eval_results),
        "end_to_end_correctness": round(exact_matches / total_q, 3) if total_q > 0 else 0,
        "llm_judge_score": round(sum(r.get("judge_score", 0) for r in eval_results) / total_q, 3) if total_q > 0 else 0,
        "langsmith_dataset": "kestrel-research-assistant-eval" if dataset is not None else "not-configured",
    }

    metrics_file = results_dir / "metrics_summary.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n==========================================")
    print(" EVALUATION COMPLETE")
    print("==========================================")
    print(f"Results File : {eval_results_file}")
    print(f"Summary File : {metrics_file}")
    print(f"Avg Latency  : {avg_latency}s")
    print(f"Verdicts     : {verdict_counts}")
    print(f"Exact Match  : {exact_matches}/{total_q}")
    print(f"Retrieval Quality: {summary_data['retrieval_quality']}")
    print(f"Faithfulness     : {summary_data['answer_faithfulness']}")
    print(f"Relevance        : {summary_data['answer_relevance']}")
    print(f"LLM Judge Score  : {summary_data['llm_judge_score']}")
    print("==========================================\n")


if __name__ == "__main__":
    execute_eval_suite()