from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    question: str
    conversation_history: List[Dict[str, str]]
    rewritten_query: str
    retrieved_chunks: List[Dict[str, Any]]
    verifier_verdict: str  # supported | partially_supported | conflicting_evidence | insufficient_evidence
    verifier_reasoning: str
    final_answer: str
    citations: List[str]   # List of cited chunk_ids
    agent_status: List[str] # Progress log for UI status indicators