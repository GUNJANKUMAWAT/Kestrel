from src.graph import build_research_graph

if __name__ == "__main__":
    app = build_research_graph()
    
    test_state = {
        "question": "What is a Beacon and how does alerting work?",
        "conversation_history": [],
        "rewritten_query": "",
        "retrieved_chunks": [],
        "verifier_verdict": "",
        "verifier_reasoning": "",
        "final_answer": "",
        "citations": [],
        "agent_status": []
    }
    
    result = app.invoke(test_state)
    
    print("\n=== AGENT STEPS ===")
    for step in result["agent_status"]:
        print(f" -> {step}")
        
    print("\n=== VERIFIER VERDICT ===")
    print(f"Verdict: {result['verifier_verdict']}")
    print(f"Reasoning: {result['verifier_reasoning']}")
    
    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])
    
    print("\n=== CITATIONS ===")
    print(result["citations"])