# Kestrel Research Assistant

A grounded multi-agent research assistant for Kestrel product documentation. It uses a local Chroma vector database, a LangGraph workflow, and a Groq LLM to answer questions using only retrieved evidence from the local corpus.

## Setup

1. Create a Python environment using Python 3.11+.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with the following values:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=qwen/qwen3.8-27b
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=kestrel-research-assistant
   ```

   Set `GROQ_MODEL` to a model that exists in your Groq account. If it is omitted, the app will try a small set of common models and degrade gracefully if the provider is unavailable or rate-limited.

4. Build the vector database if needed:

   ```bash
   python -c "from src.indexer import build_vector_store; build_vector_store()"
   ```

## Run the app

```bash
streamlit run app.py
```

## Run evaluation

```bash
python eval/run_eval.py
```

## LangSmith evaluation

- Public dataset: [kestrel-research-assistant-eval](https://smith.langchain.com/public/05d56625-3d0e-4acc-82dc-a76a13b47f5e/d)
- Tracing project: `Kestrel Research Agent`
- The dataset contains 20 benchmark questions with expected verdicts.
- LangSmith access may require signing in to the workspace that owns the dataset.

## Project structure

- `corpus.jsonl`: source documentation corpus; kept immutable
- `src/graph.py`: LangGraph multi-agent workflow
- `src/indexer.py`: local Chroma + SentenceTransformers indexing
- `app.py`: Streamlit frontend
- `eval/questions.json`: evaluation questions
- `results/`: artifacts and summaries

## Notes

- Retrieval is local and grounded in the supplied corpus.
- The assistant answers only with evidence-backed citations from the corpus.
- When evidence is absent or conflicting, it explicitly refuses to answer instead of guessing.
