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
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=kestrel-research-assistant
   ```

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
