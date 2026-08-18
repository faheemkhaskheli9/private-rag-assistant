# Architecture Notes: Private GPT with RAG

## Pipeline

```text
PDF -> Chunk -> Embed -> Vector DB -> Retrieve -> LLM Answer + Citations -> Chat UI
```

## Components

- PDF upload
- Chunking
- Embedding generation
- Vector database storage
- Semantic retrieval
- Answer citations
- Local or cloud LLM backend option
- Persistent chat memory

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.
