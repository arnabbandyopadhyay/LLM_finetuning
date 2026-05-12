# LLM pipeline with RAG and QLoRA Fine-Tuning

This project builds a complete Large Language Model (LLM) research pipeline:

- Fetches recent LLM papers from arXiv
- Downloads and extracts PDF text
- Chunks documents for Retrieval-Augmented Generation (RAG)
- Builds a FAISS vector database
- Generates QA datasets automatically
- Fine-tunes an open-source LLM using QLoRA
- Evaluates semantic similarity and citation quality
  
```text
arXiv Papers
     ↓
Download PDFs
     ↓
Extract Text
     ↓
Chunk Documents
     ↓
Create Embeddings
     ↓
Store in FAISS Vector DB
     ↓
Generate QA Dataset
     ↓
Fine-Tune LLM (QLoRA)
     ↓
Evaluate Model
```

## Research Paper Collection: fetch_llm_papers()
- connects to arXiv API
- Filters LLM-related papers (keywords: LLM, GPT, Transformers, RLHF, Instruction tuning)
- Keeps papers published after 2023
Output:
list of papers with:
- title
- summary
- PDF url
- year


## PDF Processing: download_and_extract()
- Downloads PDFs automatically
- extracts text using PyMuPDF (`fitz`)
- reads all pages
- extracts plain text
output:
{
'text':'...',
'metadata':{....}
}

## chunking and vector DB
- chunks documents using LangChain: chunk_documents()
- creates embeddings (converts text chunks into vectors)
- stores vectors in FAISS: build_vector_db(). This enables semantic search.

## RAG Pipeline: query_rag()
What it does:
- searches vector DB
- finds top relevant chunks
- Returns:
-  context
-  citations

## QA Dataset Generation: generate_qa_dataset()
- automatically generates question-answer pairs using GPT-2
- saves data in JSONL format
- adds source citations

## Fine-Tuning
- uses QLoRA + PEFT
- Fine-tunes `google/gemma-2b-it`
- supports GPU acceleration

## Evaluation
- semantic similarity scoring
- citation scoring
- leaderboard comparison across models

---

# Project Structure

```bash
project/
│
├── llm_papers/                 # Temporary downloaded PDFs
├── llm_qa.jsonl                 # Generated QA dataset
├── llm_finetuned_final/       # Fine-tuned model output
├── main.py                      # Main pipeline
├── requirements.txt
└── README.md
