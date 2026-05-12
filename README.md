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
'''

## Research Paper Collection
- Fetches papers from arXiv API
- Filters LLM-related papers
- Keeps papers published after 2023

## PDF Processing
- Downloads PDFs automatically
- Extracts text using PyMuPDF (`fitz`)

## RAG Pipeline
- Chunks documents using LangChain
- Creates embeddings
- Stores vectors in FAISS

## QA Dataset Generation
- Automatically generates question-answer pairs
- Saves data in JSONL format
- Adds source citations

## Fine-Tuning
- Uses QLoRA + PEFT
- Fine-tunes `google/gemma-2b-it`
- Supports GPU acceleration

## Evaluation
- Semantic similarity scoring
- Citation scoring
- Leaderboard comparison across models

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
