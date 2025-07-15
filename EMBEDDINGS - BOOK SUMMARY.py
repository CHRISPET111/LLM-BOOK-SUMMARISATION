#!/usr/bin/env python3
"""
full_book_pipeline.py

This script combines:
1. PDF text extraction (PyMuPDF)
2. Chunking
3. Embedding + FAISS indexing (Sentence-Transformers)
4. Summarization of chunks (Qwen2.5-3B-Instruct as causal LM)
5. Saving combined summary
6. Interactive Q&A using Retrieval-Augmented Generation

Usage:
    python full_book_pipeline.py
"""

import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configuration
PDF_PATH       = r""
MODEL_ID       = "Qwen/Qwen2.5-3B-Instruct"
HF_TOKEN       = ""
OUTPUT_SUMMARY = "book_summary.txt"
MAX_CHARS      = 2000
MAX_NEW_TOKENS = 150
EMBEDDING_MODEL= "all-MiniLM-L6-v2"
TOP_K          = 5

def chunk_text(text: str, max_chars: int = MAX_CHARS):
    """Split text into manageable chunks."""
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def extract_text(pdf_path: str) -> str:
    """Load PDF and extract all text."""
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)

def build_faiss_index(chunks: list) -> (faiss.IndexFlatIP, np.ndarray):
    """Compute embeddings and build FAISS index."""
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embs = embedder.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, embs

def summarize_chunks(chunks: list) -> list:
    """Summarize each chunk with Qwen2.5-3B-Instruct."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN)
    gen       = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )
    summaries = []
    for idx, chunk in enumerate(chunks, 1):
        prompt = f"Summarize the following text:\n\n{chunk}\n\nSummary:"
        out = gen(prompt)[0]["generated_text"]
        summary = out[len(prompt):].strip()
        summaries.append(summary)
        print(f"[+] Summarized chunk {idx}/{len(chunks)}")
    return summaries

def save_summary(summaries: list, filepath: str):
    """Write combined summaries to file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n\n".join(summaries))
    print(f"[+] Summary saved as '{filepath}'")

def answer_question(question: str, chunks: list, index: faiss.IndexFlatIP, embs: np.ndarray) -> str:
    """Retrieve relevant chunks and answer a question via LLM."""
    # Embed question
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    q_emb = embedder.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    _, I = index.search(q_emb, TOP_K)
    context = "\n\n".join(chunks[i] for i in I[0])

    # LLM prompt
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN)
    gen       = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_new_tokens=200,
        do_sample=False
    )
    prompt = (
        "Based on the excerpts below, answer the question.\n\n"
        f"Excerpts:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    out = gen(prompt)[0]["generated_text"]
    return out[len(prompt):].strip()

def main():
    # 1. Extract text and chunk
    print("[*] Extracting text from PDF...")
    text = extract_text(PDF_PATH)
    chunks = chunk_text(text)
    print(f"[+] Extracted and split into {len(chunks)} chunks")

    # 2. Summarize chunks
    summaries = summarize_chunks(chunks)
    save_summary(summaries, OUTPUT_SUMMARY)

    # 3. Build FAISS index
    print("[*] Building FAISS index for RAG...")
    index, embs = build_faiss_index(chunks)

    # 4. Interactive Q&A
    while True:
        q = input("\nEnter a question about the book (or 'exit'): ")
        if q.lower() in ("exit", "quit"):
            break
        answer = answer_question(q, chunks, index, embs)
        print("\n--- Answer ---")
        print(answer)

if __name__ == "__main__":
    main()

