Full Book Processing & QA Pipeline
This Python script provides an end-to-end pipeline to analyze PDF documents and interact with them using Retrieval-Augmented Generation (RAG). It is designed for use cases like summarizing contracts, extracting insights from books, or querying lengthy reports.

Features
PDF Text Extraction using PyMuPDF

Chunking large documents into manageable text blocks

Semantic Embedding with Sentence Transformers (all-MiniLM-L6-v2)

FAISS Indexing for fast similarity search

Summarization of each chunk using Qwen2.5-3B-Instruct

Interactive Q&A with a terminal-based interface

Use Cases
Understanding employment agreements

Summarizing research papers or books

Extracting insights from legal or financial documents

Interactive querying of large PDFs

How It Works
Extract Text from a PDF file

Chunk the text into ~2000 character blocks

Embed the chunks and build a FAISS index

Summarize each chunk using a language model

Save the combined summary to a .txt file

Ask Questions in an interactive terminal session

Requirements
Python 3.8+

PyMuPDF (fitz)

FAISS

Sentence Transformers

Transformers (Hugging Face)

Hugging Face access token

Install dependencies:

bash
Copy
Edit
pip install pymupdf faiss-cpu sentence-transformers transformers
Running the Script
Update the following in the script:

PDF_PATH: Path to your PDF

HF_TOKEN: Your Hugging Face token

Then run:

bash
Copy
Edit
python full_book_pipeline.py
Output
book_summary.txt: Summarized text of the PDF

Q&A session in the terminal

License
MIT License (or choose your own)

