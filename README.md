A RAG based system that allows users to query financial reports and obtain answers with verifiable citations. The system retrieves relevant information from financial documents and generates concise answers grounded in the source material.


<img width="1278" height="786" alt="image" src="https://github.com/user-attachments/assets/16f7ec9a-7140-4cf9-b751-adbc69b7a79f" />


**Tech Stack**
**Programming Language:**
Python

**LLMS:**
Meta Llama-3.1-8B-Instruct (via Hugging Face API)

**Retrieval & Embeddings**
SentenceTransformers (all-MiniLM-L6-v2)
ChromaDB – vector database
BM25 – keyword-based retrieval

**Ranking**
Cross-Encoder Reranker (SentenceTransformers) (all-MiniLM-L6-v2)
Numpy
Scikit-learn

**Document Processing**
PyMuPDF
NLTK

**Frontend**
Streamlit
