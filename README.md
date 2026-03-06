A RAG based system that allows users to query financial reports and obtain answers with verifiable citations. The system retrieves relevant information from financial documents and generates concise answers grounded in the source material.


<img width="1278" height="786" alt="image" src="https://github.com/user-attachments/assets/16f7ec9a-7140-4cf9-b751-adbc69b7a79f" />

<br/>
<br/>
<br/>

# **Tech Stack**<br/>
**Programming Language:**<br/>
Python<br/>

**LLM for Answer Generation:**<br/>
Meta Llama-3.1-8B-Instruct (via Hugging Face API)<br/>

**Retrieval & Embeddings**<br/>
SentenceTransformers (all-MiniLM-L6-v2)<br/>
ChromaDB – vector database<br/>
BM25 – keyword-based retrieval<br/>

**Ranking**<br/>
Cross-Encoder Reranker (SentenceTransformers) (all-MiniLM-L6-v2)<br/>
Numpy<br/>
Scikit-learn<br/>

**Document Processing**<br/>
PyMuPDF<br/>
NLTK<br/>

**Frontend**<br/>
Streamlit<br/>
