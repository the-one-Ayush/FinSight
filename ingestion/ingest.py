import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
import re
import uuid
import pdfplumber
import nltk

model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(text):

    # Fix spaced numbers like "2 0 2 4"
    text = re.sub(r'(\d)\s(?=\d)', r'\1', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def chunk_text(text, max_chars=800, overlap=150):

    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += " " + sentence

        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def extract_tables_from_page(pdf_path, page_number):
    sentences = []

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]

        tables = page.extract_tables()
        for table in tables:
            if not table:
                continue

            header = table[0]

            for row in table[1:]:
                if not row:
                    continue

                row_parts = []

                for col, val in zip(header, row):
                    if val:
                        row_parts.append(f"{col} {val}")

                sentence = " ".join(row_parts)

                if sentence.strip():
                    sentences.append(sentence)

    return sentences


def ingest_pdf(pdf_path):

    doc = fitz.open(pdf_path)

    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection("financial_docs")

    all_chunks = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        cleaned_text = clean_text(text)
        chunks = chunk_text(cleaned_text)

        for chunk in chunks:

            embedding = model.encode(chunk).tolist()

            collection.add(
                ids=[str(uuid.uuid4())],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "page": page_num + 1
                }]
            )

            all_chunks.append(f"[Page {page_num+1}] {chunk}")


        table_sentences = extract_tables_from_page(pdf_path, page_num)

        for sentence in table_sentences:

            embedding = model.encode(sentence).tolist()

            collection.add(
                ids=[str(uuid.uuid4())],
                documents=[sentence],
                embeddings=[embedding],
                metadatas=[{
                    "page": page_num + 1
                }]
            )

            all_chunks.append(f"[Page {page_num+1}] TABLE: {sentence}")


    with open("data/chunks.txt", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(c + "\n")

    print(f"Ingestion complete. {len(all_chunks)} chunks added.")


if __name__ == "__main__":
    ingest_pdf("data/raw/amazon-report.pdf")