import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection("financial_docs")


def mmr(query_embedding, doc_embeddings, docs, metas, k=5, lambda_param=0.7):

    if len(docs) == 0:
        return [], []

    query_embedding = np.array(query_embedding).reshape(1, -1)
    doc_embeddings = np.array(doc_embeddings)

    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    selected_indices = []
    candidate_indices = list(range(len(docs)))

    while len(selected_indices) < min(k, len(docs)):

        if not selected_indices:
            idx = int(np.argmax(similarities))
        else:
            mmr_scores = []
            for i in candidate_indices:
                relevance = similarities[i]
                diversity = max(
                    cosine_similarity(
                        doc_embeddings[i].reshape(1, -1),
                        doc_embeddings[selected_indices]
                    )[0]
                )

                score = lambda_param * relevance - (1 - lambda_param) * diversity
                mmr_scores.append(score)

            idx = candidate_indices[int(np.argmax(mmr_scores))]

        selected_indices.append(idx)
        candidate_indices.remove(idx)

    selected_docs = [docs[i] for i in selected_indices]
    selected_metas = [metas[i] for i in selected_indices]

    return selected_docs, selected_metas


def retrieve(query, k=5):

    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=25,
        include=["documents", "metadatas"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    print("\nInitial retrieval count:", len(documents))

    if len(documents) == 0:
        print("No documents retrieved.")
        return []

    embeddings = model.encode(documents)

    selected_docs, selected_metas = mmr(
        query_embedding,
        embeddings,
        documents,
        metadatas,
        k=k
    )

    contexts = []

    for doc, meta in zip(selected_docs, selected_metas):
        page = meta["page"] if meta and "page" in meta else "Unknown"
        contexts.append((doc, page))

    return contexts