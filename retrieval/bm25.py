from rank_bm25 import BM25Okapi

def load_bm25():
    with open("data/chunks.txt", "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines()]

    tokenized = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized)

    return bm25, docs


def bm25_search(query, k=5):
    bm25, docs = load_bm25()

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [docs[i] for i in top_indices]