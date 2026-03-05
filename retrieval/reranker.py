from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, contexts, top_k=5):

    if len(contexts) == 0:
        return []

    texts = [c[0] for c in contexts]

    pairs = [[query, text] for text in texts]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(contexts, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [r[0] for r in ranked[:top_k]]