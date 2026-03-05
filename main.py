from retrieval.vector_store import retrieve
from generation.hf_llm import generate_answer
from retrieval.bm25 import bm25_search
from retrieval.reranker import rerank
from retrieval.multi_query import generate_queries

def get_answer(query):

    queries = [query] + generate_queries(query)

    all_results = []

    for q in queries:

        vector_results = retrieve(q)
        bm25_results = bm25_search(q)

        combined = list(dict.fromkeys(vector_results + bm25_results))

        all_results.extend(combined)

    all_results = list(dict.fromkeys(all_results))

    reranked = rerank(query, all_results)

    contexts = reranked[:3]

    formatted_contexts = [
        f"[Page {page}]\n{text}" for text, page in contexts
    ]

    answer = generate_answer(query, formatted_contexts)
    answer= answer.replace("Source:", "\nSource:")
    return answer, contexts