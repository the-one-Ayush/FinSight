from huggingface_hub import InferenceClient
import os

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN
)


def generate_queries(query):

    prompt = f"""
Generate 3 short search queries for retrieving documents.

Rules:
- Do NOT change the meaning of the question
- Use financial synonyms when useful
- Keep queries short

Question:
{query}

Queries:
"""

    response = client.chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.2
    )

    text = response.choices[0].message.content

    queries = [q.strip("- ").strip() for q in text.split("\n") if q.strip()]

    return queries