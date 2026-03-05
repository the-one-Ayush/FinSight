import ollama

def compress_context(query, contexts):

    compressed = []

    for chunk in contexts:

        prompt = f"""
Extract only the sentences relevant to the question.

Question:
{query}

Text:
{chunk}

Relevant sentences:
"""

        response = ollama.chat(
            model="phi3",
            messages=[{"role": "user", "content": prompt}]
        )

        result = response["message"]["content"].strip()

        if result:
            compressed.append(result)

    return compressed