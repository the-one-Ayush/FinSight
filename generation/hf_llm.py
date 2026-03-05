import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN
)

def generate_answer(query, contexts):

    trimmed_contexts = [c[:500] for c in contexts]

    context_text = "\n\n".join(
        [f"[{i+1}] {doc}" for i, doc in enumerate(trimmed_contexts)]
    )



    system_prompt = f"""
Question:
{query}

Sources:
{contexts}

Instructions:
1. Identify the sentence that directly answers the question.
2. Quote the relevant part.
3. Give the final answer and Cite the page number ONLY.
4. Do NOT include numbered citations like [1], [2], etc.
5. Do NOT add extra explanations.
6. If the answer is not present, respond exactly with: Not found in documents.

Answer format:
Answer: <answer>
Source: Page <X>
"""

    user_prompt = f"""
    Sources:
    {context_text}

    Question:
    {query}

    Provide the answer using the sources.
    """

    try:
        completion = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.2
        )

        if completion and completion.choices:
            return completion.choices[0].message.content.strip()

        return "Model returned empty response."

    except Exception as e:
        return f"LLM error: {str(e)}"