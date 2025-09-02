# test_javabench_openai.py

from openai import OpenAI
from javabench import BenchmarkRunner, DatasetRegistry

# ---- Setup OpenAI ----
client = OpenAI()

def vanilla_openai_system(query: str) -> str:
    """Simple system: directly ask GPT-4.1 without any retrieval"""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ---- Load a tiny Javabench dataset ----
# (use one of their built-in test datasets)
dataset = DatasetRegistry.get("hotpot_qa_small")  # ~20 Q&A pairs

# ---- Run benchmark ----
runner = BenchmarkRunner(
    dataset=dataset,
    system_fn=vanilla_openai_system
)

results = runner.run()

print("=== SUMMARY ===")
print(results.summary())
