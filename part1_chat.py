from openai import OpenAI
from config import VLLM_URL, VLLM_MODEL, VLLM_KEY

# Connect to the AI server
client = OpenAI(base_url=VLLM_URL, api_key=VLLM_KEY)

# The question we want to ask
question = "What is 2 + 2?"

print("\n" + "=" * 60)
print("  Part 1 - Talk to an AI")
print("=" * 60 + "\n")

print("--- Question ---")
print(question)
print()

# Send the question to the AI and get its answer
reply = client.chat.completions.create(
    model=VLLM_MODEL,
    messages=[{"role": "user", "content": question}],
    max_tokens=300,
    temperature=0.7,
)

# Print whatever the AI replied
print("--- Answer ---")
print(reply.choices[0].message.content)
print()
