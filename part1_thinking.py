from openai import OpenAI
from config import VLLM_URL, VLLM_MODEL, VLLM_KEY

# Connect to the AI server
client = OpenAI(base_url=VLLM_URL, api_key=VLLM_KEY)

# The question we want to ask
question = "What is 2 + 2?"

print("\n" + "=" * 60)
print("  Part 1b - Thinking ON vs OFF")
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
    extra_body={"chat_template_kwargs": {"thinking": True}},  # <-- NEW: turn thinking ON
)

# NEW: the AI may have "thought" before answering — show that too
msg = reply.choices[0].message
thinking = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
if thinking:
    print("--- Thinking ---")
    print(thinking)
    print()

# Print whatever the AI replied
print("--- Answer ---")
print(msg.content)
print()
