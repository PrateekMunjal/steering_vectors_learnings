from openai import OpenAI
from config import VLLM_URL, VLLM_MODEL, VLLM_KEY

# Connect to the AI server
client = OpenAI(base_url=VLLM_URL, api_key=VLLM_KEY)

# The prompt — same one we used in part1_chat.py
question = "What is 2 + 2?"

print("\n" + "=" * 60)
print("  Part 1 - Base Model vs Instruct Model")
print("=" * 60 + "\n")

print("--- Prompt ---")
print(question)
print()

# ── Base model: raw text completion ──────────────────────────
# The /completions endpoint just predicts the next tokens.
# No chat template, no "you are a helpful assistant" wrapping.
# The model sees the text and continues it — it doesn't know
# it's supposed to answer.

base_reply = client.completions.create(
    model=VLLM_MODEL,
    prompt=question,
    max_tokens=150,
    temperature=0.7,
)

print("--- Base model (raw completion) ---")
print(base_reply.choices[0].text)
print()

# ── Instruct model: chat completion ─────────────────────────
# The /chat/completions endpoint wraps the prompt in a chat
# template: system message, user/assistant turns. The model
# has been fine-tuned to expect this format and respond helpfully.

chat_reply = client.chat.completions.create(
    model=VLLM_MODEL,
    messages=[{"role": "user", "content": question}],
    max_tokens=150,
    temperature=0.7,
)

breakpoint()

print("--- Instruct model (chat) ---")
print(chat_reply.choices[0].message.content)
print()

print("-" * 60)
print("  Same model, same prompt — totally different output.")
print("  The base model continues the text.")
print("  The instruct model answers the question.")
print("  That difference is instruction tuning.")
print("-" * 60 + "\n")
