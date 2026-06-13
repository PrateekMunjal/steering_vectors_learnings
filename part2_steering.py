"""
Part 2 — Steer the model by turning an internal dial.

The idea: we teach the AI what "cheerful" vs "gloomy" sounds like,
then we turn a dial to make it more cheerful or more gloomy.

Run:  python part2_steering.py
      python part2_steering.py --strength=3.0
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlModel, ControlVector, DatasetEntry
from config import STEER_MODEL, STEER_DEVICE

# ----------------------------------------------------------------
# Step 0: What concept are we steering?
# ----------------------------------------------------------------
# "Positive" personas = one side of the dial (cheerful)
# "Negative" personas = the other side (gloomy)
# We give both to the AI so it learns the difference.
PERSONAS_POS = [
    "a very cheerful person",
    "an enthusiastic optimist",
    "someone who loves life",
]
PERSONAS_NEG = [
    "a very gloomy person",
    "a depressed pessimist",
    "someone who hates everything",
]

# Different ways to ask the AI to write something.
# We pair each with each persona to get more training examples.
SUFFIXES = [
    "Tell me about your day.",
    "What do you think of the weather?",
    "How are you feeling?",
]

# The question we will ask the AI after training the steering dial
PROMPT = "Tell me about your weekend plans."

# How much to turn the dial: negative = gloomy, 0 = normal, positive = cheerful
STRENGTHS = [-20.0, 0.0, 20.0]

# Step 1: Read CLI options
strength_override = None
for arg in sys.argv[1:]:
    if arg.startswith("--strength="):
        strength_override = float(arg.split("=")[1])

print("\n" + "=" * 60)
print("  Part 2 - Steering Vectors: Turn the Dial")
print("=" * 60 + "\n")
print(f"Concept: cheerful <-> gloomy")
print(f"Model:   {STEER_MODEL}\n")

# ----------------------------------------------------------------
# Step 2: Load the AI model into memory
# ----------------------------------------------------------------
# Pick the right number format for the hardware
dtype = torch.float32 if STEER_DEVICE == "cpu" else torch.float16

# Load the tokenizer (turns text into numbers the AI understands)
tokenizer = AutoTokenizer.from_pretrained(STEER_MODEL)

# Load the AI model itself
model = AutoModelForCausalLM.from_pretrained(
    STEER_MODEL, device_map=STEER_DEVICE, torch_dtype=dtype,
).eval()

# Figure out where the model ended up (GPU or CPU)
device = STEER_DEVICE if STEER_DEVICE != "auto" else str(model.device)
print(f"Model loaded on {device}\n")

# ----------------------------------------------------------------
# Step 3: Wrap the model so we can "steer" it
# ----------------------------------------------------------------
# We steer the last ~13 layers of the AI (that's where the
# "personality" of the answer lives)
layers = list(range(-5, -18, -1))
control_model = ControlModel(model, layers)

# ----------------------------------------------------------------
# Step 4: Build training pairs — cheerful vs gloomy
# ----------------------------------------------------------------
# Each pair is: "write as a cheerful person" vs "write as a gloomy person"
# The AI learns the difference from these pairs.
template = "You are {persona}. Write a short message."
dataset = []
for pos, neg in zip(PERSONAS_POS, PERSONAS_NEG):
    for s in SUFFIXES:
        dataset.append(DatasetEntry(
            positive=template.format(persona=pos) + " " + s,
            negative=template.format(persona=neg) + " " + s,
        ))

print(f"Training steering vector on {len(dataset)} pairs...")
vector = ControlVector.train(control_model, tokenizer, dataset)
print("Done!\n")

# ----------------------------------------------------------------
# Step 5: Generate text at different dial settings
# ----------------------------------------------------------------
strengths = [strength_override] if strength_override is not None else STRENGTHS

print("--- Prompt ---")
print(PROMPT)
print()

for s in strengths:
    # Turn the dial to this strength
    control_model.set_control(vector, coeff=s)

    # Turn our prompt into numbers and send it to the AI
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = control_model.generate(
            **inputs, max_new_tokens=60,
            do_sample=True, temperature=0.7, top_p=0.9,
        )

    # Turn the AI's numbers back into text
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    # Reset the dial back to normal for the next round
    control_model.reset()

    # Show the result
    print(f"--- strength = {s} ---")
    print(text)
    print()
