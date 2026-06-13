# Implementation Plan: "How LLMs Work & Steering Vectors" Demo

## Purpose & audience
Build a small, runnable codebase for a **live demo to non-technical people**. The
presenter runs the code and talks over it. The narrative arc is:

1. You talk to an AI, it talks back. (baseline API call)
2. It can even "think out loud" before answering. (thinking vs. non-thinking)
3. That thinking is just numbers inside the model — what if we nudged them?
4. Watch: same prompt, but we turn an internal "mood/style dial" up and down. (steering)

Code should be **clear and presentable**, not clever. Heavy comments, obvious
section headers, and print statements that look good projected on a screen.
Each script must be runnable standalone (`python part1_...py`) so the presenter
can step through them one at a time.

---

## CRITICAL ARCHITECTURE NOTE — read before writing any code

This demo uses **two completely separate stacks**. Do not mix them.

| | Part 1 (baseline + thinking) | Part 2 (steering) |
|---|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Pro` (remote) | small **dense** local model |
| How | OpenAI-compatible HTTP API call | HuggingFace `transformers` + `repeng` in-process |
| Where | existing vLLM server at `http://worker-1:8000` | the presenter's own machine |

**Part 2 must NOT call the DeepSeek server.** Steering requires modifying the
model's internal activations during the forward pass. A standard vLLM endpoint
does not expose those internals, so steering through `worker-1:8000` is
impossible by design. Steering needs the model loaded locally via HuggingFace.

---

## HARD CONSTRAINTS (bake these in; they prevent dead ends)

1. **Steering model must be DENSE, not MoE.** `repeng` does not support
   mixture-of-experts models. `DeepSeek-V4-Pro` is MoE — never load it for steering.
2. **Steering model must be SMALL.** Compute is undecided, so default to a model
   that runs on a modest GPU *or* CPU. Make device/dtype configurable.
3. **Do not hardcode the thinking-toggle parameter.** It differs by model family
   (DeepSeek: `thinking`; Qwen: `enable_thinking`; GLM: nested `thinking` object).
   Part 1 must detect/handle this at runtime (see Part 1 spec).
4. **vLLM auth:** the OpenAI client requires *some* api_key. vLLM usually accepts
   any dummy string. Use `api_key="EMPTY"` and allow override via env var.
5. Keep all secrets/URLs in one config block or `.env`, not scattered.

---

## Repo structure
```
steering-demo/
  README.md            # how to run, in order, during the talk
  requirements.txt
  config.py            # server URL, model ids, defaults (env-overridable)
  part1_chat.py        # baseline call to DeepSeek server
  part1_thinking.py    # thinking vs non-thinking, same prompt
  part2_steering.py    # train + apply a steering vector on a small local model
  utils_print.py       # pretty before/after printing for projector
```

---

## Environment & dependencies

`requirements.txt`:
```
openai>=1.0          # Part 1: talks to the vLLM OpenAI-compatible endpoint
transformers>=4.44   # Part 2
torch                # Part 2
accelerate           # Part 2 (device mapping)
repeng               # Part 2 steering vectors
python-dotenv        # config
```
Document in README that Part 1 needs only `openai`, and Part 2 needs the ML
stack — so someone with no GPU can still run Part 1.

`config.py` defaults:
```python
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://worker-1:8000/v1")
VLLM_MODEL    = os.getenv("VLLM_MODEL", "deepseek-ai/DeepSeek-V4-Pro")
VLLM_API_KEY  = os.getenv("VLLM_API_KEY", "EMPTY")

STEER_MODEL   = os.getenv("STEER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")  # dense, small, ungated
STEER_DEVICE  = os.getenv("STEER_DEVICE", "auto")  # "cuda", "cpu", or "auto"
```

---

## Part 1a — Baseline chat (`part1_chat.py`)

Goal: show the simplest possible "text in, text out" call so the audience sees
there's no magic.

- Build an OpenAI client pointed at the vLLM server:
  `OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)`.
- Send one friendly question (e.g. "Explain what a large language model is, like
  I'm five.").
- Print the question and the answer with clear visual separation.

**Acceptance:** running the script prints a readable answer from the DeepSeek
server. If the connection fails, print a clear hint pointing at `VLLM_BASE_URL`.

## Part 1b — Thinking vs. non-thinking (`part1_thinking.py`)

Goal: ask the **same question twice** — once where the model reasons out loud,
once where it just answers — and show the difference side by side. The "thinking"
block is the visual hook for the audience.

Implement a **capability-probe first**, because the exact behavior depends on how
the server was launched and on the model's chat template:

1. Send a request with thinking *enabled* and inspect the response:
   - If `response.choices[0].message` has a populated `reasoning_content` field,
     the server has a reasoning parser configured → use that field directly to
     display the "thinking."
   - Otherwise the reasoning will arrive **inline** inside `content`, wrapped in
     `<think>...</think>` tags → write a small parser that splits the think block
     from the final answer for display.
2. For toggling thinking on/off, pass the flag via `extra_body`:
   ```python
   extra_body={"chat_template_kwargs": {"thinking": True}}   # DeepSeek convention
   ```
   DeepSeek-family models use the key `thinking`. If the server **ignores** the
   key (vLLM logs a warning and the field has no effect), fall back to trying
   `enable_thinking`. Implement a tiny helper `request(thinking: bool)` that tries
   the DeepSeek key first, detects whether thinking actually changed, and falls
   back. Log which key worked so the presenter knows.
3. Display: left = "Model thinking…", right/below = "Final answer", for both the
   thinking-on and thinking-off runs.

**Acceptance:** one run visibly shows a chain-of-thought block plus answer; the
other shows just the answer. The script prints which toggle key it used.

> Note for the implementer: `DeepSeek-V4-Pro` is newer than common docs. Do not
> assume — verify the toggle behavior empirically as described above and leave a
> comment recording what you observed.

---

## Part 2 — Steering vectors (`part2_steering.py`)

Goal: the payoff. Load a small dense model locally, build a steering vector for a
human-legible concept, then generate the **same prompt** at several steering
strengths so the audience watches the output change.

Use `repeng` (simplest API for a live "wow"):

1. Load `STEER_MODEL` with `transformers` (`AutoModelForCausalLM`,
   `AutoTokenizer`), respecting `STEER_DEVICE` and using fp16 on GPU / fp32 on CPU.
2. Wrap it: `ControlModel(model, layer_ids)` over a band of **middle** layers
   (a negative range like `list(range(-5, -18, -1))` works well; expose as a
   constant so it's tunable).
3. Build a contrastive dataset of paired prompts for ONE clear concept. Pick a
   concept that reads instantly to a general audience. Good options (implement
   one as default, leave the others as easy-to-switch presets):
   - **cheerful ↔ gloomy**
   - **formal ↔ casual**
   - **confident ↔ hesitant**
   Provide a `make_dataset(template, pos_personas, neg_personas, suffixes)` helper
   (see repeng's example notebooks for the shape) that produces closely-paired
   positive/negative statements.
4. Train the vector: `vector = ControlVector.train(model, tokenizer, dataset)`.
   (Fast — seconds to a minute on a small model.)
5. Pick one neutral prompt (e.g. "Tell me about your weekend plans.") and
   generate it **three times**: strength `-2.0`, `0.0` (off), `+2.0`. Print all
   three under clear labels so the shift is obvious. Keep generation short
   (`max_new_tokens` ~60) so it's snappy live.
6. Add a tiny interactive option (optional): a `--strength` CLI arg so the
   presenter can take an audience-suggested number and rerun live.

**Acceptance:** same prompt, three strengths, visibly different tone/content;
runs end-to-end on CPU if no GPU is present (just slower).

**Reminders enforced in code/comments:**
- Never point this script at the DeepSeek server.
- Never swap in an MoE model — keep it dense (Qwen2.5-1.5B-Instruct default;
  Mistral-7B-Instruct as a heavier alternative if a GPU is available).

---

## Part 3 — Presentation glue (`utils_print.py`, `README.md`)

- `utils_print.py`: helpers for boxed/labeled output (e.g. a `banner(title)` and
  a `side_by_side(left, right)` printer) so projected text is readable.
- `README.md`: a "run order" for the talk:
  1. `python part1_chat.py` — "it just answers"
  2. `python part1_thinking.py` — "now it shows its work"
  3. `python part2_steering.py` — "now we turn the dial"
  Include the one-paragraph framing/metaphor for each step so the presenter has
  talking points. Note compute: Part 1 = any laptop; Part 2 = GPU preferred, CPU
  works for the 1.5B model.

---

## Testing checklist (implementer runs before handing back)
- [ ] `part1_chat.py` returns a real completion from `worker-1:8000`.
- [ ] `part1_thinking.py` produces a distinct thinking-on vs thinking-off result
      and prints which toggle key worked.
- [ ] Inline `<think>` fallback parser works even if no reasoning parser is set.
- [ ] `part2_steering.py` trains a vector and prints 3 clearly different outputs.
- [ ] `part2_steering.py` runs on CPU (set `STEER_DEVICE=cpu`) without crashing.
- [ ] All config is overridable by env var; no hardcoded secrets.

## Open items to confirm at implementation time
- Exact thinking-toggle key for `DeepSeek-V4-Pro` (verify empirically — see 1b).
- Whether the vLLM server was launched with a `--reasoning-parser` (affects
  whether `reasoning_content` is populated vs. inline `<think>` tags).
- Final layer band and strength values for the steering demo (tune for the
  chosen `STEER_MODEL` until the effect is obvious but the text stays coherent).

## Stretch goals (only if time allows)
- A second steering concept toggled by a flag, to show it generalizes.
- A simple "prompt vs. steering" comparison: ask the DeepSeek model nicely to be
  cheerful (Part 1) vs. steer a local model to be cheerful (Part 2) — illustrates
  that steering changes internals, not instructions.