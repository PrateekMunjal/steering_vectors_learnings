# Steering Vectors — Exercise Series

A hands-on series exploring how language models think and how we can steer them from the inside.

## Roadmap

| Date | Exercise | What We Do | Who |
|---|---|---|---|
| **13 Jun 2026** | [Today] Steering Vectors Basics | Chat with a model → see it think → reach inside and turn the dial | xxx |
| **20 Jun 2026** | Code Usage by Library | Trace how `openai`, `transformers`, `repeng`, and `torch` are actually used under the hood | xxx |
| **27 Jun 2026** | SAE Code & Clusters | Sparse Autoencoders: decompose model activations into interpretable features and cluster them | xxx |
| **11 Jul 2026** | Final Blogpost | Write it all up — from API call to steering to SAE clusters — as a polished post | xxx |

---

## Today's Exercise — 13 Jun 2026

We go from "AI just answers questions" to "we can steer its internal mood dial" in four steps.

### Setup

```bash
# Create a virtual environment
python -m venv steer && source steer/bin/activate

# Install everything
pip install -r requirements.txt
```

> **No GPU?** Steps 1–3 only need `openai` and a network connection. Step 4 works on CPU too — just slower.

### Step 1: It Just Answers

```bash
python part1_chat.py
```

You send question, you get answer back. That's it.

### Step 2: Base vs Instruct

```bash
python part1_base.py
```

Same prompt, two modes: the base model just continues the text, the instruct model actually answers. That difference is instruction tuning.

### Step 3: It Shows Its Work

```bash
python part1_thinking.py
```

Same question, but now the model thinks out loud before answering. The "thinking" is just more text it generates internally.

### Step 4: We Turn the Dial

```bash
python part2_steering.py
```

All that thinking is just numbers inside the model. We reach in and nudge them — same prompt, but the model becomes cheerful or gloomy. No prompt engineering. We're changing its internals directly.

Try your own values:
```bash
python part2_steering.py --strength 3.5    # any number
python part2_steering.py --concept formal   # different concept
```

### What's Running Where

| | Steps 1–3 | Step 4 |
|---|---|---|
| Model | DeepSeek (remote server) | Qwen 1.5B (your machine) |
| How | HTTP API | Local Python + repeng |

Step 4 never calls the remote server — steering needs access to the model's internals.

---

## Configuration

All defaults work out of the box. Override with environment variables if needed:

| Variable | Default | What |
|---|---|---|
| `VLLM_BASE_URL` | `http://worker-1:8000/v1` | Remote model server |
| `VLLM_MODEL` | `deepseek-ai/DeepSeek-V4-Pro` | Remote model name |
| `VLLM_API_KEY` | `EMPTY` | API key (vLLM accepts any string) |
| `STEER_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Local steering model |
| `STEER_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `STEER_CONCEPT` | `cheerful` | Default concept preset |

---

## Coming Up

- **20 Jun** — We'll trace each library call: what `openai` sends over the wire, how `transformers` loads a model, what `repeng` does when it trains a steering vector, and where `torch` actually runs the math.
- **27 Jun** — We'll build Sparse Autoencoders to decompose a model's activations into interpretable directions, then cluster those directions to find structure.
- **11 Jul** — Everything becomes a blogpost.
