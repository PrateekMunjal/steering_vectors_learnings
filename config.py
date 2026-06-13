"""
Settings file.
If you want to change anything, you can set environment variables
or just edit the values below.
"""

import os

# The address of the AI server we talk to in Part 1
VLLM_URL   = os.getenv("VLLM_BASE_URL", "http://worker-1:8000/v1")

# Which AI model to use on that server
VLLM_MODEL = os.getenv("VLLM_MODEL", "deepseek-ai/DeepSeek-V4-Pro")

# A password to access the server (usually not needed, so it's "EMPTY")
VLLM_KEY   = os.getenv("VLLM_API_KEY", "EMPTY")

# Which smaller AI model to load locally for Part 2
STEER_MODEL = os.getenv("STEER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# Where to run the Part 2 model: "cuda" = GPU, "cpu" = CPU, "auto" = pick for me
STEER_DEVICE = os.getenv("STEER_DEVICE", "auto")
