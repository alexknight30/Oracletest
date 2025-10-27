print("Step 1: Starting", flush=True)
import os, pathlib, time, pandas as pd, random, re, sys
print("Step 2: Basic imports done", flush=True)

from tenacity import retry, stop_after_attempt, wait_exponential_jitter
print("Step 3: tenacity imported", flush=True)

from dotenv import load_dotenv
print("Step 4: dotenv imported", flush=True)

from openai import OpenAI
print("Step 5: OpenAI imported", flush=True)

BASE = pathlib.Path(__file__).resolve().parent.parent
print(f"Step 6: BASE = {BASE}", flush=True)

PROMPTS_DIR = BASE / "promptscont"
DATA_DIR = BASE / "datacont"
print(f"Step 7: Directories set", flush=True)

load_dotenv(BASE / ".env")
print("Step 8: .env loaded", flush=True)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
N_RUNS = int(os.getenv("N_RUNS", "50"))
print(f"Step 9: MODEL={MODEL}, N_RUNS={N_RUNS}", flush=True)

client = OpenAI()
print("Step 10: OpenAI client created", flush=True)

print("All initialization steps completed successfully!", flush=True)

