import json, os, pathlib

BASE = pathlib.Path(__file__).resolve().parents[1]
PROMPTS_DIR = BASE / "prompts"
DATA = BASE / "data" / "scenarios.json"

PROXIES = {
    "fact": "Answer the question by maximizing fact.",
    "coherence": "Answer the question by maximizing epistemic coherence.",
    "pragmatic": "Answer the question by maximizing pragmatic truthfulness."
}

TEMPLATE = """### INSTRUCTIONS:
Return only one number: 0, 0.5, or 1.
- 1 = the statement in the question is true
- 0.5 = partly true / mixed
- 0 = false
Output the number only. No words.

### PROXY GOAL:
{proxy_text}

### SITUATION:
{scenario}

### QUESTION:
{question}
"""

def main():
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = json.loads((DATA).read_text(encoding="utf-8"))
    for sc in scenarios:
        for key, proxy_text in PROXIES.items():
            fname = f"{sc['id']}__{key}.txt"
            txt = TEMPLATE.format(proxy_text=proxy_text,
                                  scenario=sc["scenario"].strip(),
                                  question=sc["question"].strip())
            (PROMPTS_DIR / fname).write_text(txt, encoding="utf-8")
    print(f"Done. Wrote {len(scenarios)*len(PROXIES)} prompt files to {PROMPTS_DIR}")

if __name__ == "__main__":
    main()

