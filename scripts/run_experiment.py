import os, pathlib, time, pandas as pd, random, re, sys
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from dotenv import load_dotenv
from openai import OpenAI

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

BASE = pathlib.Path(__file__).resolve().parents[1]
PROMPTS_DIR = BASE / "prompts"
DATA_DIR = BASE / "data"
RUNS_CSV = DATA_DIR / "runs.csv"

# Load environment variables from .env file in project root
load_dotenv(BASE / ".env")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
N_RUNS = int(os.getenv("N_RUNS", "50"))

client = OpenAI()

number_pat = re.compile(r"^(0(\.0)?|0\.5|1(\.0)?)\s*$")

@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(1, 8))
def call_model(prompt_text: str) -> str:
    print(".", end="", flush=True)  # Show activity for each API call
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def normalize_score(s: str) -> float | None:
    s = s.strip()
    if number_pat.match(s):
        return float(s)
    tokens = re.findall(r"(?:^|\s)(0(?:\.0)?|0\.5|1(?:\.0)?)(?:\s|$)", s)
    return float(tokens[0]) if tokens else None

def main():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    rows = []
    prompt_files = sorted(PROMPTS_DIR.glob("*.txt"))
    total_prompts = len(prompt_files)
    total_calls = total_prompts * N_RUNS
    
    print(f"\n{'='*70}", flush=True)
    print(f"TRUTH PROXIES EXPERIMENT - STARTING", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Total prompts: {total_prompts}", flush=True)
    print(f"Runs per prompt: {N_RUNS}", flush=True)
    print(f"Total API calls: {total_calls}", flush=True)
    print(f"Model: {MODEL}", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    call_count = 0
    start_time = time.time()
    
    for prompt_idx, pf in enumerate(prompt_files, 1):
        scenario_id, proxy = pf.stem.split("__")
        prompt_text = pf.read_text(encoding="utf-8")
        
        print(f"\n[{prompt_idx}/{total_prompts}] {scenario_id} - {proxy.upper()}", flush=True)
        print(f"  Progress: ", end="", flush=True)
        
        for run_idx in range(N_RUNS):
            out = call_model(prompt_text)
            score = normalize_score(out)
            rows.append({
                "scenario_id": scenario_id,
                "proxy": proxy,
                "run_idx": run_idx,
                "raw": out,
                "score": score
            })
            call_count += 1
            
            # Progress indicator every 10 calls
            if (run_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / call_count
                remaining = (total_calls - call_count) * avg_time
                print(f" [{run_idx + 1}/{N_RUNS}]", end="", flush=True)
            
            time.sleep(0.05 + random.random()*0.1)
        
        elapsed = time.time() - start_time
        print(f" DONE | Total calls: {call_count}/{total_calls} | Elapsed: {elapsed:.1f}s", flush=True)
        
    total_time = time.time() - start_time
    print(f"\n{'='*70}", flush=True)
    print(f"ALL API CALLS COMPLETE!", flush=True)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)", flush=True)
    print(f"Saving results to CSV...", flush=True)
    
    df = pd.DataFrame(rows)
    if RUNS_CSV.exists():
        old = pd.read_csv(RUNS_CSV)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(RUNS_CSV, index=False)
    print(f"SUCCESS - Wrote {len(df)} rows to {RUNS_CSV}", flush=True)
    print(f"{'='*70}\n", flush=True)

if __name__ == "__main__":
    main()

