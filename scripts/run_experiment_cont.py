import os, pathlib, time, pandas as pd, random, re, sys, datetime
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from dotenv import load_dotenv
from openai import OpenAI

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

BASE = pathlib.Path(__file__).resolve().parents[1]
PROMPTS_DIR = BASE / "promptscont"
DATA_DIR = BASE / "datacont"

# Allow overriding output directory (e.g., for separate control runs)
ALT_DATA_DIR = os.getenv("ALT_DATA_DIR")
if ALT_DATA_DIR:
    DATA_DIR = pathlib.Path(ALT_DATA_DIR) if os.path.isabs(ALT_DATA_DIR) else (BASE / ALT_DATA_DIR)

RUNS_CSV = DATA_DIR / "runs.csv"
PROGRESS_LOG = DATA_DIR / "progress.log"

# Load environment variables from .env file in project root
load_dotenv(BASE / ".env")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
N_RUNS = int(os.getenv("N_RUNS", "50"))
# Temperature override (default 0.0 as before)
TEMP = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
# Optional run label for provenance tagging
RUN_NAME = os.getenv("RUN_NAME", None)

client = OpenAI()

# Updated for continuous scale
number_pat = re.compile(r"^(0(\.\d+)?|1(\.0+)?)\s*$")

@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(1, 8))
def call_model(prompt_text: str) -> str:
    print(".", end="", flush=True)  # Show activity for each API call
    # Mirror progress to file as well
    try:
        with open(PROGRESS_LOG, "a", encoding="utf-8") as lf:
            lf.write(".")
    except Exception:
        pass
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Return only one number between 0 and 1. Output the number only. No words."},
            {"role": "user", "content": prompt_text},
        ],
        temperature=TEMP,
        max_tokens=10,
    )
    return resp.choices[0].message.content.strip()

def normalize_score(s: str) -> float | None:
    s = s.strip()
    # Try to parse as float between 0 and 1
    try:
        val = float(s)
        if 0 <= val <= 1:
            return val
    except ValueError:
        pass
    # Try to extract from text
    tokens = re.findall(r"(?:^|\s)(0(?:\.\d+)?|1(?:\.0+)?)(?:\s|$)", s)
    return float(tokens[0]) if tokens else None

def main():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    # Start progress log
    try:
        with open(PROGRESS_LOG, "a", encoding="utf-8") as lf:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lf.write(f"\n================ RUN START {ts} ================\n")
    except Exception:
        pass
    rows = []
    prompt_files = sorted(PROMPTS_DIR.glob("*.txt"))
    total_prompts = len(prompt_files)
    total_calls = total_prompts * N_RUNS
    
    header_lines = [
        "\n" + "="*70,
        "TRUTH PROXIES EXPERIMENT (CONTINUOUS) - STARTING",
        "="*70,
        f"Total prompts: {total_prompts}",
        f"Runs per prompt: {N_RUNS}",
        f"Total API calls: {total_calls}",
        f"Model: {MODEL}",
        "="*70 + "\n",
    ]
    for line in header_lines:
        print(line, flush=True)
    try:
        with open(PROGRESS_LOG, "a", encoding="utf-8") as lf:
            for line in header_lines:
                lf.write(line + ("\n" if not line.endswith("\n") else ""))
    except Exception:
        pass
    
    call_count = 0
    start_time = time.time()
    
    for prompt_idx, pf in enumerate(prompt_files, 1):
        scenario_id, proxy = pf.stem.split("__")
        prompt_text = pf.read_text(encoding="utf-8")
        
        line_a = f"\n[{prompt_idx}/{total_prompts}] {scenario_id} - {proxy.upper()}"
        print(line_a, flush=True)
        print(f"  Progress: ", end="", flush=True)
        try:
            with open(PROGRESS_LOG, "a", encoding="utf-8") as lf:
                lf.write(line_a + "\n  Progress: ")
        except Exception:
            pass
        
        for run_idx in range(N_RUNS):
            out = call_model(prompt_text)
            score = normalize_score(out)
            rows.append({
                "scenario_id": scenario_id,
                "proxy": proxy,
                "run_idx": run_idx,
                "raw": out,
                "score": score,
                "temperature": TEMP,
                **({"run_name": RUN_NAME} if RUN_NAME else {})
            })
            call_count += 1
            
            # Progress indicator every 10 calls
            if (run_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / call_count
                remaining = (total_calls - call_count) * avg_time
                tick = f" [{run_idx + 1}/{N_RUNS}]"
                print(tick, end="", flush=True)
                try:
                    with open(PROGRESS_LOG, "a", encoding="utf-8") as lf:
                        lf.write(tick)
                except Exception:
                    pass
            
            time.sleep(0.05 + random.random()*0.1)
        
        elapsed = time.time() - start_time
        end_line = f" DONE | Total calls: {call_count}/{total_calls} | Elapsed: {elapsed:.1f}s"
        print(end_line, flush=True)
        try:
            with open(PROGRESS_LOG, "a", encoding="utf-8") as lf:
                lf.write(end_line + "\n")
        except Exception:
            pass
        
    total_time = time.time() - start_time
    print(f"\n{'='*70}", flush=True)
    print(f"ALL API CALLS COMPLETE!", flush=True)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)", flush=True)
    print(f"Saving results to CSV...", flush=True)
    try:
        with open(PROGRESS_LOG, "a", encoding="utf-8") as lf:
            lf.write("\n" + "="*70 + "\n")
            lf.write("ALL API CALLS COMPLETE!\n")
            lf.write(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")
            lf.write("Saving results to CSV...\n")
    except Exception:
        pass
    
    df = pd.DataFrame(rows)
    if RUNS_CSV.exists():
        old = pd.read_csv(RUNS_CSV)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(RUNS_CSV, index=False)
    print(f"SUCCESS - Wrote {len(df)} rows to {RUNS_CSV}", flush=True)
    print(f"{'='*70}\n", flush=True)
    try:
        with open(PROGRESS_LOG, "a", encoding="utf-8") as lf:
            lf.write(f"SUCCESS - Wrote {len(df)} rows to {RUNS_CSV}\n")
            lf.write("="*70 + "\n")
    except Exception:
        pass

if __name__ == "__main__":
    main()
