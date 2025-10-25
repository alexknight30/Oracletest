# Truth Proxies Alignment Experiment

This project runs a controlled alignment experiment to test how different proxy goals steer an LLM when the overarching (hidden) aim is "maximize truth."

## What We're Testing

Three proxy instructions are tested across 10 structured-ambiguity scenarios:
1. **Maximize fact**
2. **Maximize epistemic coherence**
3. **Maximize pragmatic truthfulness**

Each scenario has a gold truth score (0, 0.5, or 1) that serves as the objective yardstick for measuring alignment vs. misalignment.

## Project Structure

```
truth-proxies-exp/
├─ prompts/                  # 30 prompt .txt files (10 scenarios × 3 proxies)
├─ data/
│  ├─ runs.csv               # raw model outputs from all runs
│  ├─ scenarios.json         # 10 scenarios with gold standards
│  └─ figures/               # 22 PNG charts (created by plot_results.py)
├─ scripts/
│  ├─ generate_prompts.py    # builds the 30 prompt files
│  ├─ run_experiment.py      # calls the API 50x per prompt, logs CSV
│  └─ plot_results.py        # makes all 22 graphs
├─ .env                      # OPENAI_API_KEY configuration
├─ requirements.txt
└─ README.md
```

## Setup Instructions

### 1. Install Dependencies

Create a virtual environment and install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key

Edit the `.env` file and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-key-here
```

Optionally, you can customize the model or number of runs:

```
OPENAI_MODEL=gpt-4o
N_RUNS=50
```

## Running the Experiment

### Step 1: Generate the 30 Prompts

This creates 30 prompt files (10 scenarios × 3 proxies):

```powershell
python scripts/generate_prompts.py
```

You should see: `Done. Wrote 30 prompt files to ...\prompts`

### Step 2: Run the Experiment

This calls the OpenAI API 50 times per prompt (1,500 total calls) and logs all results to `data/runs.csv`:

```powershell
python scripts/run_experiment.py
```

Note: This will take some time (approximately 15-30 minutes depending on API rate limits). The script includes automatic retry logic and pacing to handle rate limits gracefully.

### Step 3: Generate Visualizations

This creates 22 PNG charts showing aggregate scores and misalignment:

```powershell
python scripts/plot_results.py
```

Charts are saved to `data/figures/`:
- 10 aggregate score charts (one per scenario)
- 10 misalignment charts (one per scenario)
- 1 overall aggregate chart
- 1 overall misalignment chart

## Understanding the Results

### Metrics

- **Aggregate score**: Mean of 50 runs per (scenario × proxy)
- **Misalignment**: |aggregate - gold| for each proxy
- **Overall metrics**: Averaged across all scenarios

### What This Tells Us

- Which proxy best approximates truth (higher aggregate, lower misalignment)
- Where misalignment tends to arise:
  - *Fact* may fail on surface-vs-inference conflicts
  - *Coherence* may rationalize contradictions
  - *Pragmatic truthfulness* may trade candor for politeness or over-hedge
- **Propensity, not just capability**: measures how often each proxy drifts from the gold standard under repeated sampling

## Scenarios Included

The experiment includes 10 carefully designed scenarios with structured ambiguity:
- S01: Desert explorer and historical lake
- S02: Philosopher's dream lecture
- S03: Silent apology
- S04: Lighthouse reflection
- S05: Forgotten door lock
- S06: Valley echo
- S07: Frost damage revision
- S08: Candle count
- S09: Train arrival timing
- S10: Restored painting

Each scenario has a recoverable, defensible truth despite narrative ambiguity.

## Troubleshooting

### Rate Limits
If you encounter rate limit errors, the script includes automatic retry logic with exponential backoff. You can also:
- Reduce `N_RUNS` in your `.env` file
- Add longer sleep intervals in `run_experiment.py`

### API Key Issues
Make sure your `.env` file is in the project root and contains a valid OpenAI API key.

### Missing Dependencies
If you get import errors, ensure you've activated your virtual environment and installed all requirements:

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Notes

- The experiment uses Chat Completions API with temperature=0.2 for consistency
- Results are appended to `runs.csv` if it exists (allows resuming interrupted runs)
- All prompt files follow a strict template for consistency across conditions

