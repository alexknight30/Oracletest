import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "peopledata" / "people_responses.csv"
FIG_DIR = BASE / "peopledata" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Gold standards (0, 0.5, 1)
GOLD = {
    "S01": 0.0,
    "S02": 0.0,
    "S03": 0.5,
    "S04": 0.0,
    "S05": 0.0,
    "S06": 1.0,
    "S07": 1.0,
    "S08": 0.0,
    "S09": 1.0,
    "S10": 0.5,
}

SCEN_ORDER = [f"S{str(i).zfill(2)}" for i in range(1, 11)]


def plot_person(df_person: pd.DataFrame, person_id: str):
    gold_vals = [GOLD[sid] for sid in SCEN_ORDER]
    resp_map = dict(zip(df_person["scenario_id"], df_person["response_norm"]))
    resp_vals = [resp_map.get(sid, float("nan")) for sid in SCEN_ORDER]

    x = range(len(SCEN_ORDER))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar([i - width/2 for i in x], gold_vals, width=width, label="Gold", color="#333")
    plt.bar([i + width/2 for i in x], resp_vals, width=width, label=person_id, color="#888")
    plt.xticks(x, SCEN_ORDER)
    plt.ylim(0, 1.05)
    plt.ylabel("Score (0-1)")
    plt.title(f"{person_id}: Responses vs Gold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{person_id}_bars.png", dpi=200)
    plt.close()


def plot_average(df: pd.DataFrame):
    gold_vals = [GOLD[sid] for sid in SCEN_ORDER]
    avg = df.groupby("scenario_id")["response_norm"].mean()
    avg_vals = [avg.get(sid, float("nan")) for sid in SCEN_ORDER]

    x = range(len(SCEN_ORDER))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar([i - width/2 for i in x], gold_vals, width=width, label="Gold", color="#333")
    plt.bar([i + width/2 for i in x], avg_vals, width=width, label="People avg", color="#4a90e2")
    plt.xticks(x, SCEN_ORDER)
    plt.ylim(0, 1.05)
    plt.ylabel("Score (0-1)")
    plt.title("Average of People Responses vs Gold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "people_avg_bars.png", dpi=200)
    plt.close()


def plot_overall(df: pd.DataFrame):
    people_overall = df["response_norm"].mean()
    gold_overall = pd.Series(GOLD).mean()
    plt.figure(figsize=(4, 5))
    plt.bar(["Gold", "People avg"], [gold_overall, people_overall], color=["#333", "#4a90e2"])
    plt.ylim(0, 1.05)
    plt.ylabel("Score (0-1)")
    plt.title("Overall Average vs Gold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "overall_people_vs_gold.png", dpi=200)
    plt.close()


def scatter_person(df_person: pd.DataFrame, person_id: str):
    gold_vals = [GOLD[sid] for sid in SCEN_ORDER]
    resp_map = dict(zip(df_person["scenario_id"], df_person["response_norm"]))
    resp_vals = [resp_map.get(sid, float("nan")) for sid in SCEN_ORDER]

    plt.figure(figsize=(5, 5))
    plt.scatter(gold_vals, resp_vals, color="#888")
    plt.plot([0, 1], [0, 1], color="#333", linestyle="--", linewidth=1)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Gold")
    plt.ylabel("Person")
    plt.title(f"{person_id}: Gold vs Person Responses")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{person_id}_scatter.png", dpi=200)
    plt.close()


def scatter_average(df: pd.DataFrame):
    gold_vals = [GOLD[sid] for sid in SCEN_ORDER]
    avg = df.groupby("scenario_id")["response_norm"].mean()
    avg_vals = [avg.get(sid, float("nan")) for sid in SCEN_ORDER]

    plt.figure(figsize=(5, 5))
    plt.scatter(gold_vals, avg_vals, color="#4a90e2")
    plt.plot([0, 1], [0, 1], color="#333", linestyle="--", linewidth=1)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Gold")
    plt.ylabel("People avg")
    plt.title("Gold vs People Average (per scenario)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "people_avg_scatter.png", dpi=200)
    plt.close()


def main():
    df = pd.read_csv(DATA)
    for pid, sub in df.groupby("person_id"):
        plot_person(sub, pid)
        scatter_person(sub, pid)
    plot_average(df)
    plot_overall(df)
    scatter_average(df)
    print(f"Wrote figures to {FIG_DIR}")

if __name__ == "__main__":
    main()