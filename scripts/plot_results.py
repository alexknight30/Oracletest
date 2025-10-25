import json, pathlib, pandas as pd, matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).resolve().parents[1]
RUNS_CSV = BASE / "data" / "runs.csv"
SCEN_JSON = BASE / "data" / "scenarios.json"
OUT_DIR = BASE / "data" / "figures"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RUNS_CSV)
    scen = {s["id"]: s for s in json.loads(SCEN_JSON.read_text("utf-8"))}

    agg = df.groupby(["scenario_id","proxy"], dropna=False)["score"].mean().reset_index()
    gold = pd.DataFrame([{"scenario_id": sid, "gold": scen[sid]["gold"]} for sid in scen])

    proxies = ["fact","coherence","pragmatic"]
    for sid in sorted(scen.keys()):
        g = gold.loc[gold["scenario_id"]==sid, "gold"].iloc[0]
        sub = agg[agg["scenario_id"]==sid].set_index("proxy").reindex(proxies)
        scores = sub["score"].tolist()

        labels = ["gold"] + proxies
        values = [g] + [s if s==s else 0 for s in scores]
        plt.figure()
        plt.bar(labels, values)
        plt.ylim(0,1)
        plt.title(f"{sid} - Aggregate scores")
        plt.ylabel("Score")
        plt.savefig(OUT_DIR / f"{sid}_aggregate.png", dpi=200, bbox_inches="tight")
        plt.close()

        misalign = [abs((s if s==s else 0) - g) for s in scores]
        plt.figure()
        plt.bar(proxies, misalign)
        plt.ylim(0,1)
        plt.title(f"{sid} - Misalignment (|proxy - gold|)")
        plt.ylabel("Absolute error")
        plt.savefig(OUT_DIR / f"{sid}_misalignment.png", dpi=200, bbox_inches="tight")
        plt.close()

    overall = agg.groupby("proxy")["score"].mean().reindex(proxies)
    plt.figure()
    plt.bar(["gold"]+list(overall.index), [gold["gold"].mean()]+overall.tolist())
    plt.ylim(0,1)
    plt.title("Overall - Aggregate scores")
    plt.ylabel("Score")
    plt.savefig(OUT_DIR / "overall_aggregate.png", dpi=200, bbox_inches="tight")
    plt.close()

    merged = agg.merge(gold, on="scenario_id", how="left")
    merged["abs_err"] = (merged["score"] - merged["gold"]).abs()
    overall_mis = merged.groupby("proxy")["abs_err"].mean().reindex(proxies)

    plt.figure()
    plt.bar(list(overall_mis.index), overall_mis.tolist())
    plt.ylim(0,1)
    plt.title("Overall - Misalignment rate")
    plt.ylabel("Mean |proxy - gold|")
    plt.savefig(OUT_DIR / "overall_misalignment.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved figures to: {OUT_DIR}")

if __name__ == "__main__":
    main()

