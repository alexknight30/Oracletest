import os, pathlib, pandas as pd, matplotlib.pyplot as plt, numpy as np

BASE = pathlib.Path(__file__).resolve().parents[1]
DISCRETE_CSV = BASE / "data" / "runs.csv"
CONT_CSV = BASE / "datacont" / "runs.csv"
OUT_DIR = BASE / "histograms" / "discretevcont"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_disc = pd.read_csv(DISCRETE_CSV)
    df_cont = pd.read_csv(CONT_CSV)

    # Expected proxies and scenarios inferred from CSVs
    proxies = ["coherence", "fact", "pragmatic"]
    scenario_ids = sorted(set(df_disc["scenario_id"]).union(set(df_cont["scenario_id"])))

    # Consistent bins from 0 to 1
    bins = np.linspace(0, 1, 21)

    for sid in scenario_ids:
        for proxy in proxies:
            disc_scores = (
                df_disc[(df_disc["scenario_id"] == sid) & (df_disc["proxy"] == proxy)]["score"]
                .dropna()
                .astype(float)
            )
            cont_scores = (
                df_cont[(df_cont["scenario_id"] == sid) & (df_cont["proxy"] == proxy)]["score"]
                .dropna()
                .astype(float)
            )

            plt.figure(figsize=(7, 4))
            # Overlay histograms with transparency
            plt.hist(
                disc_scores,
                bins=bins,
                color="red",
                alpha=0.5,
                label=f"Discrete (n={len(disc_scores)})",
                edgecolor="black",
            )
            plt.hist(
                cont_scores,
                bins=bins,
                color="blue",
                alpha=0.5,
                label=f"Continuous (n={len(cont_scores)})",
                edgecolor="black",
            )

            plt.xlim(0, 1)
            plt.ylim(bottom=0)
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.title(f"{sid} — {proxy} : Discrete (red) vs Continuous (blue)")
            plt.legend()

            out_path = OUT_DIR / f"{sid}__{proxy}_hist.png"
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()

    print(f"Saved histograms to: {OUT_DIR}")


if __name__ == "__main__":
    main()


