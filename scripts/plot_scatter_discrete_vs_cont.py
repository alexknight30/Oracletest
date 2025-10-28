import pathlib, pandas as pd, matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).resolve().parents[1]
DISCRETE_CSV = BASE / "data" / "runs.csv"
CONT_CSV = BASE / "datacont" / "runs.csv"
OUT_DIR = BASE / "scatterplots" / "discretevcont"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_disc = pd.read_csv(DISCRETE_CSV)
    df_cont = pd.read_csv(CONT_CSV)

    proxies = ["coherence", "fact", "pragmatic"]
    scenario_ids = sorted(set(df_disc["scenario_id"]).union(set(df_cont["scenario_id"])))

    for sid in scenario_ids:
        for proxy in proxies:
            disc = df_disc[(df_disc["scenario_id"] == sid) & (df_disc["proxy"] == proxy)].copy()
            cont = df_cont[(df_cont["scenario_id"] == sid) & (df_cont["proxy"] == proxy)].copy()

            # Ensure numeric and ordered by run index for intuitive x-axis
            disc = disc.dropna(subset=["score", "run_idx"]).astype({"score": float, "run_idx": int}).sort_values("run_idx")
            cont = cont.dropna(subset=["score", "run_idx"]).astype({"score": float, "run_idx": int}).sort_values("run_idx")

            plt.figure(figsize=(8, 4.5))
            # x = run index (0..49), y = score
            plt.scatter(disc["run_idx"], disc["score"], color="red", alpha=0.8, label=f"Discrete (n={len(disc)})", s=24)
            plt.scatter(cont["run_idx"], cont["score"], color="blue", alpha=0.8, label=f"Continuous (n={len(cont)})", s=24)

            plt.xlim(-1, 50)
            plt.ylim(0, 1)
            plt.xlabel("Run index (0-49)")
            plt.ylabel("Score")
            plt.title(f"{sid} — {proxy} : Discrete (red) vs Continuous (blue)")
            plt.legend()
            plt.grid(True, axis="y", alpha=0.3)

            out_path = OUT_DIR / f"{sid}__{proxy}_scatter.png"
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()

    print(f"Saved scatterplots to: {OUT_DIR}")


if __name__ == "__main__":
    main()


