import os, pathlib, pandas as pd, matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).resolve().parents[1]

# Config via environment variables
RUN_A_DIR = os.getenv("RUN_A_DIR")  # required
RUN_B_DIR = os.getenv("RUN_B_DIR")  # required
A_LABEL = os.getenv("A_LABEL", "A")
B_LABEL = os.getenv("B_LABEL", "B")
A_COLOR = os.getenv("A_COLOR", "red")
B_COLOR = os.getenv("B_COLOR", "blue")
OUT_DIR = pathlib.Path(os.getenv("OUT_DIR") or (BASE / "scatterplots" / "comparison"))


def main():
    if not RUN_A_DIR or not RUN_B_DIR:
        raise SystemExit("RUN_A_DIR and RUN_B_DIR environment variables are required.")
    run_a = pathlib.Path(RUN_A_DIR)
    run_b = pathlib.Path(RUN_B_DIR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_a = pd.read_csv(run_a / "runs.csv")
    df_b = pd.read_csv(run_b / "runs.csv")

    proxies = ["coherence", "fact", "pragmatic"]
    scenario_ids = sorted(set(df_a["scenario_id"]).union(set(df_b["scenario_id"])))

    for sid in scenario_ids:
        for proxy in proxies:
            a = df_a[(df_a["scenario_id"] == sid) & (df_a["proxy"] == proxy)].copy()
            b = df_b[(df_b["scenario_id"] == sid) & (df_b["proxy"] == proxy)].copy()

            a = a.dropna(subset=["score", "run_idx"]).astype({"score": float, "run_idx": int}).sort_values("run_idx")
            b = b.dropna(subset=["score", "run_idx"]).astype({"score": float, "run_idx": int}).sort_values("run_idx")

            plt.figure(figsize=(8, 4.5))
            plt.scatter(a["run_idx"], a["score"], color=A_COLOR, alpha=0.8, label=f"{A_LABEL} (n={len(a)})", s=24)
            plt.scatter(b["run_idx"], b["score"], color=B_COLOR, alpha=0.8, label=f"{B_LABEL} (n={len(b)})", s=24)

            plt.xlim(-1, 50)
            plt.ylim(0, 1)
            plt.xlabel("Run index (0-49)")
            plt.ylabel("Score")
            plt.title(f"{sid} — {proxy} : {A_LABEL} vs {B_LABEL}")
            plt.legend()
            plt.grid(True, axis="y", alpha=0.3)

            out_path = OUT_DIR / f"{sid}__{proxy}_scatter.png"
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()

    print(f"Saved scatterplots to: {OUT_DIR}")


if __name__ == "__main__":
    main()


