import os, pathlib, pandas as pd, matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).resolve().parents[1]

# Config via environment variables (required RUN_DIR)
RUN_DIR = os.getenv("RUN_DIR")
OUT_DIR = pathlib.Path(os.getenv("OUT_DIR") or (BASE / "scatterplots" / "single"))
COLOR = os.getenv("COLOR", "red")
LABEL = os.getenv("LABEL", "run")


def main():
    if not RUN_DIR:
        raise SystemExit("RUN_DIR environment variable is required.")
    run_path = pathlib.Path(RUN_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(run_path / "runs.csv")
    proxies = ["coherence", "fact", "pragmatic"]
    scenario_ids = sorted(df["scenario_id"].unique())

    for sid in scenario_ids:
        for proxy in proxies:
            sub = df[(df["scenario_id"] == sid) & (df["proxy"] == proxy)].copy()
            sub = sub.dropna(subset=["score", "run_idx"]).astype({"score": float, "run_idx": int}).sort_values("run_idx")

            plt.figure(figsize=(8, 4.5))
            plt.scatter(sub["run_idx"], sub["score"], color=COLOR, alpha=0.85, label=f"{LABEL} (n={len(sub)})", s=26)
            plt.xlim(-1, 50)
            plt.ylim(0, 1)
            plt.xlabel("Run index (0-49)")
            plt.ylabel("Score")
            plt.title(f"{sid} — {proxy} : {LABEL}")
            plt.legend()
            plt.grid(True, axis="y", alpha=0.3)

            out_path = OUT_DIR / f"{sid}__{proxy}_scatter.png"
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()

    print(f"Saved scatterplots to: {OUT_DIR}")


if __name__ == "__main__":
    main()


