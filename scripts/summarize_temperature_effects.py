import pathlib, pandas as pd, json

BASE = pathlib.Path(__file__).resolve().parents[1]

RUNS = {
    "discrete": {
        "t0.0": BASE / "experiments" / "discrete" / "t0.0" / "runs.csv",
        "t0.2": BASE / "experiments" / "discrete" / "t0.2" / "runs.csv",
        "t0.8": BASE / "experiments" / "discrete" / "t0.8" / "runs.csv",
    },
    "continuous": {
        "t0.0": BASE / "experiments" / "continuous" / "t0.0" / "runs.csv",
        "t0.2": BASE / "experiments" / "continuous" / "t0.2" / "runs.csv",
        "t0.8": BASE / "experiments" / "continuous" / "t0.8" / "runs.csv",
    },
}


def load_df(path: pathlib.Path, expected_temp: float | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Some legacy CSVs may not contain 'temperature'; inject from folder if missing
    if "temperature" not in df.columns and expected_temp is not None:
        df["temperature"] = expected_temp
    return df


def short(temp_key: str) -> float:
    return float(temp_key.replace("t", ""))


def summarize_block(kind: str, files: dict[str, pathlib.Path]) -> dict:
    out = {}
    rows = []
    for tkey, p in files.items():
        tval = short(tkey)
        df = load_df(p, tval)
        # Verify recorded temperatures (if present)
        uniq = sorted(df["temperature"].dropna().unique()) if "temperature" in df.columns else []
        out[f"{tkey}_recorded_temps"] = [float(x) for x in uniq] if len(uniq) > 0 else ["(missing column)"]

        overall_mean = float(df["score"].mean())
        overall_std = float(df["score"].std())
        by_proxy = df.groupby("proxy")["score"].agg(["mean", "std"]).reset_index()
        out[f"{tkey}_overall_mean"] = overall_mean
        out[f"{tkey}_overall_std"] = overall_std
        out[f"{tkey}_by_proxy"] = by_proxy.to_dict(orient="records")

        rows.append({"temp": tval, "overall_mean": overall_mean})

    # Sort temps and compute deltas vs t0.0
    base = next((r for r in rows if abs(r["temp"] - 0.0) < 1e-9), None)
    if base:
        for r in rows:
            r["delta_vs_t0.0"] = r["overall_mean"] - base["overall_mean"]
    out["overall_means"] = rows
    return out


def main():
    results = {}
    for kind, files in RUNS.items():
        results[kind] = summarize_block(kind, files)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


