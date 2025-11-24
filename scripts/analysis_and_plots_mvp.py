import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_summary(results_dir):
    path = os.path.join(results_dir, "summary.json")
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultsdir", required=True)
    parser.add_argument("--out", default="reports/experiment_report.md")
    args = parser.parse_args()

    df = load_summary(args.resultsdir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if df.empty:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# Experiment Report\n\nNo results found.\n")
        print("No results found; empty report written.")
        return

    plt.figure(figsize=(8, 5))
    sns.boxplot(x="T", y="kappa_mean", data=df)
    plt.title("Mean curvature (κ) by recursion depth T")
    plt.xlabel("Recursion cycles (T)")
    plt.ylabel("Mean curvature κ̄")
    fig_path = os.path.join(os.path.dirname(args.out), "kappa_by_T.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    corr = df[["kappa_mean", "echo", "energy"]].corr()

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# NOESIS MVP Experiment Report\n\n")
        f.write("## 1. Configuration Snapshot\n\n")
        f.write(f"- Results directory: `{args.resultsdir}`\n")
        f.write(f"- Number of runs: {len(df)}\n")
        f.write("\n## 2. Summary Statistics\n\n")
        f.write(df.describe().to_markdown())
        f.write("\n\n## 3. Correlations (κ̄, echo, energy)\n\n")
        f.write(corr.to_markdown())
        f.write("\n\n## 4. Curvature vs. Recursion Depth\n\n")
        f.write("![Mean curvature by T](kappa_by_T.png)\n")

    print("Report written to", args.out)
    print("Figure written to", fig_path)


if __name__ == "__main__":
    main()
