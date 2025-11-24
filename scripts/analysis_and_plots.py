    import argparse, os, json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns


    def load_summary(results_dir: str):
        summary_path = os.path.join(results_dir, "summary.json")
        if not os.path.exists(summary_path):
            return pd.DataFrame()
        data = json.load(open(summary_path))
        return pd.DataFrame(data)


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--resultsdir", required=True)
        parser.add_argument("--out", default="reports/experiment_report.md")
        args = parser.parse_args()

        df = load_summary(args.resultsdir)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

        if df.empty:
            with open(args.out, "w") as f:
                f.write("# Experiment Report

No summary.json found.
")
            print("no summary.json, wrote empty report")
            return

        plt.figure(figsize=(8, 5))
        sns.boxplot(x="T", y="kappa_mean", data=df)
        plt.title("Mean curvature κ̄ by recursion depth T")
        plt.savefig("reports/kappa_box.png", bbox_inches="tight")
        plt.close()

        corr = df[["kappa_mean", "echo", "energy"]].corr()

        with open(args.out, "w") as f:
            f.write("# NOESIS MVP Experiment Report

")
            f.write("## Summary statistics

")
            f.write(df.describe().to_markdown())
            f.write("

## Correlation matrix (κ̄, echo, energy)

")
            f.write(corr.to_markdown())
            f.write("

![kappa_box](kappa_box.png)
")

        print("report written to", args.out)


    if __name__ == "__main__":
        main()
