import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results_root", type=str, default="lead_optimization")
parser.add_argument("--reference_table", type=str, default="references/reference_pmo.csv")
parser.add_argument("--exclude_prescreen", action="store_true", help="Exclude GenMol and f-RAG from the table")
parser.add_argument("--include_std", action="store_true", help="Include standard deviations in the table")
args = parser.parse_args()

# Collect stats for "InVirtuoGen"
rows = []
all_runs_data = {}
for task in sorted(os.listdir(args.results_root)):
    task_dir = os.path.join(args.results_root, task)
    if not os.path.isdir(task_dir):
        continue
    csv_path = None
    for file in os.listdir(task_dir):
        if file.endswith(".csv") and file.startswith("results_"):
            csv_path = os.path.join(task_dir, file)
            break
    if not csv_path or not os.path.isfile(csv_path):
        continue
    df = pd.read_csv(csv_path)
    all_runs_data[task] = df["auc_top10"].values
    mean10 = df["auc_top10"].mean() if args.include_std else df["auc_top10"][0]
    std10 = df["auc_top10"].std(ddof=0) if args.include_std else 0
    rows.append((task, mean10, std10))

summary = pd.DataFrame(rows, columns=["Oracle", "mean10", "std10"])
summary["Oracle"] = summary["Oracle"].str.replace("_", " ")

# Sum statistics
task_means = {task: np.mean(values) for task, values in all_runs_data.items()}
n_runs = max((len(v) for v in all_runs_data.values()), default=3)

sum_per_run = []
for run_idx in range(n_runs):
    run_sum = 0
    for task, values in all_runs_data.items():
        run_sum += values[run_idx] if run_idx < len(values) else task_means[task]
    sum_per_run.append(run_sum)
    if not args.include_std:
        break

invirtuo_sum_mean = float(np.mean(sum_per_run))
invirtuo_sum_std = float(np.std(sum_per_run, ddof=0))

# Reference
reference = pd.read_csv(args.reference_table)
reference["Oracle"] = reference["Oracle"].str.replace("_", " ")

if args.exclude_prescreen:
    ref_methods = ["Genetic GFN", "Mol GA", "REINVENT", "Graph GA"]
else:
    ref_methods = ["GenMol", "f-RAG"]

# Create missing std columns as NaN
for method in ref_methods:
    std_col = f"{method}_std"
    reference[std_col] = np.nan

# Create missing sum std columns as NaN
for method in ref_methods:
    sum_std_col = f"{method}_sum_std"
    if sum_std_col not in reference.columns:
        reference[sum_std_col] = np.nan

merged = pd.merge(summary, reference, on="Oracle", how="left")
if args.exclude_prescreen:
    methods = ["mean10", "Genetic GFN", "Mol GA", "REINVENT", "Graph GA"]
    std_methods = ["std10", "Genetic GFN_std", "Mol GA_std", "REINVENT_std", "Graph GA_std"]
    sum_std_methods = ["invirtuo_sum_std", "Genetic GFN_sum_std", "Mol GA_sum_std", "REINVENT_sum_std", "Graph GA_sum_std"]
    latex_headers = ["InVirtuoGen (no prescreen)", "Gen. GFN", "Mol GA", "REINVENT", "Graph GA"]
else:
    methods = ["mean10", "GenMol", "f-RAG"]
    std_methods = ["std10", "GenMol_std", "f-RAG_std"]
    sum_std_methods = ["invirtuo_sum_std", "GenMol_sum_std", "f-RAG_sum_std"]
    latex_headers = ["InVirtuoGen", "GenMol", "f-RAG"]

latex = []
latex.append(r"\begin{table}[ht]")
latex.append(r"\centering")
if args.exclude_prescreen:
    caption = r"\caption{The results of the best performing models on the PMO benchmark, where we quote the AUC-top10 averaged over 3 runs"
    if args.include_std:
        caption += r" with standard deviations"
    caption += r". The best results are highlighted in bold. Values within one standard deviation of the best are also marked in bold. The results for Genetic GFN \citep{kim2024geneticguidedgflownetssampleefficient} and Mol GA \citep{tripp2023geneticalgorithmsstrongbaselines} are taken from the respective papers. The other results are taken from the original PMO benchmark paper by \citep{gao2022sampleefficiencymattersbenchmark}.}"
else:
    caption = r"\caption{Comparison of models on the PMO benchmark that screen ZINC250k before initialization. We report the AUC-top10 scores, averaged over three runs"
    if args.include_std:
        caption += r" with standard deviations"
    caption += r". Best results and those within one standard deviation of the best are indicated in bold. The scores for $f$-RAG \citep{lee2024moleculegenerationfragmentretrieval} and GenMol \cite{genmol} are taken from the respective publications.}"
latex.append(caption)
latex.append(r"\label{tab:our_vs_baselines} " if not args.exclude_prescreen else r"\label{tab:prescreened}")
latex.append(r"\begin{tabularx}{\linewidth}{l|p{2.2cm} "+("Y " * (len(methods)-1))+"}")
latex.append(r"\toprule")
latex.append("Oracle & " + " & ".join(latex_headers) + r" \\")
latex.append(r"\midrule")

for _, row in merged.iterrows():
    values, stds = [], []
    for i, m in enumerate(methods):
        v = row[m] if pd.notna(row[m]) else 0.0
        values.append(float(v))
        if args.include_std and i < len(std_methods):
            sc = std_methods[i]
            s = row[sc] if (sc in row and pd.notna(row[sc])) else None
            stds.append(float(s) if s is not None else None)
        else:
            stds.append(None)

    max_val = max(values)
    max_idx = values.index(max_val)
    max_std = stds[max_idx]  # may be None

    row_fmt = []
    for i, (v, s) in enumerate(zip(values, stds)):
        def fmt(x): return f"{x:.3f}"
        is_within = args.include_std and (max_std is not None) and (v >= max_val - max_std) and (i != max_idx)
        if i == max_idx:
            row_fmt.append("$\mathbf{"+ fmt(v) + "}$" + " {\\tiny (" + "$\\pm$ " + f"{fmt(s)}"  + ")}" if args.include_std and s is not None else f"$\\mathbf{{{fmt(v)}}}$")
        elif is_within:
            row_fmt.append("$\mathbf{"+ fmt(v) + "}$" + " {\\tiny (" + "$\\pm$ " + f"{fmt(s)}"  + ")}" if args.include_std and s is not None else f"$\\mathbf{{{fmt(v)}}}$")
        else:
            row_fmt.append(f"${fmt(v)}$" + " {\\tiny (" + "$\\pm$ " + f"{fmt(s)}"  + ")}" if args.include_std and s is not None else f"{fmt(v)}")
    latex.append(r"\small{" + f"{row['Oracle']}" + "} & " + " & ".join(row_fmt) + r" \\")

latex.append(r"\midrule")

# Sums
sums = [invirtuo_sum_mean]
sum_stds = [invirtuo_sum_std if args.include_std else None]

for i in range(1, len(methods)):
    m = methods[i]
    sums.append(float(merged[m].sum()))
    if args.include_std:
        ssc = sum_std_methods[i]
        if ssc in reference.columns and not reference[ssc].isna().all():
            sum_stds.append(float(reference[ssc].iloc[0]))
        else:
            sum_stds.append(None)
    else:
        sum_stds.append(None)

max_sum = max(sums)
max_sum_idx = sums.index(max_sum)
max_sum_std = sum_stds[max_sum_idx]

sum_fmt = []
for i, (s, sd) in enumerate(zip(sums, sum_stds)):
    def fmt(x): return f"{x:.3f}"
    if i == max_sum_idx:
        sum_fmt.append("$\mathbf{"+ fmt(s) + "}$" + " {\\tiny (" + "$\\pm$ " + f"{fmt(sd)}"  + ")}" if args.include_std and sd is not None else f"$\\mathbf{{{fmt(s)}}}$")
    else:
        within = args.include_std and (max_sum_std is not None) and (s >= max_sum - max_sum_std) and (s != max_sum)
        if within:
            sum_fmt.append("$\mathbf{"+ fmt(s) + "}$" + " {\\tiny (" + "$\\pm$ " + f"{fmt(sd)}"  + ")}" if args.include_std and sd is not None else f"$\\mathbf{{{fmt(s)}}}$")
        else:
            sum_fmt.append(f"${{{fmt(s)}}}$" + " {\\tiny (" + "$\\pm$ " + f"{fmt(sd)}"  + ")}" if args.include_std and sd is not None else f"{fmt(s)}")

latex.append(f"\\textbf{{Sum}} & " + " & ".join(sum_fmt) + r" \\")
latex.append(r"\bottomrule")
latex.append(r"\end{tabularx}")
latex.append(r"\end{table}")

print("\n".join(latex))

output_file = "pmo_comparison_table"
if args.include_std:
    output_file += "_with_std"
if args.exclude_prescreen:
    output_file += "_no_prescreen"
output_file += ".tex"

with open(output_file, "w") as f:
    f.write("\n".join(latex))
print(f"\nTable saved to {output_file}")