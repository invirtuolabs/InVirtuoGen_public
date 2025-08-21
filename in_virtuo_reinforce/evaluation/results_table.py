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
all_runs_data = {}  # Store individual run data for sum calculation
for task in sorted(os.listdir(args.results_root)):
    task_dir = os.path.join(args.results_root, task)

    if not os.path.isdir(task_dir):
        print("not a directory")
        continue
    for file in os.listdir(task_dir):
        if file.endswith(".csv"):
            if not file.startswith("results_"):
                continue
            csv_path = os.path.join(task_dir, file)

    if not os.path.isfile(csv_path):
        print("not a file")
        continue
    df = pd.read_csv(csv_path)

    # Store individual run values for this task
    all_runs_data[task] = df["auc_top10"].values

    mean10 = df["auc_top10"].mean()
    std10 = df["auc_top10"].std(ddof=0)
    rows.append((task, mean10, std10))

summary = pd.DataFrame(rows, columns=["Oracle", "mean10", "std10"])
summary["Oracle"] = summary["Oracle"].str.replace("_", " ")

# Calculate sum statistics properly
# For each run, calculate the sum across all tasks, then compute mean and std of those sums
n_runs = max(len(v) for v in all_runs_data.values()) if all_runs_data else 3
sum_per_run = []
for run_idx in range(n_runs):
    run_sum = 0
    for task, values in all_runs_data.items():

        if run_idx < len(values):
            run_sum += values[run_idx]

    sum_per_run.append(run_sum)
    if not args.include_std:
        break
invirtuo_sum_mean = np.mean(sum_per_run)
invirtuo_sum_std = np.std(sum_per_run, ddof=0)

# Load reference table
reference = pd.read_csv(args.reference_table)
reference["Oracle"] = reference["Oracle"].str.replace("_", " ")

# Add dummy std columns for reference methods if not present
if args.exclude_prescreen:
    ref_methods = ["Genetic GFN", "Mol GA", "REINVENT", "Graph GA"]
else:
    ref_methods = ["GenMol", "f-RAG"]

for method in ref_methods:
    std_col = f"{method}_std"
    if std_col not in reference.columns:
        reference[std_col] = 0.0  # Assume 0 std if not provided

# Also add sum std columns if not present
for method in ref_methods:
    sum_std_col = f"{method}_sum_std"
    if sum_std_col not in reference.columns:
        reference[sum_std_col] = 0.0  # Assume 0 std if not provided

# Merge and prepare full table
merged = pd.merge(summary, reference, on="Oracle", how="left")

# Define all methods
if args.exclude_prescreen:
    methods = ["mean10", "Genetic GFN", "Mol GA", "REINVENT", "Graph GA"]
    std_methods = ["std10", "Genetic GFN_std", "Mol GA_std", "REINVENT_std", "Graph GA_std"]
    sum_std_methods = ["invirtuo_sum_std", "Genetic GFN_sum_std", "Mol GA_sum_std", "REINVENT_sum_std", "Graph GA_sum_std"]
    latex_headers = ["InVirtuoGen", "Gen. GFN", "Mol GA", "REINVENT", "Graph GA"]
else:
    methods = ["mean10", "GenMol", "f-RAG"]
    std_methods = ["std10", "GenMol_std", "f-RAG_std"]
    sum_std_methods = ["invirtuo_sum_std", "GenMol_sum_std", "f-RAG_sum_std"]
    latex_headers = ["InVirtuoGen", "GenMol", "f-RAG"]

# Generate LaTeX table
latex = []
latex.append(r"\begin{table}[ht]")
latex.append(r"\centering")
if args.exclude_prescreen:
    caption = r"\caption{The results of the best performing models on the PMO benchmark, where we quote the AUC-top10 averaged over 3 runs"
    if args.include_std:
        caption += r" with standard deviations"
    caption += r". The best results are highlighted in bold. Values within one standard deviation of the best are also marked in bold. The results for Genetic GFN \citep{kim2024geneticguidedgflownetssampleefficient} and Mol GA \citep{tripp2023geneticalgorithmsstrongbaselines} are taken from the respective papers. The other results are taken from the original PMO benchmark paper by \citep{gao2022sampleefficiencymattersbenchmark}.}"
else:
    caption = r"\caption{Comparison of models that initialize by screening ZINC250k on the PMO benchmark. We report the AUC-top10 scores, averaged over three runs"
    if args.include_std:
        caption += r" with standard deviations"
    caption += r". Best results and those within one standard deviation of the best are indicated in bold. The scores for $f$-RAG \citep{lee2024moleculegenerationfragmentretrieval} and GenMol \cite{genmol} are taken from the respective publications.}"

latex.append(caption)
latex.append(r"\label{tab:our_vs_baselines} " if not args.exclude_prescreen else r"\label{tab:prescreened}")
latex.append(r"\begin{tabularx}{\linewidth}{l|C "+("Y " * (len(methods)-1))+"}")
latex.append(r"\toprule")
latex.append("Oracle & " + " & ".join(latex_headers) + r" \\")
latex.append(r"\midrule")

for _, row in merged.iterrows():
    values = []
    stds = []

    # Collect values and stds
    for i, m in enumerate(methods):
        val = row[m] if pd.notna(row[m]) else 0.0
        values.append(val)

        if args.include_std and i < len(std_methods):
            std_col = std_methods[i]
            if std_col in row and pd.notna(row[std_col]):
                stds.append(row[std_col])
            else:
                stds.append(0.0)
        else:
            stds.append(0.0)

    # Find the best value
    max_val = max(values)
    max_idx = values.index(max_val)
    max_std = stds[max_idx]

    # Format each value
    row_fmt = []
    for i, (v, s) in enumerate(zip(values, stds)):
        if i == max_idx:
            # This is the best value
            if args.include_std and s > 0:
                row_fmt.append(f"$\\mathbf{{{v:.3f} \\pm {s:.3f}}}$")
            else:
                row_fmt.append(f"$\\mathbf{{{v:.3f}}}$")
        else:
            # Check if this value is within one std of the best
            within_std = (v + s >= max_val - max_std) if args.include_std else False

            if within_std and v != max_val:
                # Within one std of the best, mark with bold
                if args.include_std and s > 0:
                    row_fmt.append(f"$\\mathbf{{{v:.3f} \\pm {s:.3f}}}$")
                else:
                    row_fmt.append(f"$\\mathbf{{{v:.3f}}}$")
            else:
                # Regular formatting
                if args.include_std and s > 0:
                    row_fmt.append(f"${v:.3f} \\pm {s:.3f}$")
                else:
                    row_fmt.append(f"{v:.3f}")

    latex.append(f"{row['Oracle']} & " + " & ".join(row_fmt) + r" \\")

latex.append(r"\midrule")

# Calculate sums and use proper standard deviations
sums = []
sum_stds = []

# InVirtuoGen sum (already calculated above)
sums.append(invirtuo_sum_mean)
sum_stds.append(invirtuo_sum_std if args.include_std else 0.0)

# Other methods' sums
for i in range(1, len(methods)):
    m = methods[i]
    col_sum = merged[m].sum()
    sums.append(col_sum)

    if args.include_std:
        # Use the sum_std column if available, otherwise use 0
        sum_std_col = sum_std_methods[i]
        if sum_std_col in reference.columns and not reference[sum_std_col].isna().all():
            # If we have pre-computed sum std for this method, use it
            sum_stds.append(reference[sum_std_col].iloc[0])
        else:
            # Otherwise default to 0 (we don't have individual run data for reference methods)
            sum_stds.append(0.0)
    else:
        sum_stds.append(0.0)

# Find best sum
max_sum = max(sums)
max_sum_idx = sums.index(max_sum)
max_sum_std = sum_stds[max_sum_idx]

# Format sums
sum_fmt = []
for i, (s, std) in enumerate(zip(sums, sum_stds)):
    if i == max_sum_idx:
        # This is the best sum
        if args.include_std and std > 0:
            sum_fmt.append(f"$\\mathbf{{{s:.3f} \\pm {std:.3f}}}$")
        else:
            sum_fmt.append(f"$\\mathbf{{{s:.3f}}}$")
    else:
        # Check if within one std of the best
        within_std = (s + std >= max_sum - max_sum_std) if args.include_std else False

        if within_std and s != max_sum:
            if args.include_std and std > 0:
                sum_fmt.append(f"$\\mathbf{{{s:.3f} \\pm {std:.3f}}}$")
            else:
                sum_fmt.append(f"$\\mathbf{{{s:.3f}}}$")
        else:
            if args.include_std and std > 0:
                sum_fmt.append(f"${s:.3f} \\pm {std:.3f}$")
            else:
                sum_fmt.append(f"{s:.3f}")

latex.append(f"\\textbf{{Sum}} & " + " & ".join(sum_fmt) + r" \\")
latex.append(r"\bottomrule")
latex.append(r"\end{tabularx}")

latex.append(r"\end{table}")

print("\n".join(latex))

# Also save to file
output_file = "pmo_comparison_table"
if args.include_std:
    output_file += "_with_std"
if args.exclude_prescreen:
    output_file += "_no_prescreen"
output_file += ".tex"

with open(output_file, 'w') as f:
    f.write("\n".join(latex))
print(f"\nTable saved to {output_file}")