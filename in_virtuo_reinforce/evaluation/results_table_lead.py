import re
import glob
import os
import pandas as pd
import numpy as np

def parse_baseline_table(raw_text):
    """
    Parse the baseline table from raw text into a structured DataFrame
    """
    lines = raw_text.strip().splitlines()

    rows = []
    current_target = None

    for line in lines:
        # Skip header line
        if 'Target' in line or line.strip() == '':
            continue

        # Split by | character
        parts = line.split('|')
        if len(parts) < 8:
            continue

        # Extract target if present
        if parts[0].strip() and not parts[0].strip().replace('.', '').replace('-', '').replace(' ', '').isdigit():
            current_target = parts[0].strip()

        # Skip if we don't have a current target
        if current_target is None:
            continue

        def parse_value(val):
            val = val.strip()
            if val == '-' or val == '':
                return None
            try:
                return float(val)
            except:
                return None

        # Parse the values
        try:
            seed_score = parse_value(parts[1])
            if seed_score is None:  # Skip if no seed score
                continue

            row = {
                'target': current_target,
                'seed_score': seed_score,
                'GenMol_0.4': parse_value(parts[2]),
                'RetMol_0.4': parse_value(parts[3]),
                'GraphGA_0.4': parse_value(parts[4]),
                'GenMol_0.6': parse_value(parts[5]),
                'RetMol_0.6': parse_value(parts[6]),
                'GraphGA_0.6': parse_value(parts[7]) if len(parts) > 7 else None,
            }
            rows.append(row)
        except:
            continue

    return pd.DataFrame(rows)

def baseline_df_to_multiindex(df):
    """
    Convert baseline DataFrame to MultiIndex format
    """
    # Group by target and assign indices
    targets = []
    indices = []
    data_dict = {}

    for target in df['target'].unique():
        target_df = df[df['target'] == target].reset_index(drop=True)
        for idx in range(len(target_df)):
            targets.append(target)
            indices.append(idx)

            row_data = target_df.iloc[idx]
            for threshold in ['0.4', '0.6']:
                for method in ['GenMol', 'RetMol', 'GraphGA']:
                    col_name = f'{method}_{threshold}'
                    value = row_data.get(col_name)

                    key = (target, idx, f'δ = {threshold}', method)
                    if pd.notna(value):
                        data_dict[key] = f"{value:.1f}"
                    else:
                        data_dict[key] = "-"

    # Create MultiIndex DataFrame
    row_index = pd.MultiIndex.from_arrays([targets, indices], names=['protein', 'idx'])

    # Create columns
    thresholds = ['δ = 0.4', 'δ = 0.6']
    methods = ['GenMol', 'RetMol', 'GraphGA']
    column_tuples = [(thr, method) for thr in thresholds for method in methods]
    columns = pd.MultiIndex.from_tuples(column_tuples, names=['threshold', 'method'])

    # Create DataFrame
    result_df = pd.DataFrame(index=row_index, columns=columns)

    # Populate DataFrame
    for (target, idx, threshold, method), value in data_dict.items():
        if (target, idx) in result_df.index and (threshold, method) in result_df.columns:
            result_df.loc[(target, idx), (threshold, method)] = value

    return result_df

def load_your_results(results_dir, method_name="InVirtuo"):
    """
    Load your experimental results from CSV files
    """
    rx = re.compile(r"docking_(?P<target>[\w\-]+)_idx(?P<idx>\d+)_thr(?P<thr>[46])\.csv")
    files = glob.glob(os.path.join(results_dir, "**/docking_*_idx*_thr*.csv"), recursive=True)

    # Collect all unique values
    targets = set()
    idxs = set()
    thrs = set()

    for fn in files:
        m = rx.search(os.path.basename(fn))
        if not m:
            continue
        target, idx, thr = m.group("target"), int(m.group("idx")), m.group("thr")
        targets.add(target)
        idxs.add(idx)
        thrs.add(thr)

    targets = sorted(list(targets))
    idxs = sorted(list(idxs))
    thrs = sorted(list(thrs))

    # Create results dictionary
    results = {}

    for target in targets:
        results[target] = {}
        for idx in idxs:
            results[target][idx] = {}
            for thr in thrs:
                fn = os.path.join(results_dir, f"docking_{target}_idx{idx}_thr{thr}.csv")

                if not os.path.exists(fn):
                    results[target][idx][thr] = []
                    continue

                try:
                    df_temp = pd.read_csv(fn)

                    # Check required columns
                    required_cols = ['sim', 'qed', 'sa', 'seed', 'docking score']
                    if not all(col in df_temp.columns for col in required_cols):
                        print(f"Warning: Missing columns in {fn}")
                        results[target][idx][thr] = []
                        continue

                    # Apply filters
                    thr_val = float(thr) / 10.0
                    df_filtered = df_temp[
                        (df_temp['sim'] > thr_val) &
                        (df_temp['qed'] > 0.6) &
                        (df_temp['sa'] < 4)
                    ]

                    if df_filtered.empty:
                        results[target][idx][thr] = []
                        continue

                    # Get best (max) docking score for each seed
                    # Your scores are positive (absolute values), so max() gets the best score
                    seed_best_scores = df_filtered.groupby("seed")["docking score"].max()
                    # Convert to negative to match convention
                    seed_best_scores = [-score for score in seed_best_scores.tolist()]
                    results[target][idx][thr] = seed_best_scores

                except Exception as e:
                    print(f"Error processing {fn}: {e}")
                    results[target][idx][thr] = []

    # Create MultiIndex DataFrame
    row_tuples = [(target, idx) for target in targets for idx in idxs]
    row_index = pd.MultiIndex.from_tuples(row_tuples, names=['protein', 'idx'])

    column_tuples = [(f"δ = 0.{thr}", method_name) for thr in thrs]
    columns = pd.MultiIndex.from_tuples(column_tuples, names=['threshold', 'method'])

    df_final = pd.DataFrame(index=row_index, columns=columns)

    # Populate the table
    for target in targets:
        for idx in idxs:
            for thr in thrs:
                scores = results[target][idx][thr]

                if not scores:
                    value = "-"
                elif len(scores) == 1:
                    value = f"{scores[0]:.1f}"
                else:
                    mean_val = np.mean(scores)
                    std_val = np.std(scores, ddof=1)
                    value = f"{mean_val:.1f}±{std_val:.1f}"

                df_final.loc[(target, idx), (f"δ = 0.{thr}", method_name)] = value

    return df_final

def combine_results(baseline_df, your_df):
    """
    Combine baseline and your results into a single DataFrame
    """
    # Get all unique indices from both DataFrames, maintaining original order
    baseline_indices = baseline_df.index.tolist()
    your_indices = your_df.index.tolist()

    # Combine indices while preserving baseline order first
    all_indices = []
    for idx in baseline_indices:
        if idx not in all_indices:
            all_indices.append(idx)
    for idx in your_indices:
        if idx not in all_indices:
            all_indices.append(idx)

    # Get all unique columns - preserving the order correctly
    all_columns = []

    # First add all columns from baseline_df
    for col in baseline_df.columns:
        if col not in all_columns:
            all_columns.append(col)

    # Then add all columns from your_df
    for col in your_df.columns:
        if col not in all_columns:
            all_columns.append(col)

    # Create combined DataFrame with proper MultiIndex
    combined_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(all_indices),
                               columns=pd.MultiIndex.from_tuples(all_columns))

    # Fill in baseline data
    for idx in baseline_df.index:
        for col in baseline_df.columns:
            value = baseline_df.loc[idx, col]
            combined_df.loc[idx, col] = value if value != "-" and pd.notna(value) else "-"

    # Fill in your data
    for idx in your_df.index:
        for col in your_df.columns:
            value = your_df.loc[idx, col]
            combined_df.loc[idx, col] = value if value != "-" and pd.notna(value) else "-"

    # Fill any remaining NaN values with "-"
    combined_df = combined_df.fillna("-")

    return combined_df

def create_latex_table(df, caption="Comparison of docking scores (lower is better)", label="tab:docking_comparison", baseline_raw_df=None):
    """
    Create a LaTeX table from the combined DataFrame with sum row and best values marked
    """
    # Create a mapping from (target, idx) to seed_score if baseline_raw_df is provided
    seed_score_map = {}
    if baseline_raw_df is not None:
        for target in baseline_raw_df['target'].unique():
            target_df = baseline_raw_df[baseline_raw_df['target'] == target].reset_index(drop=True)
            for idx in range(len(target_df)):
                seed_score = target_df.iloc[idx]['seed_score']
                seed_score_map[(target, idx)] = seed_score

    # Get unique thresholds and methods
    thresholds = df.columns.get_level_values(0).unique()
    methods_per_threshold = {}
    for thr in thresholds:
        methods_per_threshold[thr] = [m for t, m in df.columns if t == thr]

    # Helper function to extract mean and std from value
    def parse_value(value):
        if value == "-" or pd.isna(value):
            return None, None
        value_str = str(value)
        if "±" in value_str:
            parts = value_str.split("±")
            return float(parts[0]), float(parts[1])
        else:
            return float(value_str), 0.0

    # Calculate sums for each method and threshold (excluding bracketed values)
    sums = {}
    for thr in thresholds:
        for method in methods_per_threshold[thr]:
            col_sum = 0
            count = 0
            for idx in df.index:
                value = df.loc[idx, (thr, method)]
                mean_val, _ = parse_value(value)
                if mean_val is not None:
                    # Get seed score for this row
                    seed_score = seed_score_map.get(idx, None)
                    # Only include in sum if better than seed (more negative)
                    if seed_score is None or mean_val < seed_score:
                        col_sum += mean_val
                        count += 1
            sums[(thr, method)] = col_sum if count > 0 else None

    # Find best (minimum) sum for each threshold
    best_per_threshold = {}
    for thr in thresholds:
        min_sum = float('inf')
        best_method = None
        for method in methods_per_threshold[thr]:
            if sums[(thr, method)] is not None and sums[(thr, method)] < min_sum:
                min_sum = sums[(thr, method)]
                best_method = method
        best_per_threshold[thr] = best_method

    # Start building LaTeX
    latex_lines = []

    # Table header
    latex_lines.extend([
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",  # Make table smaller if needed
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}"
    ])

    # Calculate total columns
    total_cols = 2  # Protein and Seed columns
    for methods in methods_per_threshold.values():
        total_cols += len(methods)

    # Column specification
    col_spec = "ll" + "c" * (total_cols - 2)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append(r"\toprule")

    # Header row 1: thresholds with \delta
    header1_parts = ["Protein", "Seed"]
    for threshold in thresholds:
        n_methods = len(methods_per_threshold[threshold])
        # Replace δ with $\delta$
        threshold_latex = threshold.replace("δ", "$\\delta$")
        header1_parts.append(f"\\multicolumn{{{n_methods}}}{{c}}{{{threshold_latex}}}")
    latex_lines.append(" & ".join(header1_parts) + r" \\")

    # Add cmidrule for each threshold group
    col_start = 3  # Start after Protein and Seed columns
    cmidrules = []
    for threshold in thresholds:
        n_methods = len(methods_per_threshold[threshold])
        col_end = col_start + n_methods - 1
        cmidrules.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")
        col_start = col_end + 1
    if cmidrules:
        latex_lines.append(" ".join(cmidrules))

    # Header row 2: methods
    header2_parts = ["", ""]  # Empty for Protein and Seed
    for threshold in thresholds:
        for method in methods_per_threshold[threshold]:
            header2_parts.append(method)
    latex_lines.append(" & ".join(header2_parts) + r" \\")
    latex_lines.append(r"\midrule")

    # Data rows
    current_protein = None
    for (protein, idx), row in df.iterrows():
        # Get seed score for this row
        seed_score = seed_score_map.get((protein, idx), None)

        # Find best value for this row for each threshold (excluding bracketed values)
        best_values_per_threshold = {}
        for threshold in thresholds:
            values_with_method = []
            for method in methods_per_threshold[threshold]:
                value = row.get((threshold, method), "-")
                mean_val, std_val = parse_value(value)
                if mean_val is not None:
                    # Only consider for "best" if better than seed
                    if seed_score is None or mean_val < seed_score:
                        values_with_method.append((mean_val, std_val, method, value))

            if values_with_method:
                # Find minimum mean value
                min_mean = min(v[0] for v in values_with_method)
                best_values_per_threshold[threshold] = []

                # Mark all values that are best or within 1 std of best
                for mean_val, std_val, method, orig_value in values_with_method:
                    # Find the std of the best value(s)
                    best_std = max(v[1] for v in values_with_method if v[0] == min_mean)
                    # A value is marked if it's the best OR within 1 std of the best
                    if mean_val == min_mean or mean_val <= min_mean + best_std:
                        best_values_per_threshold[threshold].append(method)

        # Add spacing between different proteins
        if protein != current_protein:
            if current_protein is not None:
                latex_lines.append(r"\addlinespace[0.5em]")
            current_protein = protein
            row_data = [protein]  # Show protein name
        else:
            row_data = [""]  # Empty for repeated protein

        # Add seed score
        if seed_score is not None:
            seed_score_str = f"{seed_score:.1f}"
        else:
            seed_score_str = str(idx)
        row_data.append(seed_score_str)

        # Add data for each threshold and method
        for threshold in thresholds:
            for method in methods_per_threshold[threshold]:
                value = row.get((threshold, method), "-")
                if pd.isna(value):
                    value_str = "-"
                else:
                    value_str = str(value)
                    # Replace ± with \pm for LaTeX
                    value_str = value_str.replace("±", r"$\pm$")

                    # Check if value is worse than seed (should be in brackets)
                    mean_val, _ = parse_value(value)
                    if mean_val is not None and seed_score is not None and mean_val >= seed_score:
                        # Put in brackets if not better than seed
                        value_str = f"({value_str})"
                    else:
                        # Mark best values with textbf (only if not bracketed)
                        if threshold in best_values_per_threshold and method in best_values_per_threshold[threshold]:
                            value_str = f"\\textbf{{{value_str}}}"

                row_data.append(value_str)

        latex_lines.append(" & ".join(row_data) + r" \\")

    # Add sum row
    latex_lines.append(r"\midrule")
    sum_row = ["\\multicolumn{2}{l}{Sum}", ""]  # Merge first two columns for "Sum"
    for threshold in thresholds:
        for method in methods_per_threshold[threshold]:
            if sums[(threshold, method)] is not None:
                sum_val = f"{sums[(threshold, method)]:.1f}"
                # Mark best (minimum) value with \textbf
                if method == best_per_threshold[threshold]:
                    sum_val = f"\\textbf{{{sum_val}}}"
                sum_row.append(sum_val)
            else:
                sum_row.append("-")

    # Remove the empty string we added after the multicolumn
    sum_row = [sum_row[0]] + sum_row[2:]
    latex_lines.append(" & ".join(sum_row) + r" \\")

    # Table footer
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    return "\n".join(latex_lines)

# Main execution
if __name__ == "__main__":
    # Raw baseline data
    raw_baseline = """
Target | Seed score | GenMol | RetMol | Graph GA | GenMol | RetMol | Graph GA
parp1 | -7.3 | -10.6 | -9.0 | -8.3 | -10.4 | - | -8.6
 | -7.8 | -11.0 | -10.7 | -8.9 | -9.7 | - | -8.1
 | -8.2 | -11.3 | -10.9 | - | -9.2 | - | -
fa7 | -6.4 | -8.4 | -8.0 | -7.8 | -7.3 | -7.6 | -7.6
 | -6.7 | -8.4 | - | -8.2 | -7.6 | - | -7.6
 | -8.5 | - | - | - | - | - | -
5ht1b | -4.5 | -12.9 | -12.1 | -11.7 | -12.1 | - | -11.3
 | -7.6 | -12.3 | -9.0 | -12.1 | -12.0 | -10.0 | -12.0
 | -9.8 | -11.6 | - | - | -10.5 | - | -
braf | -9.3 | -10.8 | -11.6 | -9.8 | - | - | -
 | -9.4 | -10.8 | - | - | -9.7 | - | -
 | -9.8 | -10.6 | - | -11.6 | -10.5 | - | -10.4
jak2 | -7.7 | -10.2 | -8.2 | -8.7 | -9.3 | -8.1 | -
 | -8.0 | -10.0 | -9.0 | -9.2 | -9.4 | - | -9.2
 | -8.6 | -9.8 | - | - | - | - | -
"""

    # Parse baseline data
    baseline_raw_df = parse_baseline_table(raw_baseline)
    print("Parsed baseline data:")
    print(baseline_raw_df)
    print("\n" + "="*80 + "\n")

    # Convert to MultiIndex format
    baseline_df = baseline_df_to_multiindex(baseline_raw_df)
    print("Baseline data in MultiIndex format:")
    print(baseline_df)
    print("\n" + "="*80 + "\n")

    # Load your results (update the path)
    results_dir = "/home/bkaech/projects/InVirtuoGEN/results/lead_optimization/20250813_115916"

    # Check if directory exists before loading
    if os.path.exists(results_dir):
        your_df = load_your_results(results_dir, method_name="InVirtuo")
        print("Your results:")
        print(your_df)
        print("\n" + "="*80 + "\n")

        # Combine results
        combined_df = combine_results(baseline_df, your_df)

        # Display combined results more clearly
        print("Combined results (first checking a sample):")
        print(f"Sample - parp1, idx 0:")
        print(f"  Baseline δ=0.4: GenMol={combined_df.loc[('parp1', 0), ('δ = 0.4', 'GenMol')]}, "
              f"RetMol={combined_df.loc[('parp1', 0), ('δ = 0.4', 'RetMol')]}, "
              f"GraphGA={combined_df.loc[('parp1', 0), ('δ = 0.4', 'GraphGA')]}")
        print(f"  Baseline δ=0.6: GenMol={combined_df.loc[('parp1', 0), ('δ = 0.6', 'GenMol')]}, "
              f"RetMol={combined_df.loc[('parp1', 0), ('δ = 0.6', 'RetMol')]}, "
              f"GraphGA={combined_df.loc[('parp1', 0), ('δ = 0.6', 'GraphGA')]}")
        print(f"  InVirtuo: δ=0.4={combined_df.loc[('parp1', 0), ('δ = 0.4', 'InVirtuo')]}, "
              f"δ=0.6={combined_df.loc[('parp1', 0), ('δ = 0.6', 'InVirtuo')]}")
        print("\nFull combined table (may wrap due to width):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(combined_df)
        print("\n" + "="*80 + "\n")

        # Create LaTeX table
        latex_table = create_latex_table(combined_df, baseline_raw_df=baseline_raw_df)
        print("LaTeX table:")
        print(latex_table)
    else:
        print(f"Warning: Results directory '{results_dir}' not found.")
        print("Creating LaTeX table with baseline data only:")
        latex_table = create_latex_table(baseline_df, baseline_raw_df=baseline_raw_df)
        print(latex_table)