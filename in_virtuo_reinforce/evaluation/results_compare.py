import os, pandas as pd, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results_root", nargs="+", default="results_root")
parser.add_argument("--out_dir",       type=str, default="target_property")
args = parser.parse_args()



# collect sum of auc_top10 per timestamp per hp
data = {}
task_df = pd.DataFrame(columns=os.listdir(args.results_root[0]))
for run in args.results_root:

    sum_auc = 0

    for task in os.listdir(run):
        for file in os.listdir(os.path.join(run, task)):

            if file.endswith(".csv") and file.startswith("results_"):
                csv = os.path.join(run, task, file)
                df = pd.read_csv(csv)
                total = df["auc_top10"].mean()
                sum_auc += total
                task_df.loc[run, task] = total
    data[run] = sum_auc

print(task_df)
# build DataFrame: index=timestamp, cols=hp_names
df = (
    pd.DataFrame.from_dict(data, orient="index")
)

os.makedirs(args.out_dir, exist_ok=True)
df.to_csv(os.path.join(args.out_dir, "sum_top10_by_timestamp.csv"))
task_df.to_csv(os.path.join(args.out_dir, "sum_top10_by_task.csv"))
print("Wrote:", os.path.join(args.out_dir, "sum_top10_by_timestamp.csv"))