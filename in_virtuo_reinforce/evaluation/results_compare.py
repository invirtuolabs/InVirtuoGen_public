import os, pandas as pd, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results_root", type=str, default="results_root")
parser.add_argument("--out_dir",       type=str, default="target_property")
args = parser.parse_args()

# discover hyperparams
hp_names = sorted(
    name for name in os.listdir(args.results_root)
    if os.path.isdir(os.path.join(args.results_root, name))
)
hp_paths = [os.path.join(args.results_root, hp) for hp in hp_names]

# collect sum of auc_top10 per timestamp per hp
data = {}
for  base in hp_paths:
    for ts in os.listdir(base):
        ts_dir = os.path.join(base, ts)
        if not os.path.isdir(ts_dir):
            continue
        sum_auc = 0

        for task in os.listdir(ts_dir):
            for file in os.listdir(os.path.join(ts_dir, task)):

                if file.endswith(".csv") and file.startswith("results_"):
                    csv = os.path.join(ts_dir, task, file)
                    df = pd.read_csv(csv)
                    total = df["auc_top10"].mean()
                    sum_auc += total
        data[ts] = sum_auc
        print(ts, sum_auc)

print(data)

# build DataFrame: index=timestamp, cols=hp_names
df = (
    pd.DataFrame.from_dict(data, orient="index")
)

print(df)
os.makedirs(args.out_dir, exist_ok=True)
df.to_csv(os.path.join(args.out_dir, "sum_top10_by_timestamp.csv"))
print("Wrote:", os.path.join(args.out_dir, "sum_top10_by_timestamp.csv"))