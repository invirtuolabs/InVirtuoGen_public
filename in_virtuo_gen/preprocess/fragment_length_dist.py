"""
build_vocab_with_running_means.py

This script builds a fixed‐size fragment vocabulary annotated with running mean
scores and counts, based on QED evaluations of fragments extracted from ZINC.
Fragments appearing multiple times across molecules will have their mean score
updated via a running‐mean formula. The final vocabulary is sorted by mean score
descending, truncated to `vocab_size`, and saved as JSON:

[
    {
        "fragment_ids": [int, int, …],
        "score": float,
        "count": int
    },
    …
]
"""
import os
import re
import json
import pandas as pd
import tdc
from ..data.datamodule import IVGDataModule
from ..preprocess.generate_fragments import process_smiles
from ..utils.fragments import bridge_smiles_fragments, remove_stereochemistry
import tqdm
def build_fragment_vocab(
    frag_csv_path: str,
    data_path: str,
    val_data_path: str,
    tokenizer_path: str,
    vocab_size: int = 1000,
    max_frags: int = 5,
    batch_size: int = 100,
    oracle_name: str = "qed",
    save_max: bool = True
):
    """
    1. Loads the IVGDataModule to get the SMILES tokenizer.
    2. Reads raw fragment‐strings from `frag_csv_path`.
    3. Bridges them to full SMILES, re‐fragments (up to `max_frags`), cleans tags.
    4. Scores EVERY occurrence with the TDC QED oracle.
    5. Maintains a running mean and count per fragment (token‐ID tuple).
    6. Sorts by descending mean, keeps the top `vocab_size`.
    7. Saves to configs/vocab.json.
    """
    # ——— 1) Instantiate data module to grab tokenizer ———
    dm = IVGDataModule(
        data_path=data_path,
        val_data_path=val_data_path,
        tokenizer_path=tokenizer_path,
        bucket=True,
        n_conds=0,
        val_batch_size=batch_size,
        batch_size=batch_size,


    )
    dm.setup("fit")
    tokenizer = dm.tokenizer
    name="configs/vocab_{}_{}.json".format(oracle_name,"max" if save_max else "mean")
    # if os.path.exists(name):
    #     return
    # oracle = tdc.Oracle(name=oracle_name)
    frag_stats      = {}  # maps tuple(token_ids) → [score, count]
    frag_ids_cache  = {}  # maps fragment string → tuple(token_ids)
    max_score = 0
    reader = pd.read_csv(frag_csv_path, header=None).itertuples(index=False)
    pbar = tqdm.tqdm(reader, desc=f"Building vocab, max score {max_score}")
    for row in pbar:
        if row[0].find("@")>-1:
            raise
        full_smi = bridge_smiles_fragments(row[0].split(), print_flag=True)
        if full_smi=="CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O" or full_smi.find(".")>-1:
            print("found a bad one")
            raise
        frags     = process_smiles(full_smi, max_frags=max_frags)[0].split()

        # score the *whole* molecule once


        for f in frags:
            score = 0# oracle(f)
            if score > max_score:
                max_score = score
                # update the progress‐bar description in place
                pbar.set_description(f"Building vocab, max score {max_score}")
            n = f.count("*]")
            if n not in (1, 2):
                continue

            parts = f.split("*]")
            if n == 1:
                frag ="[1*]" + parts[1]
            else:
                frag = "[1*]" + parts[1][:-2] + "[2*]" + parts[2]

            # tokenize with caching
            ids = frag_ids_cache.get(frag)
            if ids is None:
                ids = tuple(tokenizer.encode(frag))
                frag_ids_cache[frag] = ids

            # update running stats with the *molecule* score
            if save_max:
                prev = frag_stats.get(ids, [0, 1])[0]
                frag_stats[ids] = [max(score, prev), 1]
            else:
                if ids not in frag_stats:
                    frag_stats[ids] = [score, 1]
                else:
                    old_mean, old_count = frag_stats[ids]
                    new_count = old_count + 1
                    new_mean  = (old_mean * old_count + score) / new_count
                    frag_stats[ids] = [new_mean, new_count]

    sorted_stats = sorted(
        frag_stats.items(),
        key=lambda kv: kv[1][0],  # kv = (frag_ids, [mean, count])
        reverse=True
    )[:vocab_size]

    vocab_list = [
        {
            "fragment_ids": list(frag_ids),
            "score": value[0],
            "count": value[1],
        }
        for frag_ids, value in sorted_stats
    ]

    with open(name, "w") as f:
        json.dump(vocab_list, f, indent=2)

    print(f"Saved {len(vocab_list)} fragments with running means to {name}")

if __name__ == "__main__":
    # for oracle in ["albuterol_similarity","amlodipine_mpo","celecoxib_rediscovery","deco_hop","drd2","fexofenadine_mpo","gsk3b","isomers_c7h8n2o2","isomers_c9h10n2o2pf2cl","jnk3","median1","median2","mestranol_similarity","osimertinib_mpo","perindopril_mpo","qed","ranolazine_mpo","scaffold_hop","sitagliptin_mpo","thiothixene_rediscovery","troglitazone_rediscovery","valsartan_smarts","zaleplon_mpo"]:
        # try:
            build_fragment_vocab(
                frag_csv_path="frags_zinc250.csv",
                data_path="/home/bkaech/projects/InVirtuoGEN/mmap_buckets_mixed_partial",
                val_data_path="/home/bkaech/projects/InVirtuoGEN/mmap_buckets_mixed_partial",
                tokenizer_path="/home/bkaech/projects/InVirtuoGEN/tokenizer/smiles_new.json",
                vocab_size=100000,
                max_frags=5,
                batch_size=1000,
                oracle_name='default',#oracle,
                save_max=True
            )
        # except Exception as e:
        #     print(e)
        #     import traceback
        #     traceback.print_exc()