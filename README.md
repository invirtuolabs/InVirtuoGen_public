# InVirtuoGEN Benchmark

This repository contains code and pretrained checkpoints for **InVirtuoGEN**, a discrete flow model for fragment-based molecular generation and optimization.

## Checkpoints

- `invirtuo_big.ckpt`
- `invirtuo.ckpt`

**Download from Zenodo:** [https://zenodo.org/records/16874868](https://zenodo.org/records/16874868)
```bash
mkdir checkpoints
cd checkpoints
curl https://zenodo.org/api/records/16874868/files-archive -o checkpoints.zip
unzip checkpoints.zip
unzip invirtuo_gen_big.ckpt.zip
unzip invirtuo_gen.ckpt.zip
cd ..

```
## (0) Setup Environment

```bash
mamba create -n invgen python=3.10.16 -y
mamba activate invgen
pip install -r requirements.txt
pip install -e .
```

## (1) De Novo Molecule Generation

```bash
python -m in_virtuo_gen.evaluation.denovo \
    --outdir plots \
    --device 0 \
    --dt 0.1 0.01 0.001 \
    --checkpoint_path checkpoints/invirtuo_gen_big.ckpt \
    --num_samples 1000 \
    --eta 999 \
    --num_seeds 3
```

## (2) Downstream Tasks Evaluation

```bash
python -m in_virtuo_gen.evaluation.downstream \
    --model_path checkpoints/invirtuo_gen.ckpt \
    --dt 0.001 \
    --temperature 1 \
    --all \
    --noise 0 \
    --device 0 \
    --eta 1
```

## (3) Target Property Optimization

### (a) Build Fragment Vocabulary

To reproduce the PMO results starting with a prescreened initial population, download the fragmented dataset from Zenodo and build the vocabulary:

```bash
mkdir -p data
wget -O data/frags_zinc250.csv https://zenodo.org/records/16742898/files/frags_zinc250.csv
python -m in_virtuo_reinforce.preprocess.get_vocab \
    --datapath data/frags_zinc250.csv \
    --outpath in_virtuo_reinforce/vocab/zinc250k.csv
```

### (b) Run Optimization

**Runtime:** ~24h on RTX 4090 with 3 random seeds.

```bash
python -m in_virtuo_reinforce.genetic_ppo \
    --ckpt path/to/invirtuo.ckpt \
    --device 0 \
    --start_t 0.0 \
    --offspring_size 64 \
    --start_task 0 \
    --seed 0 \
    --max_oracle_calls 10000 \
    --alpha 0.01 \
    --num_reinforce_steps 10 \
    --clip_eps 0.5 \
    --use_prompter \
    --sample_uni \
    --use_prescreen \
    --use_mut \
    --train_mut
```

### (c) Produce Results Table

```bash
python -m in_virtuo_reinforce.evaluation.results_table \
    --results_root results/target_property_optimization/{TIMESTAMP}
```

## PMO Benchmark Results (TOP-10 AUC) using prescreening

| Oracle | InVirtuoGEN | GenMol | f-RAG |
|--------|:-----------:|:------:|:-----:|
| albuterol similarity | **0.983** | 0.937 | 0.977 |
| amlodipine mpo | **0.819** | 0.810 | 0.749 |
| celecoxib rediscovery | **0.845** | 0.826 | 0.778 |
| deco hop | **0.985** | 0.960 | 0.936 |
| drd2 | **0.993** | 0.995 | 0.992 |
| fexofenadine mpo | **0.900** | 0.894 | 0.856 |
| gsk3b | **0.990** | 0.986 | 0.969 |
| isomers c7h8n2o2 | **0.974** | 0.942 | 0.955 |
| isomers c9h10n2o2pf2cl | **0.901** | 0.833 | 0.850 |
| jnk3 | **0.930** | 0.906 | 0.904 |
| median1 | 0.386 | **0.398** | 0.340 |
| median2 | **0.363** | 0.359 | 0.323 |
| mestranol similarity | **0.989** | 0.982 | 0.671 |
| osimertinib mpo | 0.864 | **0.876** | 0.866 |
| perindopril mpo | **0.740** | 0.718 | 0.681 |
| qed | **0.943** | 0.942 | 0.939 |
| ranolazine mpo | **0.859** | 0.821 | 0.820 |
| scaffold hop | **0.936** | 0.628 | 0.576 |
| sitagliptin mpo | **0.669** | 0.584 | 0.601 |
| thiothixene rediscovery | **0.737** | 0.692 | 0.584 |
| troglitazone rediscovery | 0.747 | **0.867** | 0.448 |
| valsartan smarts | **0.920** | 0.822 | 0.627 |
| zaleplon mpo | **0.694** | 0.584 | 0.486 |
| **Sum** | **19.167** | 18.362 | 16.928 |