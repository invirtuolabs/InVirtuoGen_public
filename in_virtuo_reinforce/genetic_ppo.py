import json
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import lightning
import numpy as np
import pandas as pd
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import ConcatDataset

from lightning.pytorch import Trainer
import random
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import get_cosine_schedule_with_warmup
from in_virtuo_gen.utils.mol import compute_properties
import wandb
from in_virtuo_gen.models.invirtuofm import InVirtuoFM
from in_virtuo_gen.preprocess.preprocess_tokenize import custom_decode_sequence
from in_virtuo_gen.train_utils.metrics import evaluate_smiles
from in_virtuo_gen.utils.fragments import bridge_smiles_fragments, order_fragments_by_attachment_points, smiles2frags
from contextlib import nullcontext
from .pbo import BaseOptimizer
from .utils import SoftmaxBandit, decompose_smiles, visualize_top_smiles, ExperienceReplay, GeneticPrompter
from .ga.ga import reproduce
from .train_utils import compute_seq_logp, sample_path, filter_valid_new, custom_collate
from dataclasses import asdict, dataclass
import sascorer
from rdkit import Chem
from rdkit.Chem import QED


@dataclass
class OptimizerConfig:
    """Configuration class for InVirtuoFMOptimizer"""

    ckpt: str
    oracle: str
    lr: float
    target: str
    offspring_size: int
    device: int
    start_t: float
    num_reinforce_steps: int
    big: bool
    alpha: float
    beta: float
    output_dir: str
    dt: float
    temperature: float
    use_prompter: bool
    clip_eps: float
    c_neg: float
    use_mutation: bool
    identifier: str
    reverse_kl: bool
    start_seed: int
    seed: int
    mutation_size: int
    task: str
    experience_replay_size: int
    train_mutation: bool
    # Additional configs that might be passed via *args, **kwargs
    max_oracle_calls: int
    no_sample_uni: bool
    use_prescreen: bool
    vocab_size: int
    first_pop_size: int
    vocab_mask: bool
    rl_lr: float
    no_stop: bool = False
    mse_loss: bool = False
    max_frags: int = 5


def compute_kl_divergence(logits_new, logits_old, source_mask, reverse=False):
    """Compute KL divergence for PPO: KL(π_new || π_old)"""
    if reverse:
        logits_new, logits_old = logits_old, logits_new
    kl_per_token = F.kl_div(F.log_softmax(logits_old, dim=-1), F.log_softmax(logits_new, dim=-1), reduction="none", log_target=True).sum(dim=-1)  # input  # target

    # The above actually computes -KL(old || new), so we need to flip it
    # For proper KL(new || old), we should use:
    p_new = F.softmax(logits_new, dim=-1)
    log_p_new = F.log_softmax(logits_new, dim=-1)
    log_p_old = F.log_softmax(logits_old, dim=-1)

    kl_per_token = (p_new * (log_p_new - log_p_old)).sum(dim=-1)

    # Apply mask and average
    kl = (kl_per_token * source_mask.float()).sum() / source_mask.float().sum().clamp(min=1)
    return kl


def pad_collate_with_masks(batch):
    """
    Collate function for batches without replay masks (7 elements per sample).
    """
    ids, scores, uni, t, logprobs, x_ts, masks = zip(*batch)

    # Pad sequences
    ids_padded = pad_sequence(list(ids), batch_first=True, padding_value=0)
    uni_padded = pad_sequence(list(uni), batch_first=True, padding_value=0)

    # Convert scalars
    scores = torch.tensor(scores, dtype=torch.float, device=ids_padded.device)
    t = torch.tensor(t, dtype=torch.float, device=ids_padded.device)

    # Stack logprobs (these should already have the same shape)
    logprobs = torch.stack(logprobs, dim=0)

    # Handle x_ts and masks with variable sequence lengths
    batch_size = len(batch)
    num_timesteps = x_ts[0].shape[0]  # Should be num_reinforce_steps
    max_seq_len = ids_padded.shape[1]  # Maximum sequence length after padding

    # Initialize padded tensors
    x_ts_padded = torch.zeros(batch_size, num_timesteps, max_seq_len, dtype=x_ts[0].dtype, device=ids_padded.device)
    masks_padded = torch.zeros(batch_size, num_timesteps, max_seq_len, dtype=masks[0].dtype, device=ids_padded.device)

    # Fill in the values
    for i, (x_t, mask) in enumerate(zip(x_ts, masks)):
        seq_len = x_t.shape[-1]  # Original sequence length
        x_ts_padded[i, :, :seq_len] = x_t
        masks_padded[i, :, :seq_len] = mask

    return ids_padded, scores, uni_padded, t, logprobs, x_ts_padded, masks_padded




def sample_path(t, x_0, x_1, n=1):
    sigma_t = 1 - t**n
    source_indices = torch.rand(size=x_1.shape, device=x_1.device) < sigma_t.unsqueeze(-1)
    return torch.where(condition=source_indices, input=x_0, other=x_1), source_indices


def filter_valid_new(valid, smiles, off_seqs, all_x_0, mol_buffer):

    valid_new = [i for i in valid if smiles[i] not in mol_buffer]
    valid_new_seqs = [off_seqs[i] for i in valid_new]
    valid_new_smiles = [smiles[i] for i in valid_new]
    valid_new_x_0 = [all_x_0[i] for i in valid_new]

    return valid_new_smiles, valid_new_seqs, valid_new_x_0


def compute_seq_logp(logits, ids, source_mask):
    # logits: [B, T, V], ids: [B, T]
    # 1) log-softmax over vocab
    log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
    # 2) pick each token’s log-prob
    token_lp = log_probs.gather(-1, ids.unsqueeze(-1))  # [B, T, 1]
    token_lp = token_lp.squeeze(-1)  # [B, T]
    # 3) mask pads and sum over the sequence
    # [B, T]
    return (token_lp * source_mask).sum(dim=-1) / torch.max(torch.ones_like(source_mask[:, 0]), (source_mask.float().sum(dim=-1)))


class InVirtuoFMOptimizer(BaseOptimizer):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        base_optimizer_kwargs = {"device": kwargs.get("device", "cpu"), "output_dir": kwargs.get("output_dir", "target_prop_opt")}

        # Remove base optimizer args from kwargs to avoid conflicts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["device", "output_dir"] or k in OptimizerConfig.__dataclass_fields__}

        # Create config from provided arguments
        self.config = OptimizerConfig(**filtered_kwargs)

        # Initialize base optimizer
        super().__init__(**base_optimizer_kwargs)

        # Initialize wandb with config
        wandb.init(project="tdc-ppo-finetune", config=asdict(self.config))

        wandb.run.log_code("./")  # type: ignore[attr-defined]
        # Initialize instance variables from config
        self.device = torch.device(self.config.device)
        self.sa, self.qed, self.qed_ga, self.sa_ga, self.ga_scores, self.ga_tries = [], [], [], [], [], []
        self.global_step = 0
        self.stop_counter = 0

        # Model initialization
        ckpt = self.config.ckpt
        self.model = InVirtuoFM.load_from_checkpoint(ckpt, map_location="cpu").to(self.device)

        self.prior = InVirtuoFM.load_from_checkpoint(ckpt, map_location="cpu").to(self.device)
        self.prior.model.requires_grad = False
        self.tokenizer = self.model.tokenizer
        loaded = torch.load("configs/zinc_dist.pt")
        self.seq_lengths = loaded["unique_lengths"]
        self.seq_prior_probs = loaded["probs"]
        self.seq_len_dist = Categorical(self.seq_prior_probs.clone())
        # Initialize bandit
        self.bandit = SoftmaxBandit(prior_probs=self.seq_prior_probs.cpu().numpy(), lengths=self.seq_lengths, lr=self.config.lr, beta=self.config.beta)

        # Initialize prompter if needed
        if self.config.use_prompter:
            self.prompter = GeneticPrompter(
                tokenizer=self.tokenizer,
                bandit=self.bandit,
                offspring_size=int(self.config.offspring_size * 2),
                vocab_size=self.config.vocab_size,
                kappa=0.001,
                always_ok=None,
                max_frags=self.config.max_frags,
                pad_id=3,
                score_based=False,
                K=1 #if self.config.use_prescreen else 2,
            )

        self.model_opt = optim.Adam(self.model.parameters(), lr=self.config.rl_lr)
        self.lr_scheduler = get_cosine_schedule_with_warmup(self.model_opt, num_warmup_steps=0, num_training_steps=10000 * self.config.num_reinforce_steps // self.config.offspring_size * 2)

        # Initialize trainer
        trainer = Trainer(max_steps=1, limit_train_batches=0.0, logger=False)
        self.model.trainer = trainer
        self.used_mutation = not self.config.use_mutation
        self.prev_validity = 1
        # Initialize training data storage
        self.train_ids, self.train_prompts, self.train_scores, self.train_old_logprobs, self.train_x_0 = [], [], [], [], []

        self.old_num_oracle_calls = 0
        self.config.experience_replay_size = self.config.experience_replay_size
        self.experience = ExperienceReplay(max_size=self.config.experience_replay_size * 10, device=self.device)
        self.c_neg = self.config.c_neg
        self.config.mutation_size = self.config.mutation_size if self.config.use_mutation else 0
        self.max_oracle_calls = self.config.max_oracle_calls
        if self.config.vocab_mask:
            with open("tokenizer/smiles_minimal.json", "r") as f:
                tokenizer_data = json.load(f)
            allowed_tokens = set(tokenizer_data["model"]["vocab"].keys())
            self.vocab_mask = torch.tensor([i in allowed_tokens for i in range(len(self.model.tokenizer))], device=self.device).unsqueeze(0)

        else:
            self.vocab_mask = None
        self.prev_novelty = 1

    def set_c_neg(self, novelty):
        """
        Set the negative sample weight value based on the number of valid new sequences.
        Also controls the stop counter.
        """
        self.global_step += 1
        wandb.log({"train/c_neg": self.c_neg, "num_oracle_calls": len(self.mol_buffer.values())}, step=self.global_step)
        if novelty >= 0.5*(1-self.config.start_t):
            self.c_neg = self.config.c_neg
            self.stop_counter -= 1
            self.stop_counter = max(0, self.stop_counter)
        elif novelty > 0.1 and self.c_neg > 0:
            self.c_neg = 0
            self.config.c_neg *= 0.9
        elif novelty < 0.1:
            self.c_neg = 0
            self.config.c_neg *= 0.9
            self.stop_counter += 1
        self.global_step += 1
        wandb.log({"train/c_neg": self.c_neg, "train/stop_counter": self.stop_counter, "num_oracle_calls": len(self.mol_buffer.values())}, step=self.global_step)

    def run_ga(self, max_tries=1000):
        """
        Run the mol-opt molecule mutations
        """

        start = len(self.ga_scores)
        new_smiles = len(self.train_ids)
        tries = 0
        i_=0
        for iteration in range(self.config.mutation_size):
            score = None
            i_ = 0

            while i_ < max_tries:
                # Select parent based on use_prescreen mode
                if self.config.use_prescreen:
                    selected_parent = sorted(self.mol_buffer.items(), key=lambda x: x[1][0], reverse=True)[i_ : i_ + 1]
                else:
                    selected_items = sorted(self.mol_buffer.items(), key=lambda x: x[1][0], reverse=True)
                    selected_idx = i_ % len(selected_items) if selected_items else 0
                    selected_parent = [selected_items[selected_idx]] if selected_items else []

                if not selected_parent:
                    break

                smi, tries_temp = reproduce(selected_parent, 1)
                smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                tries += tries_temp + i_
                if smi and smi not in self.mol_buffer:
                    score = self.oracle(smi)

                    if score is not None:
                        self.ga_scores.append(score)
                        self.qed_ga.append(QED.qed(Chem.MolFromSmiles(smi)))
                        self.sa_ga.append(sascorer.calculateScore(Chem.MolFromSmiles(smi)))

                        if (score > sorted(self.mol_buffer.values(), key=lambda x: x[0], reverse=True)[min(len(self.mol_buffer.values()) - 1, 9)][0]
                            and self.config.train_mutation):
                            new_seq = torch.tensor(self.tokenizer.encode(" ".join(decompose_smiles(smi))))
                            self.train_ids.append(new_seq)
                            self.train_scores.append(score)
                            self.train_x_0.append(torch.randint(4, 203, (len(new_seq),)))
                            self.prompter.update_with_score(smi, score)
                        new_smiles += 1
                        break

                i_ += 1

            if new_smiles >= self.config.offspring_size:
                break
        self.used_mutation = True
        if len(self.ga_scores[start:]) > 0:
            self.global_step += 1
            wandb.log(
                {
                    "ga/num_generated": len(self.ga_scores) - start,
                    "ga/best_ga_scores": np.max(self.ga_scores[start:]),
                    "num_oracle_calls": len(self.mol_buffer.values()),
                    "ga/tries": tries/max_tries,
                    "ga/qed": np.mean(self.qed_ga[start:]),
                    "ga/sa": np.mean(self.sa_ga[start:]),
                },
                step=self.global_step,
            )

    def evaluate_offspring(self, valid_new_seqs, valid_new_x_0, valid_new_smiles):
        """
        Evaluate the offspring sequences.
        """
        for i, (seq, x_0, smi) in enumerate(zip(valid_new_seqs, valid_new_x_0, valid_new_smiles)):
            seq = torch.tensor(seq)
            qed = QED.qed(Chem.MolFromSmiles(smi))
            sa = sascorer.calculateScore(Chem.MolFromSmiles(smi))
            self.qed.append(qed)
            self.sa.append(sa)
            score = self.oracle(smi)
            self.scores.append(score)
            if score == 1 and self.config.no_stop:
                self.config.num_reinforce_steps = 0
            if self.config.use_prompter:
                if score>0:

                    self.prompter.update_with_score(smi, score)  # type: ignore[attr-defined]
                    self.prompter.bandit.update(len(seq[seq != self.model.pad_token_id]), score)
            else:
                self.bandit.update(len(seq[seq != self.model.pad_token_id]), score)
            self.train_ids.append(seq)
            self.train_scores.append(score)  # *qed if (qed>0.8 and sa<3) else score*(qed*0.5)*(min(0,4-sa))
            self.train_x_0.append(x_0)

            if len(self.train_ids) >= self.config.offspring_size - self.config.mutation_size * (not self.used_mutation and self.config.train_mutation):
                break
        return


    def add_experience(self):
        """
        Add the experience to the experience replay buffer.
        """
        new_experience = []
        for i in range(len(self.train_ids)):
            seq = self.train_ids[i]
            uni_ = self.train_x_0[i]
            valid_len = (seq != self.model.pad_token_id).sum()
            seq_trimmed, uni_trimmed = seq[:valid_len], uni_[:valid_len]
            new_experience.append((seq_trimmed, self.train_scores[i], uni_trimmed))
        self.experience.add_experience(new_experience)

    def pad_seqs(self, ids):
        """
        Pad the sequences to the same length.
        """
        return torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=self.model.pad_token_id).to(self.device)

    def construct_rollout_ds(self, ids, scores, uni):
        """
        Construct the rollout dataset.
        Note that the logprobs needs to be precomputed because we cant evaluate the whole trajectory at once like in a LLM
        """
        ids = self.pad_seqs([torch.tensor(i) for i in ids]).to(self.device)
        uni = self.pad_seqs([torch.tensor(u) if self.config.no_sample_uni else torch.randint(4, 203, (len(u),), device=self.device) for u in uni]).to(self.device)
        scores = torch.tensor(scores, device=self.device)
        t_roll = torch.rand(len(ids), device=self.device) * (1 - self.config.start_t) + self.config.start_t if self.config.use_prompter else torch.rand(len(ids), device=self.device)
        # Pre-compute old logprobs AND masks for each timestep
        with torch.no_grad():
            old_logprobs = []
            x_ts_list = []
            source_masks_list = []
            for i in range(self.config.num_reinforce_steps):
                current_t = (t_roll/10 + i * (1.0 - self.config.start_t) / self.config.num_reinforce_steps) % 1
                old_lp, _, x_t, source_mask = self.compute_logprobs(ids, uni, current_t, prior=False)
                old_logprobs.append(old_lp.detach().unsqueeze(1))
                x_ts_list.append(x_t.unsqueeze(1))
                source_masks_list.append(source_mask.unsqueeze(1).clone())
            old_logprobs = torch.cat(old_logprobs, dim=1)
            x_ts_all = torch.cat(x_ts_list, dim=1)
            source_masks_all = torch.cat(source_masks_list, dim=1)
        return TensorDataset(ids, scores, uni, t_roll, old_logprobs, x_ts_all, source_masks_all)

    def construct_loader(self, ids, scores, uni):
        """
        Construct the loader for the PPO training.
        """

        scores = torch.tensor(scores, device=self.device)
        rollout_ds = self.construct_rollout_ds(ids, scores, uni)

        if len(self.experience) >= self.config.experience_replay_size and self.config.experience_replay_size > 0:
            exp_seqs, exp_scores, exp_uni = self.experience.sample(self.config.experience_replay_size)
            exp_ds = self.construct_rollout_ds(exp_seqs, exp_scores, exp_uni)
            train_ds = ConcatDataset([rollout_ds, exp_ds])
        else:
            train_ds = rollout_ds
        loader = DataLoader(
            train_ds, batch_size=min(self.config.offspring_size + self.config.experience_replay_size, 128), shuffle=True, collate_fn=pad_collate_with_masks, drop_last=False
        )  # Need to update this too
        return loader

    def compute_logprobs(self, ids, uni, t, prior=False, x_t=None, source_mask=None):
        """Compute log probabilities under the prior model"""
        B, L = ids.size()
        mask = ids != self.model.pad_token_id
        attn = (~mask).unsqueeze(1).expand(B, L, L).float()
        attn = attn.masked_fill(attn.bool(), float("-inf")).unsqueeze(1)

        # Only sample new mask if not provided
        if x_t is None or source_mask is None:
            x_t, source_mask = sample_path(t, uni, ids)
        else:
            x_t = x_t.clone()
        source_mask = source_mask & mask
        x_t[~mask] = self.model.pad_token_id

        # Get logits
        with torch.no_grad() if prior else nullcontext():
            logits = self.prior.model(x=x_t, t=t, attn_mask=attn, return_hidden=False) if prior else self.model.model(x=x_t, t=t, attn_mask=attn, return_hidden=False)
            logprobs_ = compute_seq_logp(logits, ids, source_mask.float())

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        step_entropy = dist.entropy() * source_mask.float()
        mask_sum = source_mask.float().sum(dim=-1)
        seq_entropy = step_entropy.sum(dim=1) / torch.clamp(mask_sum, min=1.0)

        return logprobs_, seq_entropy, x_t, source_mask

    def reinforce(self, ids, scores, uni):
        """PPO-style reinforce with (optional) experience replay."""
        self.prior.eval()
        self.model.train()

        loader = self.construct_loader(ids, scores, uni)
        for i in range(self.config.num_reinforce_steps):
            for batch_ids, batch_scores, batch_uni, batch_t, batch_old_lp, batch_x_ts, batch_masks in loader:  #
                self.model_opt.zero_grad()
                self.lr_scheduler.step()

                # Extract pre-computed values for this epoch
                batch_old_lp_i = batch_old_lp[:, i]
                batch_x_t_i = batch_x_ts[:, i]
                batch_mask_i = batch_masks[:, i]
                current_t = (batch_t + i * (1.0 - self.config.start_t) / self.config.num_reinforce_steps) % 1.0

                # Compute new logprobs with SAME mask
                agent_lp, entropy, _, _ = self.compute_logprobs(batch_ids, batch_uni, current_t, prior=False, x_t=batch_x_t_i, source_mask=batch_mask_i)
                ratio = torch.exp((agent_lp - batch_old_lp_i))
                # More robust advantage calculation
                score_std = batch_scores.std()
                if batch_scores.numel() > 1 and not score_std < 1e-8:
                    adv = (batch_scores - batch_scores.mean()) / (score_std + 1e-8)
                else:
                    adv = torch.zeros_like(batch_scores)
                #reduce weight of negative samples
                adv = torch.where(adv > 0, adv, self.c_neg * adv)#.clamp(min=-1, max=1)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv
            pg_loss = -torch.min(surr1, surr2).mean()
            loss = pg_loss - 0.01 * entropy.mean()

            if torch.isnan(loss) or torch.isinf(loss) or pg_loss.sum() == 0:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2)
            self.model_opt.step()
            self.global_step += 1

            wandb.log(
                {
                    "reinforce/loss": loss.item(),
                    "reinforce/pg_loss": pg_loss.item(),
                    "reinforce/ratio": ratio.mean().item(),
                    "reinforce/reward": batch_scores.mean().item(),
                    "reinforce/adv": adv.mean().item(),
                    "reinforce/grad_norm": torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None]).norm().item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                    "reinforce/entropy": entropy.mean().item(),
                    "num_oracle_calls": len(self.mol_buffer.values()),
                },
                step=self.global_step,
            )

    def initialize_population(self):
        """
        Initialize population by sampling from the pretrained model.

        Returns:
            List of (sequence, score, smile) triples
        """
        B = int(self.config.first_pop_size * 1.1)  # generate more samples to ensure we have enough valid ones

        n_oracle = [67 if self.config.oracle=="valsartan_smarts" and not self.config.use_prescreen else self.bandit.select_length() for _ in range(int(B))] #

        samples, init_ids = self.model.sample(num_samples=B, temperature=1.0 if self.prev_novelty>0.5 else 1.5, noise=0.0, oracle=n_oracle, eta=1, return_uni=True, vocab_mask=self.vocab_mask,)
        valid, smiles, _ = evaluate_smiles(
            generated_ids=samples,
            tokenizer=self.tokenizer,
            return_values=True,
            print_flag=False,
            print_metrics=False,
            exclude_salts=False,
        )

        # keep only the valid ones
        samples = [torch.tensor(samples[i]) for i in valid][: self.config.first_pop_size]
        init_ids = [torch.tensor(init_ids[i]) for i in valid][: self.config.first_pop_size]
        n_oracle = [n_oracle[i] for i in valid][: self.config.first_pop_size]
        smiles = [smiles[i] for i in valid][: self.config.first_pop_size]
        scores = self.oracle(smiles)

        if self.config.use_prescreen:
            # 1. Load ZINC SMILES
            df = pd.read_csv("in_virtuo_reinforce/vocab/zinc250k.csv").sort_values(by=self.config.oracle, ascending=False)[: self.config.first_pop_size]

            print(f"Loaded {len(smiles)} ZINC SMILES")
            samples = [torch.tensor(self.tokenizer.encode(" ".join(decompose_smiles(sm, max_frags=self.config.max_frags)))) for sm in smiles] + samples
            smiles = df["smiles"].values.tolist() + smiles
            scores = df[self.config.oracle].values.tolist() + scores
            assert len(samples) == len(smiles)
            init_ids = [torch.randint(4, 203, (len(s),)) for s in samples] + init_ids
        self.scores.extend(scores)
        self.qed.extend([QED.qed(Chem.MolFromSmiles(smi)) for smi in smiles])
        self.sa.extend([sascorer.calculateScore(Chem.MolFromSmiles(smi)) for smi in smiles])


        for i in range(len(samples)):
            if scores[i]>0:
                if self.config.use_prompter:

                    self.prompter.update_with_score(smiles[i], scores[i])
                    self.prompter.bandit.update(len(samples[i][samples[i] != self.model.pad_token_id]), scores[i])
                else:
                    self.bandit.update(len(samples[i][samples[i] != self.model.pad_token_id]), scores[i])
        self.train_ids = [s[s != self.model.pad_token_id] for s in samples]
        self.train_x_0 = [i[: len(s)] for i, s in zip(init_ids, self.train_ids)]
        self.train_scores = scores
        if max(self.train_scores)>0 and len(self.train_ids)>=self.config.first_pop_size//2:
            self.reinforce([item for item in self.train_ids], [item for item in self.train_scores], [item for item in self.train_x_0])
            self.add_experience()
        self.prior.model = copy.deepcopy(self.model.model)
        self.train_ids, self.train_scores, self.train_x_0 = [], [], []

    def generate_offspring_batch(self):
        n_oracle = []
        num_samples = min(5000,int((self.config.offspring_size-len(self.train_ids)) // max(self.prev_validity,0.01)))
        if (self.config.use_prompter and  max(self.scores)>0): #or self.config.use_prescreen:
            self.prompter.offspring_size = num_samples
            prompts, n_oracle = self.prompter.build_prompts_and_masks(dev=self.device)


        else:
            prompts = None
            for i in range(num_samples):
                n_oracle.append(67 if self.config.oracle.find("smarts") != -1 and not self.config.use_prescreen else self.bandit.select_length()) #self.b andit.select_length()
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            all_seqs, all_x_0 = self.model.sample(
                prompt=prompts,
                num_samples=num_samples,
                temperature=self.config.temperature,
                noise=1 if self.config.use_prescreen else 0,
                oracle=n_oracle,
                start_t=self.config.start_t ,
                fade_prompt=False,
                dt=self.config.dt,
                temperature_scaling=False,
                eta=1,
                return_uni=True,
                vocab_mask=self.vocab_mask,
            )
        valid, all_smiles, metrics = evaluate_smiles(all_seqs, self.tokenizer, exclude_salts=True, return_values=True, print_flag=False, print_metrics=False, return_unique_indices=True)

        valid_new_smiles, valid_new_seqs, valid_new_x_0 = filter_valid_new(valid, all_smiles, all_seqs, all_x_0, self.mol_buffer)
        novelty = len(valid_new_seqs) / len(all_smiles)
        self.prev_novelty = novelty
        valid_new_smiles, valid_new_seqs, valid_new_x_0 = valid_new_smiles[:self.config.offspring_size], valid_new_seqs[:self.config.offspring_size], valid_new_x_0[:self.config.offspring_size]

        self.prev_validity = min(len(valid_new_seqs) / self.config.offspring_size, 1)

        self.global_step += 1
        wandb.log(
            {
                "smiles/novelty": novelty,
                "smiles/validity": metrics["validity"],
                "smiles/uniqueness": metrics["uniqueness"],
                "smiles/diversity": metrics["diversity"],
                "smiles/quality": metrics["quality"],
                "smiles/qed": metrics["qed"],
                "smiles/sa": metrics["sa"],
                "num_oracle_calls": len(self.mol_buffer.values()),
            },
            step=self.global_step,
        )
        self.evaluate_offspring(valid_new_seqs, valid_new_x_0, valid_new_smiles)
        if self.config.use_mutation and not self.used_mutation and len(self.train_ids) < self.config.offspring_size:
            self.run_ga()
        self.set_c_neg(novelty)


    def evolve_population(self):
        self.old_num_oracle_calls = len(self.mol_buffer.values())
        self.generate_offspring_batch()
        if len(self.train_ids) >= self.config.offspring_size and self.config.num_reinforce_steps > 0:
            assert len(self.train_ids) == self.config.offspring_size, f"len(self.train_ids) = {len(self.train_ids)}"
            self.used_mutation = False
            self.reinforce([item for item in self.train_ids], [item for item in self.train_scores], [item for item in self.train_x_0])  # type: ignore[attr-defined]

            new_experience = []
            for i in range(len(self.train_ids)):
                seq = self.train_ids[i]
                uni_ = self.train_x_0[i]
                valid_len = (seq != self.model.pad_token_id).sum()
                seq_trimmed, uni_trimmed = seq[:valid_len], uni_[:valid_len]
                new_experience.append((seq_trimmed, self.train_scores[i], uni_trimmed))
            self.train_ids, self.train_scores, self.train_x_0 = [], [], []

            self.experience.add_experience(new_experience)
        elif self.config.num_reinforce_steps == 0:
            print("resetting train_ids, train_scores, train_x_0")
            self.train_ids, self.train_scores, self.train_x_0 = [], [], []

        best_score = sorted(self.mol_buffer.values(), key=lambda x: x[0], reverse=True)[0][0]

        self.global_step += 1

        wandb.log({"best_score": best_score, "num_oracle_calls": len(self.mol_buffer.values())}, step=self.global_step)

    def _optimize(self, oracle, config):
        """
        config is expected to be a dict with keys:
          - batch_size: how many candidates to propose per oracle call
        """
        self.oracle.assign_evaluator(oracle)
        self.model.gen_batch_size = self.config.offspring_size * 2
        print("Initializing population...")
        self.initialize_population()
        while not self.finish:
            self.evolve_population()
            if (sum([x[0] for x in sorted(list(self.mol_buffer.values()), reverse=True)[:10]]) >= 9.99) or self.stop_counter == 10:
                if self.config.no_stop:
                    self.config.num_reinforce_steps = 0
                    self.config.offspring_size = 500
                else:
                    print("STOPPING")

                    for i in range(self.max_oracle_calls - len(self.mol_buffer)):
                        self.mol_buffer["finished" + str(i)] = [0, len(self.mol_buffer) + i]


            if len(self.mol_buffer) > 0:
                smis_, true_scores_ = [], []
                for smi, score in list(self.mol_buffer.items()):
                    smis_.append(smi)
                    true_scores_.append(score[0])
                self.global_step += 1
                visualize_top_smiles(smis_, scores=true_scores_, target=self.config.target, oracle_name=self.config.oracle, device=self.device)  # type: ignore[attr-defined]
                if self.config.use_prompter:
                    sorted_vocab = sorted(self.prompter.vocab.items(), key=lambda x: x[1], reverse=True)[:10]
                    smiles = [bridge_smiles_fragments(custom_decode_sequence(self.tokenizer, list(x[0])).split(" ")) for x in sorted_vocab]
                    scores = [x[1] for x in sorted_vocab]
                    visualize_top_smiles(smiles, scores=scores, target=self.config.target, oracle_name=self.config.oracle, device=self.device, prefix="vocab")
                if not self.config.use_prompter:
                    bandit = self.bandit
                else:
                    bandit = self.prompter.bandit
                bandit.plot_distribution(save_path=f"plots/tdc/{self.config.oracle}/length_bandit_{self.device}.png", log_wandb=True, global_step=self.global_step, smiles_list=smis_)

                try:
                    self.global_step += 1
                    wandb.log({"train/auc": wandb.Image("plots/tdc/" + self.config.oracle + "/tdc_avg_top10" + str(self.device) + ".png")}, step=self.global_step)
                except:
                    traceback.print_exc()
                    print("wandb error")
        # final logging & save
        self.log_intermediate(finish=True)
        self.save_result_tot(self.config.oracle)
        self.save_qed_sa(self.config.oracle)
        wandb.finish()
if __name__ == "__main__":

    oracles = [
        "albuterol_similarity",
        "amlodipine_mpo",
        "celecoxib_rediscovery",
        "deco_hop",
        "drd2",
        "fexofenadine_mpo",
        "gsk3b",
        "isomers_c7h8n2o2",
        "isomers_c9h10n2o2pf2cl",
        "jnk3",
        "median1",
        "median2",
        "mestranol_similarity",
        "osimertinib_mpo",
        "perindopril_mpo",
        "qed",
        "ranolazine_mpo",
        "scaffold_hop",
        "sitagliptin_mpo",
        "thiothixene_rediscovery",
        "troglitazone_rediscovery",
        "valsartan_smarts",
        "zaleplon_mpo",
    ]
    targets = {
        "albuterol_similarity": "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",
        "amlodipine_mpo": "CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C)COCCN",
        "celecoxib_rediscovery": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
        "fexofenadine_mpo": "CC(C)(C)OC(=O)N1CCC(CC1)C2=CC=CC=C2C(C3=CC=CC=C3)OCC(=O)O",
        "osimertinib_mpo": "CN[C@H]1CN(C[C@H]1C2=NC=CC(=C2)C#N)C3=CC(=CC=C3)NC4=CC=CC=C4",
        "perindopril_mpo": "CC(C)C[C@H](C(=O)N1CCCC1C(=O)O)NC(CC2=CC=CC=C2)C(=O)O",
        "ranolazine_mpo": "CC1=CC=C(C=C1)C2=CC=CC=C2C(=O)NCCN(CC)CC",
        "sitagliptin_mpo": "CC1=CN(C2=CN=C(N=C2C1)N3CCN(CC3)C(=O)C4CC4)C",
        "zaleplon_mpo": "CC(=O)N1C=CN=C1C2=CC=CC=N2C",
        "thiothixene_rediscovery": "CN(C)CCCN1C2=CC=CC=C2SC3=CC=CC=C13",
        "troglitazone_rediscovery": "CC1=C(C=C(C=C1)O)C(CC2=CC(=C(C=C2)O)OCC(=O)C)C3=CC=CC=C3",
        "mestranol_similarity": "C#C[C@]1(CC[C@H]2[C@@H]3CCC4=CC(=O)CCC4=C3CC[C@H]12)OC",
        "valsartan_smarts": "CC(C)C[C@@H](C(=O)N1CCC[C@H]1C(=O)O)NC(Cc2ccccc2)C(=O)O",
    }
    # for oracle in oracles:
    # try:
    import argparse
    difficult_ones = ["valsartan_smarts", "thiothixene_rediscovery", "scaffold_hop", "troglitazone_rediscovery","sitagliptin_mpo"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/invirtuo_gen.ckpt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--oracle", type=str, default="albuterol_similarity")
    parser.add_argument("--start_t", type=float, default=0.0)
    # parser.add_argument("--mutation_rate", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.99)
    parser.add_argument("--rl_lr", type=float, default=1e-4)

    parser.add_argument("--offspring_size", type=int, default=64)
    parser.add_argument("--num_reinforce_steps", type=int, default=10)
    parser.add_argument("--vocab_mask", action="store_true")
    parser.add_argument("--lr", default=0.1, type=float)
    # parser.add_argument("--use_reward_model", action="store_true")
    parser.add_argument("--big", action="store_true")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_oracle_calls", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--use_prompter", action="store_true")
    parser.add_argument("--clip_eps", type=float, default=0.5)
    parser.add_argument("--c_neg", type=float, default=0.2)
    parser.add_argument("--use_mutation", action="store_true")
    parser.add_argument("--reverse_kl", action="store_true")
    parser.add_argument("--start_task", type=int, default=0)
    parser.add_argument("--mutation_size", type=int, default=10)
    parser.add_argument("--experience_replay_size", type=int, default=24)
    parser.add_argument("--no_sample_uni", action="store_true")
    parser.add_argument("--train_mutation", action="store_true")
    parser.add_argument("--use_prescreen", action="store_true")
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--identifier", type=str, default="")
    parser.add_argument("--end_task", type=int, default=len(oracles))
    parser.add_argument("--first_pop_size", type=int, default=50)
    parser.add_argument("--no_stop", action="store_true")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--max_frags", type=int, default=5)
    parser.add_argument("--difficult_ones", action="store_true")
    # Then in your config setup, add:
    args = parser.parse_args()
    device = f"cuda:{args.device}" if len(args.device)==1 else args.device
    # Convert args to dict for easy passing
    config_dict = dict(vars(args))
    config_dict["device"] = device
    config_dict.pop("start_task")
    config_dict.pop("end_task")
    num_seeds=config_dict.pop("num_seeds")
    # Now you can simply pass all arguments
    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, f"target_property_optimization")
    import datetime

    identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if not args.identifier else args.identifier
    output_dir += "/" + identifier
    print("Output dir: ", output_dir)
    difficult_ones_flag = config_dict.pop("difficult_ones")
    for seed in range(args.start_seed, num_seeds):
        for oracle in oracles[args.start_task : args.end_task]:
            try:
                if difficult_ones_flag and oracle not in difficult_ones:
                    continue
                lightning.seed_everything(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                config_dict["seed"] = seed
                config_dict["target"] = targets.get(oracle, "")
                config_dict["oracle"] = oracle
                config_dict["identifier"] = identifier
                config_dict["output_dir"] = output_dir
                config_dict["task"] = oracle

                print("Target: ", config_dict["target"])
                print("Oracle: ", oracle)
                lr = args.lr if oracle != "isomers_c7h8n2o2" else 0.2
                beta = 0.99 if oracle != "isomers_c7h8n2o2" else 0.95

                optimizer = InVirtuoFMOptimizer(
                    **config_dict,
                )

                optimizer.optimize(oracle=oracle, seed=seed)
                del optimizer
                import gc

                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                traceback.print_exc()
                continue
