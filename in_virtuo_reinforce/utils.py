
import random

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from in_virtuo_gen.utils.fragments import bridge_smiles_fragments, order_fragments_by_attachment_points, smiles2frags
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn
import os
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import wandb
from in_virtuo_gen.preprocess.preprocess_tokenize import custom_decode_sequence
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import re
import random
import torch
from collections import defaultdict


def count_attachment_points(fragment: str) -> int:
    """Count the number of attachment points [I*] in a fragment."""
    attachment_pattern = r"\[\d+\*\]"
    return len(re.findall(attachment_pattern, fragment))


class PromptTimeDataset(Dataset):
    """
    Wraps a base dataset of (ids, prompts) to emit, for each
    pair, n_steps different t values in [start_t, 1).
    """

    def __init__(self, base_dataset, start_t: float, dt: float):
        self.base = base_dataset
        self.start_t = start_t

        self.dt = dt
        # number of diffusion-time samples per example
        self.n_steps = int((1.0 - start_t) / dt)
        assert self.n_steps > 0, "dt too large or start_t >= 1"
        self.total_len = len(self.base) * self.n_steps

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # which base example
        sample_idx = idx // self.n_steps
        # which timestep for this example
        step_idx = idx % self.n_steps
        # compute t in [start_t, 1)
        t = self.start_t + step_idx * self.dt + random.uniform(0, self.dt)
        ids, scores = self.base[sample_idx]
        return ids, scores, torch.tensor(t, dtype=torch.float)


def make_dataloader(base_dataset, start_t, dt, batch_size, **dl_kwargs):
    ds = PromptTimeDataset(base_dataset, start_t, dt)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, **dl_kwargs)


def visualize_top_smiles(smiles_list, top_n=10, target="CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O", prompts=False, pairs=False, scores=None, oracle_name="", device=0, prefix=""):
    """
    Plot e top N generated SMILES by mean score.

    Parameters:
    - smiles_list: list of generated SMILES
    - top_n: number of top SMILES to display
    - target: target SMILES to display first
    """
    os.makedirs(f"plots/tdc/{oracle_name}", exist_ok=True)

    mols = []
    legends = []
    # Add target first
    target_mol = Chem.MolFromSmiles(target)
    if target_mol:
        mols.append(target_mol)
        legends.append("Target\n" + target)
    # Then each fragment
    for score, smi in zip(scores, smiles_list[:top_n]):  # type: ignore[attr-defined]
        m = Chem.MolFromSmiles(smi.replace(" ", ""))
        if m:
            mols.append(m)
            legends.append("Reward: %.2f" % score + "\n" + smi)  # "i=%d" % score[1] +
    from rdkit.Chem.Draw import rdMolDraw2D
    # 4) Draw grid (1 + top_n molecules)
    #    Adjust molsPerRow to fit nicely (e.g. 5 per row)
    opts = rdMolDraw2D.MolDrawOptions()
    opts.legendFontSize = 25  # Set legend font size here
    opts.atomLabelFontSize = 14
    opts.bondLineWidth = 2
    img = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=5, subImgSize=(300, 300), drawOptions=opts)
    try:
        img.save(f"plots/tdc/{oracle_name}/top_smiles_{device}.pdf" if not prefix else f"plots/tdc/{oracle_name}/{prefix}_{device}.pdf", format="pdf")
    except:
        pass
    plt.close()



def decompose_smiles(smi, max_frags=5, sort=False):
    try:
        frags = order_fragments_by_attachment_points(smiles2frags(smi, max_frags=max_frags)) if sort else smiles2frags(smi, max_frags=max_frags)
    except:
        frags = smiles2frags(smi, max_frags=max_frags)
        import traceback

        traceback.print_exc()
    return frags


def sort_frags(frags, tokenizer, ids=True, return_frags=False):
    if ids:
        frags = [decode(f, tokenizer) for f in frags]

    frags = order_fragments_by_attachment_points(frags)
    if return_frags:
        return [tokenizer.encode(f) for f in frags]
    frags = tokenizer.encode(" ".join(frags))

    return frags


def randomize_smiles(smiles: str, random_seed=None) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    atoms = list(mol.GetAtoms())  # type: ignore[attr-defined]
    idx = list(range(len(atoms)))
    random.seed(random_seed)
    random.shuffle(idx)
    mol = Chem.RenumberAtoms(mol, idx)
    return Chem.MolToSmiles(mol, canonical=False)


def augment_fragments(frags, num_augmentations=1):

    attempt_count = 0
    augmented_results = []
    while len(augmented_results) < num_augmentations and attempt_count < (100):
        try:
            if len(frags) > 1:
                if num_augmentations > 1:
                    smiles = bridge_smiles_fragments(frags)
                    smiles = randomize_smiles(smiles)  # type: ignore[attr-defined]
                frags = smiles2frags(smiles, max_frags=5, canonical=True)
                random.shuffle(frags)
                new_smiles = Chem.CanonSmiles(bridge_smiles_fragments(frags))

                augmented_results.append(" ".join(frags))
        except Exception as e:
            print(f"[augment_fragments] Fragment error for  {str(e)}")
        attempt_count += 1

    return augmented_results[:num_augmentations]


def decode(ids, tokenizer):
    seq = [tokenizer.decode(id) for id in ids]
    return "".join(seq)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.query = nn.Parameter(torch.ones(hidden_dim) * 1 / 50)  # context vector
        self.out = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, H, mask=None):
        """
        H: Tensor of shape (batch, seq_len, hidden_dim)
        mask: Optional BoolTensor of shape (batch, seq_len) where True=valid
        """

        z = self.norm(H) * mask.unsqueeze(-1)  # type: ignore[attr-defined]

        return self.out(torch.cat((z.sum(1) / mask.float().sum(1, keepdim=True), (mask.float().sum(1, keepdim=True) - 50) / 5), dim=-1))  # , alphas # type: ignore[attr-defined]


class ExperienceReplay:
    """Prioritized experience replay for highest scored sequences"""

    def __init__(self, max_size=1000, device="cuda"):
        self.memory = []
        self.max_size = max_size
        self.device = device

    def add_experience(self, experience):
        """Add new experiences to memory
        Args:
            experience: list of (sequence, score, prior_logprob) tuples
        """
        self.memory.extend(experience)

        # Remove duplicates based on sequence
        seen = set()
        unique_memory = []
        for exp in self.memory:
            seq_tuple = tuple(exp[0].tolist()) if torch.is_tensor(exp[0]) else tuple(exp[0])
            if seq_tuple not in seen:
                seen.add(seq_tuple)
                unique_memory.append(exp)
        self.memory = unique_memory

        # Keep only top scoring experiences
        if len(self.memory) > self.max_size:
            self.memory.sort(key=lambda x: x[1], reverse=True)
            self.memory = self.memory[: self.max_size]

    def sample(self, n):
        """Sample n experiences with probability proportional to score"""

        # Compute sampling probabilities based on scores
        ranks = np.arange(len(self.memory), dtype=np.float64)+1

        # 3) Compute weights ∝ 1 / (κ·N + rank)
        denom = 0.001 * len(self.memory) + ranks
        weights = 1.0 / denom
        probs = weights / weights.sum()
        # probs = scores / scores.sum()

        # Sample without replacement
        indices = np.random.choice(len(self.memory), size=min(n, len(self.memory)), replace=False, p=probs)
        sampled = [self.memory[i] for i in indices]

        sequences = [exp[0] for exp in sampled]
        scores = np.array([exp[1] for exp in sampled])
        # prior_logprobs = np.array([exp[2] for exp in sampled])
        unis = [exp[2] for exp in sampled]
        return sequences, scores, unis

    def __len__(self):
        return len(self.memory)


import numpy as np
from typing import Dict, List


class SoftmaxBandit:
    def __init__(
        self,
        prior_probs,  # initial prior, sums to 1
        lengths,  # list/array of actual lengths
        lr=0.1,  # Q-learning rate
        beta=0.95,  # how strongly to stick to old prior
    ):
        self.prior = np.array(prior_probs, dtype=float)
        assert np.all(self.prior >= 0) and abs(self.prior.sum() - 1.0) < 1e-6

        self.lengths = np.array(lengths)
        self.index_of = {l: i for i, l in enumerate(self.lengths)}
        self.max_length = self.lengths.max()
        self.lr = lr
        self.beta = beta
        self.Q = np.zeros_like(self.prior)

        self._history = []

    def _compute_probs(self):
        logits = np.log(self.prior + 1e-8) + (self.Q )
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def select_length(self):
        p = self._compute_probs()
        arm = np.random.choice(len(p), p=p)
        self._history.append(p.copy())
        return int(self.lengths[arm])  # +random.randint(-2,2)

    def update(self, length, reward):
        # 1) Q‐update
        length = min(length, self.max_length)
        arm = self.index_of[length]
        self.Q[arm] += self.lr * (reward - self.Q[arm])

        # 2) compute new posterior p
        p = self._compute_probs()

        # 3) blend into prior
        self.prior = self.beta * self.prior + (1 - self.beta) * p
        self.prior /= self.prior.sum()  # renormalize

    def current_probs(self):
        return self._compute_probs()

    def plot_distribution(self, save_path=None, log_wandb=False, global_step=0, smiles_list=None):

        p_cur = self.current_probs()
        x = self.lengths
        w = 0.5

        pastel_colors = ["#AEC6CF", "#FFB347"]  # light blue and peach

        fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=False)

        # Prior plot

        lengths = [len(s) for s in smiles_list]
        axs[0].hist(lengths, bins=max(lengths)-min(lengths))
        axs[0].set_title("Length Distribution")
        axs[0].set_xlabel("Sequence length")
        axs[0].set_ylabel("Frequency")


        # Current plot
        axs[1].bar(x, p_cur, color=pastel_colors[1], edgecolor="gray", width=w)
        axs[1].set_title(f"Current π ")
        axs[1].set_xlabel("Sequence length")

        # Overall adjustments
        for ax in axs[1:]:
            ax.set_ylim(0, max(max(self.prior), max(p_cur)) * 1.1)  # set same y-limits for easy comparison
            ax.grid(axis="y", linestyle="--", alpha=0.7)

        # plt.suptitle("Softmax + Prior-Update Bandit")
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore[attr-defined]
        if log_wandb:
            global_step += 1
            wandb.log({"bandit_distribution": wandb.Image(plt)}, step=global_step)  # type: ignore[attr-defined]
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def valid_new(valid, smiles, off_seqs, prompts, mol_buffer, offspring_size):
    valid = [i for i in valid if smiles[i] not in mol_buffer][:offspring_size]
    off_seqs = [off_seqs[i] for i in valid][:offspring_size]
    prompts = [prompts[i] for i in valid][:offspring_size]
    smiles = [smiles[i] for i in valid][:offspring_size]

    return valid, smiles, off_seqs, prompts


class GeneticPrompter:
    """
    Modular prompt builder: selects parent fragments, constructs crossover prompts,
    builds a vocab mask, and maintains/upates its own fragment vocabularies.
    """

    def __init__(
        self,
        tokenizer,
        bandit,
        offspring_size: int = 2,
        kappa: float = 0.001,
        always_ok: Any = None,
        max_frags: int = 5,
        K: int = 2,
        pad_id: int = 3,
        score_based=False,
        vocab_size=10,
    ):
        self.tokenizer = tokenizer
        self.bandit = bandit
        self.offspring_size = offspring_size
        self.kappa = kappa
        self.always_ok = list(always_ok) if always_ok is not None else []
        self.max_frags = max_frags
        self.pad_id = pad_id
        self.K = K
        # internal population for selection
        self.vocab = {}
        self.vocab_fps = {}
        self.population = []
        self.close = False
        self.min_tanimoto_dist = 0.7
        self.score_based = score_based
        self.vocab_size = vocab_size
        # vocab of fragments (with attachment points)

    def update_with_score(self, smiles: str, score: float) -> None:
        """
        If just_update=False:
            (your normal logic, not shown here)
        If just_update=True:
            1) Find the single existing vocab entry whose fingerprint has the highest
               Tanimoto similarity to the new molecule (but only consider sims >= threshold).
            2) If none found → do nothing.
            3) If found and new score > old_score → delete that old entry and insert the new one.
               Otherwise → do nothing.
        """
        if smiles is None:
            print("similes is None")
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return

        fp_new = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)  # type: ignore[attr-defined]
        threshold = 1.0 - self.min_tanimoto_dist
        fragments = decompose_smiles(smiles, 5, sort=False)
        frag_key = tuple(self.tokenizer.encode(" ".join(fragments)))

        highest_sim = 0
        highest_sim_key = None
        highest_sim_score = 0

        for key, (fp_old, old_score) in self.vocab_fps.items():
            sim = DataStructs.TanimotoSimilarity(fp_new, fp_old)
            if sim > highest_sim:
                highest_sim = sim
                highest_sim_key = key
                highest_sim_score = old_score

        if highest_sim > threshold:
            if score > highest_sim_score:
                del self.vocab[highest_sim_key]
                del self.vocab_fps[highest_sim_key]
                self.vocab[frag_key] = score
                self.vocab_fps[frag_key] = (fp_new, score)
            return  # Exit early - just replace if similar molecule exists

        self.vocab[frag_key] = score
        self.vocab_fps[frag_key] = (fp_new, score)

        # prune vocab to top‐K, and keep fps in sync
        self._prune(self.vocab)
        keep = set(self.vocab)
        for k in list(self.vocab_fps):
            if k not in keep:
                del self.vocab_fps[k]

    def pad_seqs(self, seqs: List[torch.Tensor]) -> torch.Tensor:
        """Pad list of 1D tensors to same length using pad token."""
        if len(seqs) > 0:
            return torch.nn.utils.rnn.pad_sequence(
                seqs,
                batch_first=True,
                padding_value=self.pad_id,
            )
        else:
            return torch.tensor([])

    def _prune(self, vocab_scores: Dict[Tuple[int, ...], float]) -> None:
        """Keep only top-K items by score."""
        if len(vocab_scores) > self.vocab_size:
            # sort by descending score and keep top K keys
            topk = sorted(vocab_scores.items(), key=lambda x: x[1], reverse=True)[: self.vocab_size]
            # rebuild dict
            vocab_scores.clear()
            vocab_scores.update({k: v for k, v in topk})

    def build_prompts_and_masks(self, dev) -> Tuple[Any, List[List[int]]]:
        """
        Selects parents, builds crossover prompts, and returns:
          - prompts: LongTensor [B, Lmax]
          - raw_prompts: List of token-ID lists (for updating bandits/vocab)
        """
        pad_id = self.pad_id
        B = self.offspring_size
        V = len(self.tokenizer)

        # select parent pairs from internal population
        p1_list, p2_list, n_oracle = [], [], []
        for _ in range(self.offspring_size):


            i1, i2 = w(self.vocab, 2, kappa=self.kappa)
            i1 = torch.tensor(i1)
            i2 = torch.tensor(i2)
            p1_list.append(i1)
            p2_list.append(i2)
            n_oracle.append(self.bandit.select_length())

        P1 = self.pad_seqs(p1_list).to(dev)
        P2 = self.pad_seqs(p2_list).to(dev)

        prompt_tensors: List[torch.Tensor] = []
        # n_oracle = []
        for b in range(B):
            ids1 = P1[b][P1[b] != pad_id]
            ids2 = P2[b][P2[b] != pad_id]
            toks = self._fragment_prompter(ids1, ids2, n_oracle[b])
            prompt_tensors.append(torch.tensor(toks, device=dev))
            # n_oracle.append(self.bandit.select_length())
        # pad and sort
        prompts = self.pad_seqs(prompt_tensors).long()

        return prompts, n_oracle

    def _fragment_prompter(self, p1_ids: torch.Tensor, p2_ids: torch.Tensor, n_oracle: int) -> List[int]:
        """
        Fragment-level crossover to flat token-ID list.
        """
        smi1 = bridge_smiles_fragments(custom_decode_sequence(self.tokenizer, p1_ids).split())
        smi2 = bridge_smiles_fragments(custom_decode_sequence(self.tokenizer, p2_ids).split())
        fr1 = decompose_smiles(smi1, self.max_frags, sort=True)
        fr2 = decompose_smiles(smi2, self.max_frags, sort=True)
        if self.close:
            frags = fr1
        else:
            frags = fr1[:-self.K] + fr2[-self.K:] if len(fr1) < 7 else fr1[:-2] + fr2[-2:]
        random.shuffle(frags)  # if not self.close else None
        frags += fr1[-self.K:] + fr2[:-self.K] if len(fr1) < 7 else fr1[:-2] + fr2[-2:]
        kept = " ".join(frags)  # +" "
        return self.tokenizer.encode(kept)[:n_oracle]


def w(smiles_scores: Dict[str, float], n_select: int, kappa: float = 0.1) -> List[str]:
    """
    Rank‐based sampling without replacement over a smiles→score dict.

    Args:
      smiles_scores: mapping from SMILES string to its numeric score
      n_select:      how many SMILES to pick
      kappa:         small constant to control flatness of the distribution

    Returns:
      A list of selected SMILES (length = n_select or fewer if dict is smaller).
    """
    keys = list(smiles_scores.keys())
    N = len(keys)
    if n_select >= N:
        return keys[0],keys[0]  # return all if asking for too many

    # 1) collect scores in same order as keys
    scores = np.array([smiles_scores[s] for s in keys], dtype=np.float64)

    # 2) sort descending, get indices
    sorted_idx = np.argsort(-scores)  # best‐score first

    # 3) assign ranks (best rank=1, second=2, ..., worst=N)
    ranks = np.empty(N, dtype=np.int64)

    ranks[sorted_idx] = np.arange(0, N)

    # 4) compute weight ∝ 1 / (κ·N + rank)
    denom = kappa * N + ranks
    weights = 1.0 / denom
    weights /= weights.sum()

    # 5) draw without replacement
    chosen_idx = np.random.choice(N, size=2, replace=True, p=weights)
    # return the corresponding SMILES strings
    return [keys[i] for i in chosen_idx]


import numpy as np
from typing import Dict, List


def score_based_selection(smiles_scores: Dict[str, float], n_select: int, kappa: float = 0.001) -> List[str]:
    """
    Rank‐based sampling without replacement over a smiles→score dict.

    Args:
      smiles_scores: mapping from SMILES string to its numeric score
      n_select:      how many SMILES to pick
      kappa:         small constant to control flatness of the distribution

    Returns:
      A list of selected SMILES (length = n_select or fewer if dict is smaller).
    """
    keys = list(smiles_scores.keys())
    N = len(keys)
    if n_select >= N:
        return keys[:]  # return all if asking for too many

    # 1) collect scores in same order as keys
    scores = np.array([smiles_scores[s] for s in keys], dtype=np.float64)
    if scores.min() < 1e-4:
        scores += 1e-6
    scores = scores / scores.std() if len(scores) > 0 and scores.std() > 0 else scores
    # 5) draw without replacement
    chosen_idx = np.random.choice(N, size=n_select, replace=True, p=scores / scores.sum())
    # return the corresponding SMILES strings
    return [keys[i] for i in chosen_idx]