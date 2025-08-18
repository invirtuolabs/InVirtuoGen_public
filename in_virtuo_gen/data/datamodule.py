import glob
from typing import List, Optional

import datasets
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import  PreTrainedTokenizerFast


from .bucketsampler import BucketDataset, BucketSampler, DistributedBucketSampler
from .dataset import MultiSourceDataSet
from ..utils.mol import compute_single_property
from ..utils.fragments import bridge_smiles_fragments

from ..preprocess.preprocess_tokenize import custom_decode_sequence, tokenize_function


def class_collate_fn(batch, tokenizer):
    """
    Collate function for the DataLoader that tokenizes and pads a batch for classification.
    """
    frags = [item['frags'] for item in batch]
    labels_list = [item['labels'] for item in batch]

    # # Tokenize the batch (list of sequences).
    input_ids = tokenize_function(frags, tokenizer, max_length=190)

    pad_id = tokenizer.encode('[PAD]')[0]
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id) # type: ignore[attr-defined]
    labels_tensor = torch.tensor(labels_list)

    return {'input_ids': padded_input_ids, 'labels': labels_tensor}
def split_frags(fragments):
    return [frag for frag in fragments.split() if frag]
def cond_collate_fn(batch,tokenizer):
    """
    Collate function to pad 'input_ids' (and optionally 'labels')
    to the max length in the batch.
    """
    qualities = [
    torch.tensor([prop])
    for ex in batch
    if (decoded := custom_decode_sequence(tokenizer=tokenizer, encoded_ids=ex)) is not None
    and (smiles := bridge_smiles_fragments(split_frags(decoded), print_flag=True)) is not None
    and (prop := compute_single_property(smiles)) is not None
]
    # Pad input_ids
    mu= torch.tensor([3,0.7]) #Ugly hack, looked at distribution of qed and sa in training set to nromalize
    stds = torch.tensor([1,0.3])
    out = {"input_ids": batch, "conds": (torch.cat(qualities)-mu)/stds}
    return out


### Example usage


class IVGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        val_data_path: str,
        prop_path: Optional[str] = None,
        valid_prop_path: Optional[str] = None,
        n_conds : Optional[int] = None,
        batch_size: int = 256,
        val_batch_size: int = 128,  # Added val_batch_size
        chunk_size: int = 1024,
        undersample_rate: int = 1,
        num_workers: int = 40,
        bucket: bool = True,
        buffer_size: int = 100000,
        ddp=False,
        world_size=1,
        safe=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        # These will be set in `setup()`
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.global_rank = 0
        self.world_size = 1
        self.safe = safe

    def setup(self, stage: Optional[str] = None):
        """
        Called by Lightning with a 'stage' identifier to set up datasets.
        """
        if stage == "fit":
            # self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.hparams.tokenizer_path)
            if not self.hparams.bucket: # type: ignore[attr-defined]
                # Instantiate your custom dataset for TRAIN
                self.train_dataset = MultiSourceDataSet(
                    data_path=self.hparams.data_path, # type: ignore[attr-defined]
                    prop_path=self.hparams.prop_path, # type: ignore[attr-defined]
                    undersample_rate=self.hparams.undersample_rate, # type: ignore[attr-defined]
                    batch_size=self.hparams.batch_size, # type: ignore[attr-defined]
                )
                self.val_dataset = MultiSourceDataSet(
                    data_path=self.hparams.val_data_path, # type: ignore[attr-defined]
                    prop_path=self.hparams.prop_path, # type: ignore[attr-defined]
                    undersample_rate=self.hparams.undersample_rate, # type: ignore[attr-defined]
                    batch_size=self.hparams.val_batch_size,  # Use val_batch_size # type: ignore[attr-defined]
                )
            else:
                val_data_path = "{}/bucket_*val.npy".format(self.hparams.val_data_path)  #  type: ignore[attr-defined]
                if len(glob.glob(val_data_path)) == 0:
                    val_data_path = "{}/bucket_*.npy".format(self.hparams.val_data_path) # type: ignore[attr-defined]
                data_path = "{}/bucket_*0.npy".format(self.hparams.data_path) # type: ignore[attr-defined] # type: ignore[attr-defined]
                self.val_sampler = BucketSampler(bucket_file_paths=glob.glob(val_data_path), batch_size=self.hparams.val_batch_size, tokens_per_batch=self.hparams.val_batch_size * 80, seed=42, safe=self.hparams.safe) # type: ignore[attr-defined]
                self.sampler = BucketSampler(bucket_file_paths=glob.glob(data_path), batch_size=self.hparams.batch_size, tokens_per_batch=self.hparams.batch_size * 80, seed=42, safe=self.hparams.safe) # type: ignore[attr-defined]

                self.train_dataset = BucketDataset(self.sampler, safe=self.hparams.safe) # type: ignore[attr-defined]
                self.val_dataset = BucketDataset(self.val_sampler, safe=self.hparams.safe) # type: ignore[attr-defined]



    def train_dataloader(self):
        """
        Return the DataLoader for training.
        Note: shuffle=False is typical when your Dataset
              has its own sampling logic built-in.
        """

        self.train_dl = StatefulDataLoader(
            self.train_dataset,# type: ignore[attr-defined]
            batch_size=self.hparams.batch_size if not self.hparams.bucket else None, # type: ignore[attr-defined]
            num_workers=self.hparams.num_workers, # type: ignore[attr-defined]
            persistent_workers=False,
            shuffle=False,  # The dataset internally manages sampling
            pin_memory=False,
            collate_fn=lambda x: cond_collate_fn(x,self.tokenizer) if self.hparams.n_conds else x # type: ignore[attr-defined]
        )
        return self.train_dl

    def val_dataloader(self):
        """
        Return the DataLoader for validation, if you have a val dataset.
        """
        if self.val_dataset is not None:
            self.val_dl = StatefulDataLoader(
                self.val_dataset,
                batch_size=self.hparams.val_batch_size if not self.hparams.bucket else None,  # Use val_batch_size # type: ignore[attr-defined]
                num_workers=self.hparams.num_workers, # type: ignore[attr-defined]
                persistent_workers=False,
                shuffle=False,
                pin_memory=False,
                collate_fn=lambda x: cond_collate_fn(x,self.tokenizer) if self.hparams.n_conds else x # type: ignore[attr-defined]
            )
            return self.val_dl
        return None

    def test_dataloader(self):
        """
        Return the DataLoader for testing, if you have a test dataset.
        """
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size, # type: ignore[attr-defined]
                num_workers=self.hparams.num_workers, # type: ignore[attr-defined]
                shuffle=False,
                collate_fn=lambda x: cond_collate_fn(x,self.tokenizer) if self.hparams.n_conds else x # type: ignore[attr-defined]
            )
        return None

    def predict_dataloader(self):
        """
        During prediction no dataloader is needed
        """
        return DataLoader(torch.zeros(1), batch_size=1) # type: ignore[attr-defined]

    def state_dict(self):
        return {"sampler": self.sampler.state_dict() if self.sampler else None, "val_sampler": self.val_sampler.state_dict() if self.val_sampler else None}

    def load_state_dict(self, state_dict):

        if self.sampler and state_dict.get("sampler"):
            self.sampler.load_state_dict(state_dict["sampler"])
        if self.val_sampler and state_dict.get("val_sampler"):
            self.val_sampler.load_state_dict(state_dict["val_sampler"])
