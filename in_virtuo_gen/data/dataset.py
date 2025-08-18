import logging
import os
import random

import numpy as np
import psutil
import torch
from torch.utils.data import Dataset


class MultiSourceDataSet(Dataset):
    """
    A PyTorch Dataset for loading and sampling from multiple data sources with support for
    distributed training and undersampling. Maintains sampling ratios between datasets
    and handles dynamic updating of available samples.
    """

    def __init__(
        self,
        data_path,
        prop_path=None,
        chunk_size=1024,
        undersample_rate=1,
        initial_state=None,
        rank=0,
        world_size=1,
        batch_size=128,
        **kwargs,
    ):
        """
        Args:
            data_path (list[str]): Paths to data files.
            prop_path (list[str] or None): Paths to property files (same length as data_path).
            chunk_size (int): (Currently unused) Could be used for chunking large files.
            undersample_rate (int): Stride used for undersampling. 1 means no skipping.
            initial_state (dict or None): State dictionary for resuming sampling (e.g. from a checkpoint).
            rank (int): Current process rank, for distributed training.
            world_size (int): Total number of processes (GPUs) for distributed training.
            batch_size (int): Used to ensure the total samples allocated per rank is multiple of this.
        """
        super().__init__()
        random.shuffle(data_path)
        print(f"Data path: {data_path}")
        self.data_path = data_path
        self.prop_path = prop_path if prop_path else [None] * len(data_path)
        self.chunk_size = chunk_size
        self.undersample_rate = int(undersample_rate)
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size

        # Initialize tracking variables
        self.datasets = []
        self.total_size = 0  # total undersampled size across all datasets
        self.total_remaining = 0
        self.remaining_per_dataset = []
        self.sampling_probs = []

        # 1. Load and initialize each dataset
        self._initialize_datasets()

        # 2. Calculate a *fixed* samples_per_rank, ensuring it's divisible by batch_size
        base_samples = self.total_size // self.world_size
        self.samples_per_rank = (base_samples // self.batch_size) * self.batch_size

        # 3. Restore state if provided
        if initial_state and "dataset_states" in initial_state:
            self._restore_state(initial_state)

        # 4. Initialize sampling probabilities and counters (once)
        self._update_state_fixed()

        # 5. Log initialization info on rank 0
        if rank == 0:
            self._log_initialization_info()

    def _initialize_datasets(self):
        """Initialize all datasets from provided paths with detailed logging."""

        if self.rank == 0:
            print(f"\nInitializing Datasets with Undersampling:")
            print(f"Undersample rate: {self.undersample_rate}")

        original_total = 0
        for d_path, p_path in zip(self.data_path, self.prop_path):
            try:
                data = np.load(
                    d_path,
                    mmap_mode="r",
                )
                props = np.load(p_path, mmap_mode="r") if p_path else None

                full_length = len(data)
                original_total += full_length

                # Compute how many samples remain after undersampling
                available_samples = full_length // self.undersample_rate
                self.total_size += available_samples

                if self.rank == 0:
                    print(f"\nDataset: {d_path}")
                    print(f"Original size: {full_length:,}")
                    print(f"After undersampling: {available_samples:,}")
                    print(f"Reduction: {(1 - available_samples / full_length) * 100:.2f}%")

                self.datasets.append(
                    {
                        "data": data,
                        "props": props,
                        "size": available_samples,
                        "used_samples": 0,
                        "available": True,
                        "path": d_path,
                        # Indices we will actually use, spaced by 'undersample_rate'
                        "stride_indices": np.arange(0, full_length, self.undersample_rate),
                    }
                )

            except Exception as e:
                logging.error(f"Error loading dataset {d_path}: {str(e)}")
                raise

        if self.rank == 0:
            print(f"\nTotal Statistics:")
            print(f"Original total samples: {original_total:,}")
            print(f"After undersampling: {self.total_size:,}")
            reduction = (1 - self.total_size / original_total) * 100 if original_total > 0 else 0
            print(f"Overall reduction: {reduction:.2f}%")
            print(f"Samples per rank (before batch-size rounding): {self.total_size // self.world_size:,}\n")

    def _update_state_fixed(self):
        """
        Update 'remaining' counts and sampling probabilities.
        We do NOT adjust self.samples_per_rank here to avoid mid-epoch changes,
        but we do recompute which datasets are still available and in what proportion.
        """
        self.remaining_per_dataset = []
        total_remaining = 0

        # Calculate how many samples remain in each dataset
        for ds in self.datasets:
            remaining = ds["size"] - ds["used_samples"]
            remaining = max(remaining, 0)
            self.remaining_per_dataset.append(remaining)
            ds["available"] = bool(remaining > 0)
            total_remaining += remaining

        self.total_remaining = total_remaining

        # Recompute sampling probabilities
        self.sampling_probs = []
        if total_remaining > 0:
            for rem in self.remaining_per_dataset:
                self.sampling_probs.append(rem / total_remaining)
        else:
            self.sampling_probs = [0.0] * len(self.datasets)

    def select_dataset(self):
        """
        Select a dataset index based on sampling probabilities.
        Raises:
            IndexError if no data is available in any dataset.
        """
        if sum(self.sampling_probs) == 0:
            raise IndexError("No more data available in any dataset")
        index = np.random.choice(len(self.datasets), p=self.sampling_probs)
        return index

    def __len__(self):
        """
        Return the number of samples allocated to this rank, in a batch-aligned manner.
        This remains constant throughout one epoch.
        """
        return self.samples_per_rank

    def __getitem__(self, idx):
        """
        Get a single sample.
        We call `_update_state_fixed()` more frequently so that sampling probabilities
        reflect the real-time depletion of datasets.
        """
        if idx >= self.samples_per_rank:
            raise StopIteration(f"Index {idx} exceeds available samples ({self.samples_per_rank})")

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                dataset_idx = self.select_dataset()
                dataset = self.datasets[dataset_idx]

                if dataset["used_samples"] >= len(dataset["stride_indices"]):
                    # This dataset is effectively exhausted
                    self._update_state_fixed()
                    continue

                local_idx = dataset["stride_indices"][dataset["used_samples"]]

                data = dataset["data"][local_idx]
                dataset["used_samples"] += 1

                # (Optional) Immediately re-check the state so probabilities reflect used_samples
                self._update_state_fixed()

                # Prepare the tensors
                input_tensor = torch.tensor(data[:-1], dtype=torch.long)
                target_tensor = torch.tensor(data[1:], dtype=torch.long)

                # If there are property data
                if dataset.get("props") is not None:
                    prop_tensor = torch.tensor(dataset["props"][local_idx], dtype=torch.float)
                    return input_tensor, target_tensor, prop_tensor

                # Fix the typo here:
                return input_tensor, target_tensor

            except Exception as e:
                if attempt == max_attempts - 1:
                    logging.error(f"Failed to get item after {max_attempts} attempts. Error: {str(e)}")
                    raise
                self._update_state_fixed()
                continue

    def shuffle_and_reset_epoch(self):
        """
        Shuffle the stride_indices of each dataset, reset the used_samples counter,
        and then update the internal state so sampling probabilities reflect the new order.
        Call this at the start of every new epoch.
        """
        for ds in self.datasets:
            # Shuffle the indices in-place
            np.random.shuffle(ds["stride_indices"])
            # Reset usage to 0 so we start from the beginning of the (now) shuffled indices
            ds["used_samples"] = 0

        # Recompute sampling probabilities and total_remaining based on reset usage
        self._update_state_fixed()

    def load_state_dict(self, state):
        """
        Restore dataset state from a checkpoint-like dictionary.
        """
        dataset_states = state["dataset_states"]
        for ds, info in zip(self.datasets, dataset_states):
            ds["used_samples"] = info["used_samples"]
            ds["size"] = info["size"]
            ds["available"] = info["available"]
            ds["stride_indices"] = info["stride_indices"]

    def get_state(self):
        """
        Return current state for checkpointing.
        """
        return {
            "dataset_states": [
                {
                    "used_samples": ds["used_samples"],
                    "size": ds["size"],
                    "available": ds["available"],
                    "stride_indices": ds["stride_indices"],
                }
                for ds in self.datasets
            ],
            "total_remaining": self.total_remaining,
            "total_size": self.total_size,
            "sampling_probs": self.sampling_probs,
        }

    def get_dataset_info(self):
        """
        Retrieve info about the datasets, such as usage, remaining samples, etc.
        """
        return {
            "total_size": self.total_size,
            "total_remaining": self.total_remaining,
            "remaining_per_dataset": self.remaining_per_dataset,
            "samples_per_rank": self.samples_per_rank,
            "used_samples": [ds["used_samples"] for ds in self.datasets],
            "sampling_probabilities": self.sampling_probs,
            "available_datasets": [ds["available"] for ds in self.datasets],
        }

    def _log_initialization_info(self):
        """
        Log the initialization stats on rank 0.
        """
        logging.info("\nDataset Information:")
        logging.info(f"Total size across all datasets: {self.total_size}")
        logging.info(f"Total remaining samples: {self.total_remaining}")
        logging.info(f"Samples per rank (batch-aligned): {self.samples_per_rank}")

        for i, ds in enumerate(self.datasets):
            remaining = ds["size"] - ds["used_samples"]
            prob = self.sampling_probs[i] if i < len(self.sampling_probs) else 0
            logging.info(f"Dataset {i} ({ds['path']}): " f"{remaining}/{ds['size']} samples remaining " f"({prob * 100:.2f}% sampling probability)")
