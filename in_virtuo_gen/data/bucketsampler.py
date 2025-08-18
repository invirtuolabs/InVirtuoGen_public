import glob
import math
import os
import random

import numpy as np
import torch
from torch.distributions import Categorical
from torch.utils.data import DataLoader, IterableDataset


class BucketSampler:
    def __init__(self, bucket_file_paths, batch_size, tokens_per_batch, seed=42, safe=False):
        """
        Args:
            bucket_file_paths (list of str): List of paths to your bucket .npy files.
            batch_size (int): How many samples to return per batch.
            tokens_per_batch (int): Target number of tokens per batch.
            seed (int, optional): Random seed for reproducibility.
            safe (bool): Whether to use uint16 dtype instead of uint8.
        """
        self.batch_size = batch_size
        self.safe = safe
        self.tokens_per_batch = tokens_per_batch
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Build a dictionary mapping a bucket label to its info.
        self.buckets = {}
        assert len(bucket_file_paths) > 0, "No bucket files found in {bucket}".format(bucket_file_paths)

        for path in bucket_file_paths:
            base = os.path.basename(path)

            # Expecting file names like "bucket_0-10.npy"
            if base.startswith("bucket_") and base.endswith(".npy"):
                label = base[len("bucket_") : -len(".npy")]  # e.g. "0-10"
            else:
                label = base

            # Get sequence length from the file name
            seq_length = int(path.split(".npy")[0].split("-")[-1].split("_")[0])

            # Calculate dynamic batch size for this bucket
            batch_size = max(1, self.tokens_per_batch // seq_length)

            data = np.memmap(path, mode="r", dtype=np.uint8 if not self.safe else np.uint16)
            data = data.reshape(-1, seq_length)
            num_samples = data.shape[0]
            self.buckets[label] = {
                "path": path,
                "data": data,
                "size": num_samples,
                "pointer": 0,
                "seq_length": seq_length,
                "batch_size": batch_size
            }

        self.bucket_labels = list(self.buckets.keys())

    def set_worker_partition(self, worker_id, num_workers):
        """
        FIXED: Partition the actual data among workers, not just the batches.
        Each worker gets a non-overlapping slice of each bucket.
        """
        for label in self.bucket_labels:
            bucket = self.buckets[label]
            total_size = bucket["size"]

            # Calculate this worker's slice of the bucket
            samples_per_worker = total_size // num_workers
            start_idx = worker_id * samples_per_worker

            if worker_id == num_workers - 1:
                # Last worker gets any remaining samples
                end_idx = total_size
            else:
                end_idx = start_idx + samples_per_worker

            # Update bucket to only contain this worker's slice
            worker_size = end_idx - start_idx
            if worker_size > 0:
                bucket["data"] = bucket["data"][start_idx:end_idx]
                bucket["size"] = worker_size
                bucket["pointer"] = 0  # Reset pointer for this worker's slice
            else:
                # This worker has no data for this bucket
                bucket["size"] = 0
                bucket["pointer"] = 0

    def __iter__(self):
        """
        Yields batches until all buckets are exhausted.
        Each iteration chooses a bucket based on the number of samples remaining.
        """
        while True:
            # Build a list of available buckets with their remaining sample counts.
            available_labels = []
            weights = []
            for label in self.bucket_labels:
                bucket = self.buckets[label]
                remaining_samples = bucket["size"] - bucket["pointer"]
                if remaining_samples > 0:
                    # Weight by remaining tokens, not just samples
                    remaining_tokens = remaining_samples * bucket["seq_length"]
                    available_labels.append(label)
                    weights.append(remaining_tokens)

            if not available_labels:
                break

            # Choose a bucket with probability proportional to its remaining samples.
            chosen_label = random.choices(available_labels, weights=weights, k=1)[0]
            bucket_info = self.buckets[chosen_label]
            ptr = bucket_info["pointer"]
            total = bucket_info["size"]
            batch_size = bucket_info["batch_size"]

            # Take a batch using the dynamic batch size
            end = min(ptr + batch_size, total)
            batch = bucket_info["data"][ptr:end]
            # Update the pointer for the chosen bucket.
            self.buckets[chosen_label]["pointer"] = end

            yield batch

    def __len__(self):
        total_batches = 0
        for label in self.bucket_labels:
            size = self.buckets[label]["size"]
            bucket_batch_size = self.buckets[label]["batch_size"]
            batches = math.ceil(size / bucket_batch_size)
            total_batches += batches
        return total_batches

    def state_dict(self):
        """Returns a dictionary representing the sampler's state."""
        state = {
            "pointers": {label: self.buckets[label]["pointer"] for label in self.bucket_labels},
            "rng_state": random.getstate(),
        }
        return state

    def load_state_dict(self, state):
        """Loads the sampler state from a state dictionary."""
        pointers = state.get("pointers", {})
        for label, pointer in pointers.items():
            if label in self.buckets:
                self.buckets[label]["pointer"] = pointer
        if "rng_state" in state:
            rng_state = state["rng_state"]
            # Recursively convert lists to tuples
            def to_tuple(x):
                if isinstance(x, list):
                    return tuple(to_tuple(item) for item in x)
                else:
                    return x
            rng_state = to_tuple(rng_state)
            random.setstate(rng_state)

    def reset(self):
        """Resets all bucket pointers to zero."""
        for label in self.bucket_labels:
            self.buckets[label]["pointer"] = 0


class DistributedBucketSampler(BucketSampler):
    def __init__(self, bucket_file_paths, batch_size, tokens_per_batch, seed=42, rank=0, world_size=1, drop_last=False):
        super().__init__(bucket_file_paths, batch_size, tokens_per_batch, seed)
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.seed = seed

        # FIXED: Partition data among distributed processes
        self.set_worker_partition(rank, world_size)

    def __iter__(self):
        # Reset pointers and RNG state at the start of each epoch.
        self.reset()
        # FIXED: Each process has its own data partition, so no need for batch filtering
        for batch in super().__iter__():
            yield batch

    def __len__(self):
        # FIXED: Return the actual number of batches for this process's data partition
        return super().__len__()

    def reset(self):
        # Call parent's reset to zero all bucket pointers.
        super().reset()
        # FIXED: Use different seeds for different processes to avoid identical sampling
        process_seed = self.seed + self.rank * 999999  # Large offset to avoid overlap
        random.seed(process_seed)
        np.random.seed(process_seed)


class BucketDataset(IterableDataset):
    def __init__(self, bucket_sampler, safe=False):
        """
        Args:
            bucket_sampler: A BucketSampler or DistributedBucketSampler instance.
            safe (bool): Whether to use uint16 dtype instead of uint8.
        """
        self.safe = safe
        self.bucket_sampler = bucket_sampler

    def __iter__(self):
        # FIXED: Handle worker partitioning properly
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # FIXED: Partition the actual bucket data among workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Create worker-specific seed to ensure different sampling patterns
            worker_seed = hash((self.bucket_sampler.seed if hasattr(self.bucket_sampler, 'seed') else 42, worker_id)) % (2**32)
            random.seed(worker_seed)
            np.random.seed(worker_seed)

            # Partition the bucket data for this worker
            self.bucket_sampler.set_worker_partition(worker_id, num_workers)

        # Reset sampler (now with worker-specific data partitions)
        self.bucket_sampler.reset()

        # Iterate through this worker's portion of the data
        for batch in self.bucket_sampler:
            yield torch.tensor(batch, dtype=torch.uint8 if not self.safe else torch.uint16)

    def __len__(self):
        """Return the number of batches for this worker."""
        return len(self.bucket_sampler)

    def state_dict(self):
        """Save the state of the bucket sampler."""
        return self.bucket_sampler.state_dict()

    def load_state_dict(self, state):
        """Load the state of the bucket sampler."""
        self.bucket_sampler.load_state_dict(state)

