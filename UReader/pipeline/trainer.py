import torch.distributed as dist
import argparse
from functools import partial

import torch

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import Trainer
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from pipeline.utils import batchify
from typing import Union, Optional

class CustomTrainer(Trainer):
    
    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        sampler = DistributedSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self._train_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
            pin_memory=False,
            collate_fn=batchify)


    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        dataset = self.eval_dataset
        sampler = DistributedSampler(dataset, shuffle=False)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self._train_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
            pin_memory=False,
            collate_fn=batchify)