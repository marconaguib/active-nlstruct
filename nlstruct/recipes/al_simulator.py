import gc
import json
import os

import pytorch_lightning as pl
import torch
from rich_logger import RichTableLogger
import pandas as pd
from nlstruct.checkpoint import ModelCheckpoint, AlreadyRunningException
from nlstruct.recipes.al_utils import EmissionMonitoringCallback,ManagingConfidenceMeasuresCallback
from carbontracker.tracker import CarbonTracker
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from random import sample,seed

class AL_Simulator():
    def __init__(
         self,
         model,
         dataset,
         selection_strategy,
         metrics,
         annotiter_size = 2,
         k = 2,
    ):
        self.model = model
        self.dataset = dataset
        self.selection_strategy = selection_strategy
        if selection_strategy == "random":
            seed(42)
        self.metrics = metrics
        self.annotiter_size = annotiter_size
        self.k = k
        if len(dataset.val_data)>0:
            print("Specifying the validation dataset size is useless, it's determined by k in AL_Simulator.")
        self.pool = self.dataset.val_data + self.dataset.train_data
        self.dataset.val_data = []
        self.dataset.train_data = []

    def run_simulation(self, num_iterations, max_steps, xp_name, gpus):
        for _ in range(self.k):
            examples = self.select_examples()
            print(examples)
            self.annotate(examples, to_dev_split=True)
        for i in range(num_iterations):
            examples = self.select_examples()
            print(examples)
            self.run_iteration(examples, max_steps=max_steps, xp_name=xp_name+str(i), gpus=gpus)

    def select_examples(self):
        if self.selection_strategy=="ordered":
            return list(range(self.annotiter_size))
        elif self.selection_strategy=="random":
            return sample(range(len(self.pool)), k=self.annotiter_size)
   
    def run_iteration(self, selected_examples, max_steps, xp_name, gpus):
        self.annotate(selected_examples)
        print("starting iteration with")
        print([d['doc_id'] for d in self.dataset.val_data[:4]], 'as val data, and')
        print([d['doc_id'] for d in self.dataset.train_data[:10]], ', ... as train data')
        self.go(max_steps=max_steps, xp_name=xp_name, gpus=gpus)

    def annotate(self, examples, to_dev_split=False):
        rec = self.dataset.val_data if to_dev_split else self.dataset.train_data
        for e in sorted(examples,reverse=True):
            #print(e, [d["doc_id"] for d in rec], [d["doc_id"] for d in self.pool[:10]])
            rec.append(self.pool.pop(e))
            #print(e, [d["doc_id"] for d in rec], [d["doc_id"] for d in self.pool[:10]])

    def go(
      self,
      max_steps,
      xp_name,
      gpus,
      ):
        print(self.dataset.describe())

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            trainer = pl.Trainer(
                gpus=gpus,
                progress_bar_refresh_rate=1,
                checkpoint_callback=False,  # do not make checkpoints since it slows down the training a lot
                callbacks=[#ModelCheckpoint(path='checkpoints/{hashkey}-{global_step:05d}' if not xp_name else 'checkpoints/' + xp_name + '-{hashkey}-{global_step:05d}'),
                           EmissionMonitoringCallback(num_train_epochs=10),
                           EarlyStopping(monitor="val_exact_f1",mode="max", patience=3),
                           ManagingConfidenceMeasuresCallback()],
                logger=[
                    RichTableLogger(key="epoch", fields={
                        "epoch": {},"step": {},
                        "(.*)_?loss": {"goal": "lower_is_better", "format": "{:.4f}"},
                        "(.*)_(precision|recall|tp)": False,
                        "(.*)_f1": {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_f1"},
                        ".*_lr|max_grad": {"format": "{:.2e}"},
                        "duration": {"format": "{:.0f}", "name": "dur(s)"},
                    }),
                    pl.loggers.TestTubeLogger("logs", name=xp_name if xp_name is not None else "untitled_experience"),
                ],
                val_check_interval=40,
                max_steps=max_steps
            )
            trainer.fit(self.model, datamodule=self.dataset)
            trainer.logger[0].finalize(True)

        except AlreadyRunningException as e:
            model = None
            print("Experiment was already running")
            print(e)
