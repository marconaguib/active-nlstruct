import gc
import json
import os

import pytorch_lightning as pl
import torch
from rich_logger import RichTableLogger
import pandas as pd
from nlstruct import MetricsCollection, get_instance, get_config
from nlstruct.checkpoint import ModelCheckpoint, AlreadyRunningException
from nlstruct.recipes.al_utils import EmissionMonitoringCallback
from carbontracker.tracker import CarbonTracker
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from nlstruct.data_utils import sentencize

from random import sample,seed

class AL_Simulator():
    def __init__(
         self,
         model,
         dataset,
         metrics,
         selection_strategy,
         annotiter_size = 2,
         k = 2,
         al_seed = 42,
         gpus = 1,
    ):
        self.model = model
        self.dataset = dataset
        self.selection_strategy = selection_strategy
        if selection_strategy == "random":
            seed(al_seed)
        self.metrics = metrics
        self.annotiter_size = annotiter_size
        self.k = k
        if len(dataset.val_data)>0:
            print("Specifying the validation dataset size is useless, it's determined by k in AL_Simulator.")
        all_docs = self.dataset.val_data + self.dataset.train_data
        self.pool = []
        for d in all_docs:
            self.pool.extend(sentencize(d, entity_overlap="split"))
        self.dataset.val_data = []
        self.dataset.train_data = []
        self.nb_iter = 0
        self.gpus = gpus
        if self.gpus:
            self.model.cuda()

    def run_simulation(self, num_iterations, max_steps, xp_name):
        for _ in range(self.k):
            examples = self.select_examples(strategy="random")
            self.annotate(examples, to_dev_split=True)
        for i in range(num_iterations):
            examples = self.select_examples(strategy=self.selection_strategy)
            self.nb_iter += 1
            self.run_iteration(examples, max_steps=max_steps, xp_name=xp_name+'_'+str(self.nb_iter))
        remaining_examples = list(range(len(self.pool)))
        self.nb_iter += len(self.pool)//self.annotiter_size
        self.run_iteration(examples, max_steps=max_steps, xp_name=xp_name+'_'+str(self.nb_iter))

    def select_examples(self, strategy):
        print(self.model.device)
        if strategy=="ordered":
            return list(range(self.annotiter_size))
        elif strategy=="random" or strategy=="confidence" and self.nb_iter<2:
            print('selecting random predictions')
            return sample(range(len(self.pool)), k=self.annotiter_size)
        elif strategy=="confidence":
            if self.gpus:
                self.model.cuda()
            print('selecting most unconifdent predictions')
            return [x[0] for x in 
                     sorted(enumerate(self.model.predict(self.pool[:200])),
                        key= lambda l : 
                             sum([p['confidence'] for p in l[1]['entities']])/len(l[1]['entities'])
                             if len(l[1]['entities']) else 0,
                        reverse=True
                        )[:self.annotiter_size]
                   ]
   
    def run_iteration(self, selected_examples, max_steps, xp_name):
        self.annotate(selected_examples)
        print("starting iteration with")
        print([d['doc_id'] for d in self.dataset.val_data[:4]], 'as val data, and')
        print([d['doc_id'] for d in self.dataset.train_data[:10]], ', ... as train data')
        self.go(max_steps=max_steps, xp_name=xp_name)

    def annotate(self, examples, to_dev_split=False):
        rec = self.dataset.val_data if to_dev_split else self.dataset.train_data
        for e in sorted(examples,reverse=True):
            rec.append(self.pool.pop(e))

    def go(
      self,
      max_steps,
      xp_name,
      ):
        print(self.dataset.describe())
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            trainer = pl.Trainer(
                gpus=self.gpus,
                progress_bar_refresh_rate=1,
                checkpoint_callback=False,  # do not make checkpoints since it slows down the training a lot
                callbacks=[#ModelCheckpoint(path='checkpoints/{hashkey}-{global_step:05d}' if not xp_name else 'checkpoints/' + xp_name + '-{hashkey}-{global_step:05d}'),
                           #EmissionMonitoringCallback(num_train_epochs=10),
                           EarlyStopping(monitor="val_exact_f1",mode="max", patience=3),
                           #ManagingConfidenceMeasuresCallback()
                           ],
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
                val_check_interval=max_steps//10,
                max_steps=max_steps
            )
            trainer.fit(self.model, datamodule=self.dataset)
            trainer.logger[0].finalize(True)
        
            result_output_filename = "checkpoints/{}.json".format(xp_name)
            #if self.gpus:
            #    self.model.cuda()
            if self.dataset.test_data:
                print("TEST RESULTS:")
            else:
                print("VALIDATION RESULTS (NO TEST SET):")
            eval_data = self.dataset.test_data if self.dataset.test_data else self.dataset.val_data

            final_metrics = MetricsCollection({
                **{metric_name: get_instance(metric_config) for metric_name, metric_config in self.metrics.items()},
            })
            
            if self.gpus:
                self.model.cuda()

            results = final_metrics(list(self.model.predict(eval_data)), eval_data)
            print(pd.DataFrame(results).T)

            def json_default(o):
                if isinstance(o, slice):
                    return str(o)
                raise

            with open(result_output_filename, 'w') as json_file:
                json.dump({
                    "config": {**get_config(self.model), "max_steps": max_steps},
                    "results": results,
                }, json_file, default=json_default)

        except AlreadyRunningException as e:
            self.model = None
            print("Experiment was already running")
            print(e)
