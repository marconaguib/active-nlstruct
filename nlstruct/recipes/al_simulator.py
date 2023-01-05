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

class AL_Simulator():
    def __init__(
         self,
         model,
         dataset,
         selection_strategy,
         metrics,
    ):
        self.model = model
        self.dataset = dataset
        self.selection_strategy = selection_strategy
        self.metrics = metrics

    def run_simulation(self, num_iterations, max_steps, xp_name, gpus):
        for i in range(num_iterations):
            examples = self.select_examples()
            self.run_iteration(examples, max_steps=max_steps, xp_name=xp_name+str(i), gpus=gpus)

    def select_examples(self):
        return None
   
    def run_iteration(self, selected_examples, max_steps, xp_name, gpus):
        self.annotate(selected_examples, self.dataset)
        self.go(max_steps=max_steps, xp_name=xp_name, gpus=gpus)

    def annotate(self, examples, dataset):
        pass

    def go(
      self,
      max_steps,
      xp_name,
      gpus,
      ):
        print(self.dataset.describe())
        mylogger = RichTableLogger(key="epoch", fields={
            "epoch": {},
            "step": {},
        
            "(.*)_?loss": {"goal": "lower_is_better", "format": "{:.4f}"},
            "(.*)_precision": False,  # {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_p"},
            "(.*)_recall": False,  # {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_r"},
            "(.*)_tp": False,
            "(.*)_f1": {"goal": "higher_is_better", "format": "{:.4f}", "name": r"\1_f1"},
        
            ".*_lr|max_grad": {"format": "{:.2e}"},
            "duration": {"format": "{:.0f}", "name": "dur(s)"},
        })

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        
        #this
        this_xp_name = xp_name
    
        try:
            trainer = pl.Trainer(
                gpus=gpus,
                progress_bar_refresh_rate=1,
                checkpoint_callback=False,  # do not make checkpoints since it slows down the training a lot
                callbacks=[#ModelCheckpoint(path='checkpoints/{hashkey}-{global_step:05d}' if not this_xp_name else 'checkpoints/' + this_xp_name + '-{hashkey}-{global_step:05d}'),
                           EmissionMonitoringCallback(num_train_epochs=10),
                           EarlyStopping(monitor="val_exact_f1",mode="max", patience=3),
                           ManagingConfidenceMeasuresCallback()],
                logger=[
                    mylogger,
                    pl.loggers.TestTubeLogger("logs", name=this_xp_name if this_xp_name is not None else "untitled_experience"),
                ],
                val_check_interval=40,
                max_steps=max_steps
            )
            trainer.fit(self.model, datamodule=self.dataset)
            mylogger.finalize(True)

            result_output_filename = "checkpoints/{}.json".format(trainer.callbacks[0].hashkey)
            if not os.path.exists(result_output_filename):
                if gpus:
                    self.model.cuda()
                if self.dataset.test_data:
                    print("TEST RESULTS:")
                else:
                    print("VALIDATION RESULTS (NO TEST SET):")
                eval_data = self.dataset.test_data if self.dataset.test_data else self.dataset.val_data

                final_metrics = MetricsCollection({
                    **{metric_name: get_instance(metric_config) for metric_name, metric_config in self.metrics.items()},
                })

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
            else:
                with open(result_output_filename, 'r') as json_file:
                    results = json.load(json_file)["results"]
                    print(pd.DataFrame(results).T)
        except AlreadyRunningException as e:
            model = None
            print("Experiment was already running")
            print(e)
