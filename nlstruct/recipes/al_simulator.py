import gc
import json
import os
import string
from typing import Dict

import pytorch_lightning as pl
import torch
from rich_logger import RichTableLogger
import pandas as pd
from nlstruct import BRATDataset, MetricsCollection, get_instance, get_config, InformationExtractor
from nlstruct.checkpoint import ModelCheckpoint, AlreadyRunningException
from nlstruct.recipes.al_utils import EmissionMonitoringCallback
from carbontracker.tracker import CarbonTracker
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from nlstruct.data_utils import sentencize
from random import sample,seed

shared_cache = {}

class AL_Simulator():
    def __init__(
         self,
         dataset_name,
         selection_strategy,
         annotiter_size = 5,
         k = 2,
         al_seed = 42,
         gpus = 1,
         sentencize_pool = True,
         unique_label = False,
         BASE_WORD_REGEX = r'(?:[\w]+(?:[’\'])?)|[!"#$%&\'’\(\)*+,-./:;<=>?@\[\]^_`{|}~]',
         BASE_SENTENCE_REGEX = r"((?:\s*\n)+\s*|(?:(?<=[\w0-9]{2,}\.|[)]\.)\s+))(?=[[:upper:]]|•|\n)",
         *args,
         **kwargs,
    ):
        self.selection_strategy = selection_strategy
        self.annotiter_size = annotiter_size
        self.k = k
        self.unique_label = unique_label
        self.dataset_name = dataset_name
        seed(al_seed)
        
        if not isinstance(dataset_name, dict):
            raise Exception("dataset must be a dict or a str")


        self.dataset = BRATDataset(
            **dataset_name,
        )
        self.word_regex = BASE_WORD_REGEX
        self.sentence_split_regex = BASE_SENTENCE_REGEX
        self.metrics = {"exact": dict(module="dem",binarize_tag_threshold=1., binarize_label_threshold=1., word_regex=self.word_regex),} 

        if unique_label:
            for split in (dataset.train_data, dataset.val_data, dataset.test_data):
                if split:
                    for doc in split:
                        for e in doc["entities"]:
                            e["label"] = "main"
        
        if len(self.dataset.val_data)>0:
            print("Specifying the validation dataset size is useless, it's determined by k in AL_Simulator.")

        all_docs = self.dataset.val_data + self.dataset.train_data
        self.pool = []
        if sentencize_pool:
            for d in all_docs:
                self.pool.extend(sentencize(d, entity_overlap="split"))
        #print(self.pool[0])
        self.dataset.val_data = []
        self.dataset.train_data = []
        self.nb_iter = 0
        self.gpus = gpus
        
        #self.model = self._classic_model_builder(finetune_bert=True, *args, **kwargs)

    def run_simulation(self, num_iterations, max_steps, xp_name):
        for _ in range(self.k):
            examples = self.select_examples(strategy="random")
            self.annotate(examples, to_dev_split=True)
        for _ in range(num_iterations):
            examples = self.select_examples(strategy=self.selection_strategy)
            self.nb_iter += 1
            self.run_iteration(examples, max_steps=max_steps, xp_name=xp_name+'_'+str(self.nb_iter))
        remaining_examples = list(range(len(self.pool)))
        self.nb_iter += len(self.pool)//self.annotiter_size
        self.run_iteration(remaining_examples, max_steps=max_steps, xp_name=xp_name+'_'+str(self.nb_iter))

    def select_examples(self, strategy,min_iter_conf=5):
        if strategy=="ordered":
            return list(range(self.annotiter_size))
        elif strategy=="random" or strategy=="confidence" and self.nb_iter<min_iter_conf:
            print('selecting random predictions')
            return sample(range(len(self.pool)), k=self.annotiter_size)
        elif strategy=="confidence":
            if self.gpus:
                self.model.cuda()
            print('selecting most unconifdent predictions')
            return [x[0] for x in 
                     sorted(enumerate(self.model.predict(self.pool)),
                        key= lambda l : 
                             sum([p['confidence'] for p in l[1]['entities']])/len(l[1]['entities'])
                             if len(l[1]['entities']) else 0,
                        reverse=True
                        )[:self.annotiter_size]
                   ]
   
    def run_iteration(self, selected_examples, max_steps, xp_name):
        self.model = self._classic_model_builder(finetune_bert=True,)# *args, **kwargs)
        self.annotate(selected_examples)
        print("starting iteration with")
        print([d['doc_id'] for d in self.dataset.val_data], 'as val data, and')
        print([d['doc_id'] for d in self.dataset.train_data[:10]], ', ... as train data')
        self.go(max_steps=max_steps, xp_name=xp_name)

    def annotate(self, examples, to_dev_split=False):
        rec = self.dataset.val_data if to_dev_split else self.dataset.train_data
        for e in sorted(examples,reverse=True):
            rec.append(self.pool.pop(e))

    def go(self, max_steps, xp_name):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            trainer = pl.Trainer(
                gpus=self.gpus,
                progress_bar_refresh_rate=1,
                checkpoint_callback=False,  # do not make checkpoints since it slows down the training a lot
                callbacks=[ModelCheckpoint(path='checkpoints/{hashkey}-{global_step:05d}' if not xp_name else 'checkpoints/' + xp_name + '-{hashkey}-{global_step:05d}'),
                           #EmissionMonitoringCallback(num_train_epochs=10),
                           EarlyStopping(monitor="val_exact_f1",mode="max", patience=3),
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
            trainer.fit(self.model, self.dataset)
            trainer.logger[0].finalize(True)
        
            result_output_filename = "checkpoints/{}.json".format(xp_name)
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

   
    def _classic_model_builder(
          self,
          seed: int = 42,
          do_char: bool = True,
          do_biaffine: bool = True,
          do_tagging: str = "full",
          doc_context: bool = True,
          finetune_bert: bool = False,
          bert_lower: bool = False,
          max_tokens: int = 256,
          n_bert_layers: int = 4,
          n_lstm_layers: int = 3,
          biaffine_size: int = 150,
          bert_proj_size: int = None,
          biaffine_loss_weight: float = 1.,
          hidden_size: int = 400,
          val_check_interval: int = None,
          bert_name: str = "camembert/camembert-base",
          fasttext_file: str = "",  # set to "" to disable
          unique_label: int = False,
          norm_bert: bool = False,
          dropout_p: float = 0.1,
          batch_size: int = 16,
          lr: float = 1e-3,
          use_lr_schedules: bool = True,
          word_pooler_mode: str = "mean",
          predict_kwargs: Dict[str, any] = {},
    ):
    
        for name, value in locals().items():
           print(name.ljust(40), value)
    
        filter_predictions = False
        if "filter_predictions" not in predict_kwargs and filter_predictions is not False:
            predict_kwargs["filter_predictions"] = filter_predictions

        model = InformationExtractor(
            seed=seed,
            preprocessor=dict(
                module="ner_preprocessor",
                bert_name=bert_name,  # transformer name
                bert_lower=bert_lower,
                split_into_multiple_samples=True,
                sentence_split_regex=self.sentence_split_regex,  # regex to use to split sentences (must not contain consuming patterns)
                sentence_balance_chars=(),  # try to avoid splitting between parentheses
                sentence_entity_overlap="split",  # raise when an entity spans more than one sentence
                word_regex=self.word_regex,  # regex to use to extract words (will be aligned with bert tokens), leave to None to use wordpieces as is
                substitutions=(),  # Apply these regex substitutions on sentences before tokenizing
                keep_bert_special_tokens=False,
                min_tokens=0,
                doc_context=doc_context,
                join_small_sentence_rate=0.,
                max_tokens=max_tokens,  # split when sentences contain more than 512 tokens
                large_sentences="equal-split",  # for these large sentences, split them in equal sub sentences < 512 tokens
                empty_entities="raise",  # when an entity cannot be mapped to any word, raise
                vocabularies={
                    **{  # vocabularies to use, call .train() before initializing to fill/complete them automatically from training data
                        "entity_label": dict(module="vocabulary", values=sorted(self.dataset.labels()), with_unk=True, with_pad=False),
                    },
                    **({
                           "char": dict(module="vocabulary", values=string.ascii_letters + string.digits + string.punctuation, with_unk=True, with_pad=False),
                       } if do_char else {})
                },
                fragment_label_is_entity_label=True,
                multi_label=False,
                filter_entities=None,  # "entity_type_score_density", "entity_type_score_lesion"),
            ),
            dynamic_preprocessing=False,

            # Text encoders
            encoder=dict(
                module="concat",
                dropout_p=0.5,
                encoders=[
                    dict(
                        module="bert",
                        path=bert_name,
                        n_layers=n_bert_layers,
                        freeze_n_layers=0 if finetune_bert is not False else -1,  # freeze 0 layer (including the first embedding layer)
                        bert_dropout_p=None if finetune_bert else 0.,
                        token_dropout_p=0.,
                        proj_size=bert_proj_size,
                        output_lm_embeds=False,
                        combine_mode="scaled_softmax" if not norm_bert else "softmax",
                        do_norm=norm_bert,
                        do_cache=not finetune_bert,
                        word_pooler=dict(module="pooler", mode=word_pooler_mode),
                    ),
                    *([dict(
                        module="char_cnn",
                        in_channels=8,
                        out_channels=50,
                        kernel_sizes=(3, 4, 5),
                    )] if do_char else []),
                    *([dict(
                        module="word_embeddings",
                        filename=fasttext_file,
                    )] if fasttext_file else [])
                ],
            ),
            decoder=dict(
                module="contiguous_entity_decoder",
                contextualizer=dict(
                    module="lstm",
                    num_layers=n_lstm_layers,
                    gate=dict(module="sigmoid_gate", init_value=0., proj=True),
                    bidirectional=True,
                    hidden_size=hidden_size,
                    dropout_p=0.4,
                    gate_reference="last",
                ),
                span_scorer=dict(
                    module="bitag",
                    do_biaffine=do_biaffine,
                    do_tagging=do_tagging,
                    do_length=False,

                    threshold=0.5,
                    max_fragments_count=200,
                    max_length=40,
                    hidden_size=biaffine_size,
                    allow_overlap=True,
                    dropout_p=dropout_p,
                    tag_loss_weight=1.,
                    biaffine_loss_weight=biaffine_loss_weight,
                    eps=1e-14,
                ),
                intermediate_loss_slice=slice(-1, None),
            ),

            _predict_kwargs=predict_kwargs,
            batch_size=batch_size,

            # Use learning rate schedules (linearly decay with warmup)
            use_lr_schedules=use_lr_schedules,
            warmup_rate=0.1,

            gradient_clip_val=10.,
            _size_factor=0.001,

            # Learning rates
            main_lr=lr,
            fast_lr=lr,
            bert_lr=5e-5,

            # Optimizer, can be class or str
            optimizer_cls="transformers.AdamW",
            metrics=self.metrics,
        ).train()

        model.encoder.encoders[0].cache = shared_cache
        os.makedirs("checkpoints", exist_ok=True)
        
        return model
  
