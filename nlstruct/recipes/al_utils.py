import os
import string
from typing import Dict

import pytorch_lightning as pl
import torch
from IPython import get_ipython
from rich_logger import RichTableLogger

from nlstruct import BRATDataset, MetricsCollection, get_instance, get_config, InformationExtractor
from nlstruct.checkpoint import ModelCheckpoint, AlreadyRunningException

from carbontracker.tracker import CarbonTracker
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

shared_cache = {}

BASE_WORD_REGEX = r'(?:[\w]+(?:[’\'])?)|[!"#$%&\'’\(\)*+,-./:;<=>?@\[\]^_`{|}~]'
BASE_SENTENCE_REGEX = r"((?:\s*\n)+\s*|(?:(?<=[\w0-9]{2,}\.|[)]\.)\s+))(?=[[:upper:]]|•|\n)"
ENTITIES = ['AdministrationRoute','Anatomy','Aspect','Assertion','BiologicalProcessOrFunction','Chemicals_Drugs','Concept_Idea','Devices','Disorder','Dosage','DrugForm','Genes_Proteins','Hospital','LivingBeings','Localization','Measurement','MedicalProcedure','Persons','SignOrSymptom','Strength','Temporal']

class EmissionMonitoringCallback(pl.Callback):
    def __init__(self, num_train_epochs):    
        self.ctr = 0
        self.num_train_epochs = num_train_epochs
        self.tracker = None
    def on_train_epoch_start(self, trainer, pl_module):
        if self.ctr==1:
            self.tracker = CarbonTracker(epochs=self.num_train_epochs, epochs_before_pred=2, monitor_epochs=9)
        if self.ctr>0:
            self.tracker.epoch_start()
    def on_train_epoch_end(self, trainer, pl_module):
        if self.ctr>0:
            self.tracker.epoch_end()
        self.ctr+=1
    def on_train_end(self, trainer, pl_module):
        if self.ctr<self.num_train_epochs-1 and self.tracker is not None:
             self.tracker.stop()

#class ManagingConfidenceMeasuresCallback(pl.Callback):
#    def on_validation_epoch_end(self, trainer, pl_module):
#        with open('collector.csv','a+') as f:
#            f.write('\n')
#    def on_train_end(self,trainer,pl_module):
#        with open('collector.csv','a+') as f:
#            f.write('=====\n')
        
def classic_build_model_dataset_and_metrics(
      dataset: Dict[str, str],
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
    if isinstance(dataset, dict):
        dataset = BRATDataset(
            **dataset,
        )
        word_regex = BASE_WORD_REGEX
        sentence_split_regex = BASE_SENTENCE_REGEX
        metrics = {
            "exact": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1., word_regex=word_regex),
        } 
    else:
        raise Exception("dataset must be a dict or a str")

    if "filter_predictions" not in predict_kwargs and filter_predictions is not False:
        predict_kwargs["filter_predictions"] = filter_predictions

    if unique_label:
        for split in (dataset.train_data, dataset.val_data, dataset.test_data):
            if split:
                for doc in split:
                    for e in doc["entities"]:
                        e["label"] = "main"
    

    #Model creation
    model = InformationExtractor(
        seed=seed,
        preprocessor=dict(
            module="ner_preprocessor",
            bert_name=bert_name,  # transformer name
            bert_lower=bert_lower,
            split_into_multiple_samples=True,
            sentence_split_regex=sentence_split_regex,  # regex to use to split sentences (must not contain consuming patterns)
            sentence_balance_chars=(),  # try to avoid splitting between parentheses
            sentence_entity_overlap="split",  # raise when an entity spans more than one sentence
            word_regex=word_regex,  # regex to use to extract words (will be aligned with bert tokens), leave to None to use wordpieces as is
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
                    "entity_label": dict(module="vocabulary", values=sorted(dataset.labels()), with_unk=True, with_pad=False),
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
        metrics=metrics,
    ).train()

    model.encoder.encoders[0].cache = shared_cache
    os.makedirs("checkpoints", exist_ok=True)
    
    return model, dataset, metrics
