import gc
import json
import os
import string
from typing import Dict
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
import torch
from rich_logger import RichTableLogger
import pandas as pd
from nlstruct import BRATDataset, MetricsCollection, get_instance, get_config, InformationExtractor
from nlstruct.checkpoint import ModelCheckpoint, AlreadyRunningException
from carbontracker.tracker import CarbonTracker
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from nlstruct.data_utils import sentencize
from nlstruct.data_utils import mappable
import random
from statistics import median as real_median
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter


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
         and_train_on_all_data = False,
         BASE_WORD_REGEX = r'(?:[\w]+(?:[’\'])?)|[!"#$%&\'’\(\)*+,-./:;<=>?@\[\]^_`{|}~]',
         BASE_SENTENCE_REGEX = r"((?:\s*\n)+\s*|(?:(?<=[\w0-9]{2,}\.|[)]\.)\s+))(?=[[:upper:]]|•|\n)",
         entities_to_ignore = [],
         entities_to_remove_from_pool = [],
         debug = False,
         *args,
         **kwargs,
    ):
        self.selection_strategy = selection_strategy
        self.annotiter_size = annotiter_size
        self.k = k
        self.unique_label = unique_label
        self.dataset_name = dataset_name
        self.and_train_on_all_data = and_train_on_all_data
        self.debug = debug
        self.preds = {}
        random.seed(al_seed)
        self.al_seed = al_seed
        self.args = args
        self.kwargs = kwargs
        
        # @mappable
        # def prepare_data(doc):
        #     doc = deepcopy(doc)
        #     for e in doc['entities']:
        #         attribute_labels = []
        #         for f in e['fragments']:
        #             f['label'] = e['label']
        #         attribute_labels.append(e['label'])
        #         for a in e['attributes']:
        #             attribute_labels.append("{}:{}".format(a['label'], a['value']))
        #         e['label'] = attribute_labels
        #     return doc

        @mappable
        def substitution_merlot(doc):
            substitutions = {'Anatomy':'ANAT',
                            'Chemicals_Drugs':'CHEM', 
                            'Disorder':'DISO',
                            'SignOrSymptom':'DISO','Devices':'DEVI',
                            'LivingBeings':'LIVB', 'Persons':'LIVB',
                            'BiologicalProcessOrFunction':'PHEN',
                            'MedicalProcedure':'PROC',
                            'Temporal':'TEMP',
                            'Dosage':'DOSE', 'Strength': 'DOSE', 'AdministrationRoute':'MODE',
                            'DrugForm': 'MODE', 
                            'Measurement':'MEAS' 
                            }
            for e in doc['entities']:
                e['label'] = substitutions.get(e['label'], e['label'])
            return doc
        
        if not isinstance(dataset_name, dict):
            raise Exception("dataset must be a dict or a str")
        self.dataset = BRATDataset(
            **dataset_name,
            #preprocess_fn=prepare_data,
            preprocess_fn=substitution_merlot,
           
        )
        self.word_regex = BASE_WORD_REGEX
        self.sentence_split_regex = BASE_SENTENCE_REGEX
        labels= [l for l in self.dataset.labels() if l not in entities_to_ignore+entities_to_remove_from_pool]
 
        self.metrics = {"exact": dict(module="dem",binarize_tag_threshold=1., binarize_label_threshold=1., word_regex=self.word_regex, 
            add_label_specific_metrics=labels,filter_entities=labels,
        )
        } 
        self.model = self._classic_model_builder(*args,**kwargs)

        if unique_label:
            for split in (self.dataset.train_data, self.dataset.val_data, self.dataset.test_data):
                if split:
                    for doc in split:
                        for e in doc["entities"]:
                            e["label"] = "main"
        
        if len(self.dataset.val_data)>0:
            print("Specifying the validation dataset size is useless, it's determined by k in AL_Simulator.")
       

        # all_docs = self.dataset.val_data + self.dataset.train_data
        all_docs = self.dataset.train_data
        self.pool = []
        if sentencize_pool:
            for d in all_docs:
                sentences = sentencize(d, reg_split=r"(?<=[.|\s])(?:\s+)(?=[A-Z])", entity_overlap="split")
                self.pool.extend([s for s in sentences if len(s['text'])>1])
        else:
            self.pool = all_docs

        if self.debug:
            self.pool = self.pool[:100]
            self.dataset.test_data = self.dataset.test_data[:10]
        
        if len(entities_to_ignore):
            for s in self.pool:
                s['entities'] = [e for e in s['entities'] if e['label'] not in entities_to_ignore]
        if len(entities_to_remove_from_pool):
            self.pool = [s for s in self.pool 
                     if not [e['label'] for e in s['entities']] in [[t] for t in entities_to_remove_from_pool]
                   ]
            
        self.too_similar = set()

        self.dataset.val_data = []
        self.dataset.train_data = []
        self.queue = []
        self.nb_iter = 0
        self.gpus = gpus if not debug else 0
        self.tracker = CarbonTracker(epochs=11, epochs_before_pred=2, monitor_epochs=10)
        self.queue_entry_counter = 0

        # generic scorers
        mean_3 = lambda l:sum(l)/len(l) if len(l)>3 else 0
        maximum = lambda l:max(l) if len(l) else 0
        minimum = lambda l:min(l) if len(l) else 0
        median_3 = lambda l:real_median(l) if len(l)>3 else 0
        pred_scorers = {
            "pred_variety": lambda i:len(set([p['label'] for p in self.preds[i]['entities']])),
            "uncertainty_median_min5": lambda i:median_3([1-p['confidence'] for p in self.preds[i]['entities']]),
            "uncertainty_mean_min3": lambda i:mean_3([1-p['confidence'] for p in self.preds[i]['entities']]),
            "uncertainty_sum": lambda i:sum([1-p['confidence'] for p in self.preds[i]['entities']]),
            "uncertainty_min": lambda i:minimum([1-p['confidence'] for p in self.preds[i]['entities']]),
            "uncertainty_max": lambda i:maximum([1-p['confidence'] for p in self.preds[i]['entities']]),
            "pred_num": lambda i:len(self.preds[i]['entities']),
        }
             
        def sample_diverse_vocab_iterative(size):
            selected = []
            vectorizer = TfidfVectorizer(ngram_range=(1, 2),)
            X = vectorizer.fit_transform([d['text'] for d in self.pool])
            #start with a random doc
            selected.append(random.choice(range(len(self.pool))))
            while len(selected)<size:
                #sort the pool by distance to the selected docs
                dists = pairwise_distances(X[selected], X, metric='cosine').ravel()
                #sum the distances to the selected docs
                dists = np.sum(dists.reshape(len(selected), -1), axis=0)
                assert len(dists)==len(self.pool)
                #select the farthest doc
                selected.append(np.argmax(dists))
            return selected
        
        def sample_diverse_pred_iterative(size):
            selected = []
            pred_dict = self.preds.copy()
            lists = [[e['label'] for e in p['entities']] for p in self.preds.values()]
            number_of_labels = len(set([e['label'] for p in self.pool for e in p['entities']]))
            def subset_multi_coverage(indices_a, list_b, number_of_values):
                    subset = [list_b[i] for i in indices_a]
                    counter = Counter([e for l in subset for e in l])
                    return min(counter.values()) if len(counter) == number_of_values else 0
            while len(selected)<size:
                next_best_doc = max(pred_dict.keys(), key=lambda i:subset_multi_coverage([i]+selected, lists, number_of_labels))
                selected.append(next_best_doc)
                del pred_dict[next_best_doc]
            return selected
        
        def sample_diverse_gold_iterative(size):
            selected = []
            pool_dict = {i:d for i,d in enumerate(self.pool)}
            lists = [[e['label'] for e in d['entities']] for d in self.pool]
            number_of_labels = len(set([e['label'] for d in self.pool for e in d['entities']]))
            def subset_multi_coverage(indices_a, list_b, number_of_values):
                    subset = [list_b[i] for i in indices_a]
                    counter = Counter([e for l in subset for e in l])
                    return min(counter.values()) if len(counter) == number_of_values else 0
            while len(selected)<size:
                next_best_doc = max(pool_dict.keys(), key=lambda i:subset_multi_coverage([i]+selected, lists, number_of_labels))
                selected.append(next_best_doc)
                del pool_dict[next_best_doc]
            return selected
                
        def sample_most_common_vocab(size):
            # get the most common n-grams avoiding french stopwords
            vectorizer = CountVectorizer(ngram_range=(1, 3),)
            X = vectorizer.fit_transform([d['text'] for d in self.pool])
            counts = np.asarray(X.sum(axis=0)).ravel()
            most_common_ngrams = np.argsort(counts)[::-1][:100]
            # for each document, get the number of n-grams that are in the most common n-grams
            X = X[:,most_common_ngrams]
            counts = np.asarray(X.sum(axis=1)).ravel()
            most_common = np.argsort(counts)[::-1][:size]
            return most_common
        
        def sample_most_common_vocab_alternative(size):
            """Selects the documents having the most neighbors within a small radius"""
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform([d['text'] for d in self.pool])
            dists = pairwise_distances(X)
            np.fill_diagonal(dists, np.inf)
            epsilon = 2.5*np.min(dists[dists>0])
            
            neighbours = np.sum(dists<epsilon, axis=1)
            for i in range(len(self.pool)):
                if len(self.pool[i]['text'])<50:
                    neighbours[i] = -1
            #get the document with the most neighbors
            res = []
            for _ in range(size):
                most_common = np.argmax(neighbours)
                if neighbours[most_common]<=0:
                    print("no more neighbours, choosing a larger epsilon")
                    epsilon *= 1.5
                res.append(most_common)
                #too similar are the docs that are within epsilon distance from the most common
                too_sim = [i for i in range(len(self.pool)) if dists[most_common,i]<epsilon]
                neighbours[most_common] = -1
                for i in too_sim:
                    neighbours[i] = -1
            return res
        
        def uncertainty_mean_for_most_diverse_vocab(size):
            if self.nb_iter <= 1 :
                most_diverse = sample_diverse_vocab_iterative(size)
                #print('Too early to count on the model to perform uncertainty sorting for most diverse vocab. Returning most diverse vocab.')
                return most_diverse[:size]
            else :
                most_diverse = sample_diverse_vocab_iterative(2*size)
                print('Computing model predictions for most diverse vocab')
                if self.gpus:
                    self.model.cuda()
                self.preds={}
                for i in most_diverse:
                    self.preds[i] = self.model.predict(self.pool[i])
                return sorted(most_diverse, key=pred_scorers["uncertainty_mean_min3"], reverse=True)[:size]
        
        def uncertainty_mean_for_most_common_vocab(size):
            most_common = sample_most_common_vocab(50)
            if self.nb_iter <= 1 :
                #print('Too early to count on the model to perform sorting.')
                return most_common[:size]
            else :
                print('Computing model predictions for most common vocab')
                if self.gpus:
                    self.model.cuda()
                self.preds={}
                for i in most_common:
                    self.preds[i] = self.model.predict(self.pool[i])
                return sorted(most_common, key=pred_scorers["uncertainty_mean_min3"], reverse=True)[:size]
        
        self.samplers = {
            "random": {
                'sample' : lambda size:random.sample(range(len(self.pool)), size),
                'visibility' : self.k + 10, 'predict_before' : False,
            },
            "random1": {
                'sample' : lambda size:random.sample(range(len(self.pool)), size),
                'visibility' : 1, 'predict_before' : False,
            },
            "ordered": {
                'sample' : lambda size:range(len(self.pool))[:size],
                'visibility' : self.k + 10, 'predict_before' : False,
            },
            "length": {
                'sample' : lambda size:sorted(range(len(self.pool)), key=lambda i:len(self.pool[i]['text']), reverse=True)[:size],
                'visibility' : self.k + 10, 'predict_before' : False,
            },
            "diverse_vocab": {
                'sample' : sample_diverse_vocab_iterative,
                'visibility' : self.k + 10, 'predict_before' : False,
            },
            "common_vocab": {
                'sample' : sample_most_common_vocab,
                'visibility' : self.k + 10, 'predict_before' : False,
            },
            "common_vocab_alternative": {
                'sample' : sample_most_common_vocab_alternative,
                'visibility' : self.k + 10, 'predict_before' : False,
            },
            "diverse_pred": {
                'sample' : sample_diverse_pred_iterative,
                'visibility' : 1,
                'predict_before' : True,
            },
            "diverse_gold": {
                'sample' : sample_diverse_gold_iterative,
                'visibility' : self.k + 10,
                'predict_before' : False
            },
            "diverse_vocab_uncertain": {
                'sample' : uncertainty_mean_for_most_diverse_vocab,
                'visibility' : 1,
                'predict_before' : False,
            },
            "common_vocab_uncertain": {
                'sample' : uncertainty_mean_for_most_common_vocab,
                'visibility' : 1,
                'predict_before' : True,
            },
            
        #add generic scorers
        } | {
            scorer : {
            'sample' : lambda size: sorted(range(len(self.pool)), key=pred_scorers[scorer], reverse=True)[:size],
            'visibility' : 1,
            'predict_before' : True,
            }
            for scorer in pred_scorers.keys()
            }
        
    def run_simulation(self, num_iterations, max_steps, xp_name):
        """Run the simulation for a given number of iterations."""
        self.xp_name = xp_name
        for _ in range(self.k):
            if len(self.queue) == 0:
                self.fill_queue()
            self.annotate_one_queue_element(to_dev_split=True)
        for _ in range(num_iterations):
            self.tracker.epoch_start()
            self.nb_iter += 1
            if len(self.queue) == 0:
                self.fill_queue()
            self.annotate_one_queue_element()
            self.go(max_steps=max_steps)
            self.tracker.epoch_end()
        if self.and_train_on_all_data:
            self.tracker.epoch_start()
            self.nb_iter += len(self.pool)//self.annotiter_size
            self.annotate_all_pool()
            self.go(max_steps=max_steps)
            self.tracker.epoch_end()
    
    def write_docselection(self):
            """Write the selected examples in a file"""
            if self.queue_entry_counter>self.k:
                filename = f'docselection/{self.xp_name}_{self.queue_entry_counter-self.k}.txt'
            else :
                filename = f'docselection/{self.xp_name}_dev{self.queue_entry_counter}.txt'
            print(f'selected examples are written in {filename}')
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            with open(filename,"w") as f:
                f.write(f"====== selected docs at annotiter {self.queue_entry_counter} ============\n")
                for d in self.queue[-1]:
                    f.write(f'-------{d["doc_id"]}------\n')
                    f.write(d['text']+'\n')

    def fill_queue(self):
        strategies = self.selection_strategy.split('+')
        #array of a repartition of the number of examples to select into the number of strategies
        #e.g. if we want to select 10 examples and we have 3 strategies, we will select 4, 3 and 3 examples
        repartition = np.array([self.annotiter_size//len(strategies) for _ in range(len(strategies))])
        repartition[:self.annotiter_size%len(strategies)]+=1
        samplers = [self.samplers[s] for s in strategies]
        if any([s['predict_before'] for s in samplers]) and self.nb_iter > 1:
            print('Computing the new model predictions')
            self.preds={}
            if self.gpus:
                self.model.cuda()
            self.preds = {i:p for i,p in enumerate(self.model.predict(list(self.pool)))}
        for _ in range(min([s['visibility'] for s in samplers])):
            selected_examples = []
            for idx, (sampler, size) in enumerate(zip(samplers, repartition)):
                if self.nb_iter <= 1 and sampler['predict_before']:
                    print(f'Selecting {size} examples randomly (instead of {strategies[idx]}), because model not trained yet')
                    #print(f'Too early to count on the model to perform the {[s for s in strategies if self.samplers[s]["predict_before"]]} strategies. Selecting randomly')
                    sampler = self.samplers['random1']
                    selected_examples.extend(sampler['sample'](size))
                else:
                    print(f"Selecting {size} examples with the {strategies[idx]} strategy")
                    selected_examples.extend(sampler['sample'](size))
            selected_examples = list(set(selected_examples))
            if len(selected_examples) < self.annotiter_size:
                print('found duplicates, adding random examples')
                sampler = self.samplers['random1']
                selected_examples.extend(sampler['sample'](self.annotiter_size-len(selected_examples)))
            res =[]
            for e in sorted(selected_examples, reverse=True):
                res.append(self.pool.pop(e))
            self.queue.append(res)
            self.queue_entry_counter+=1
            self.write_docselection()
                
    def annotate_one_queue_element(self, to_dev_split=False):
        """Annotate one element from the queue"""
        print("annotating one element from the queue")
        rec = self.dataset.val_data if to_dev_split else self.dataset.train_data
        rec.extend(self.queue.pop(0))
    
    def annotate_all_pool(self):
        """Annotate all the pool"""
        # empty the queue
        self.dataset.train_data.extend([e for q in self.queue for e in q])
        self.queue = []
        # add all the pool
        self.dataset.train_data.extend(self.pool)
        self.pool = {}

    def go(self, max_steps):
        """Train the model"""
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        
        iter_name = self.xp_name + '_' + str(self.nb_iter)
        try:
            os.makedirs(os.path.join("checkpoints",os.path.dirname(iter_name)), exist_ok=True)
            trainer = pl.Trainer(
                gpus=self.gpus,
                fast_dev_run=self.debug,
                progress_bar_refresh_rate=1,
                checkpoint_callback=False,  # do not make checkpoints since it slows down the training a lot
                callbacks=[ModelCheckpoint(path='checkpoints/{hashkey}-{global_step:05d}' if not iter_name else 'checkpoints/' + iter_name + '-{global_step:05d}')]
                            if not self.debug else None,
                           #ModelCheckpoint(path='checkpoints/{hashkey}-{global_step:05d}' if not iter_name else 'checkpoints/' + iter_name + '-{hashkey}-{global_step:05d}'),
                           #EarlyStopping(monitor="val_exact_f1",mode="max", patience=3),
                logger=[
                    RichTableLogger(key="epoch", fields={
                        "epoch": {},"step": {},
                        "(.*)_?loss": {"goal": "lower_is_better", "format": "{:.4f}"},
                        "(.*)_(precision|recall|tp)": False,
                        "val_exact_f1": {"goal": "higher_is_better", "format": "{:.4f}"},
                        "(.*)(?<!val_exact)_f1": False,
                        ".*_lr|max_grad": {"format": "{:.2e}"},
                        "duration": {"format": "{:.0f}", "name": "dur(s)"},
                    }),
                    pl.loggers.TestTubeLogger("logs", name=iter_name if iter_name is not None else "untitled_experience"),
                ],
                val_check_interval=max_steps//10,
                max_steps=max_steps
            )
            trainer.fit(self.model, self.dataset)
            trainer.logger[0].finalize(True)
        
            result_output_filename = "checkpoints/{}.json".format(iter_name)
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
          bert_name: str = "camembert/camembert-base",
          fasttext_file: str = "",  # set to "" to disable
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
            bert_lr=3e-5,

            # Optimizer, can be class or str
            optimizer_cls="transformers.AdamW",
            metrics=self.metrics,
        ).train()

        model.encoder.encoders[0].cache = shared_cache
        
        return model
  
