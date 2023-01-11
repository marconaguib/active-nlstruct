import pytorch_lightning as pl
from carbontracker.tracker import CarbonTracker

#ENTITIES = ['AdministrationRoute','Anatomy','Aspect','Assertion','BiologicalProcessOrFunction','Chemicals_Drugs','Concept_Idea','Devices','Disorder','Dosage','DrugForm','Genes_Proteins','Hospital','LivingBeings','Localization','Measurement','MedicalProcedure','Persons','SignOrSymptom','Strength','Temporal']

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

#def classic_build_model(
#      dataset: BRATDataset,
#      seed: int = 42,
#      do_char: bool = True,
#      do_biaffine: bool = True,
#      do_tagging: str = "full",
#      doc_context: bool = True,
#      finetune_bert: bool = False,
#      bert_lower: bool = False,
#      max_tokens: int = 256,
#      n_bert_layers: int = 4,
#      n_lstm_layers: int = 3,
#      biaffine_size: int = 150,
#      bert_proj_size: int = None,
#      biaffine_loss_weight: float = 1.,
#      hidden_size: int = 400,
#      val_check_interval: int = None,
#      bert_name: str = "camembert/camembert-base",
#      fasttext_file: str = "",  # set to "" to disable
#      unique_label: int = False,
#      norm_bert: bool = False,
#      dropout_p: float = 0.1,
#      batch_size: int = 16,
#      lr: float = 1e-3,
#      use_lr_schedules: bool = True,
#      word_pooler_mode: str = "mean",
#      predict_kwargs: Dict[str, any] = {},
#):
#     return
