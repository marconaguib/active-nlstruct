# pyner
Named entity recognition (nested or not) in Python

### How to train
```python
from pyner import NER, Vocabulary, load_pretrained
from pyner.datasets import BRATDataset
import string
import torch
import pytorch_lightning as pl
from rich_logger import RichTableLogger


bert_name = "camembert/camembert-large"
ner = NER(
    seed=42,
    sentence_split_regex=r"((?:\s*\n)+\s*|(?:(?<=[a-z0-9)]\.)\s+))(?=[A-Z])",
    sentence_balance_chars=("()",),
    preprocessor=dict(
        module="preprocessor",
        bert_name=bert_name,
        vocabularies=torch.nn.ModuleDict({
            "char": Vocabulary(string.punctuation + string.ascii_letters + string.digits, with_unk=True, with_pad=True),
            "label": Vocabulary(with_unk=False, with_pad=False),
        }).train(),
        word_regex='[\\w\']+|[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]',
        substitutions=(
            (r"(?<=[{}\\])(?![ ])".format(string.punctuation), r" "),
            (r"(?<![ ])(?=[{}\\])".format(string.punctuation), r" "),
            #("(?<=[a-zA-Z])(?=[0-9])", r" "),
            #("(?<=[0-9])(?=[A-Za-z])", r" "),
        )
    ),

    use_embedding_batch_norm=True,
    word_encoders=[
        dict(
            module="char_cnn",
            n_chars=None, # automatically infered from data
            in_channels=8,
            out_channels=50,
            kernel_sizes=(3, 4, 5),
        ),
        dict(
            module="bert",
            path=bert_name,
            n_layers=4,
            freeze_n_layers=-1, # unfreeze all
            dropout_p=0.10,
        )
    ],
    decoder=dict(
        module="exhaustive_biaffine_ner",
        dim=192,
        label_dim=48,
        n_labels=None, # automatically infered from data
        dropout_p=0.3,
        use_batch_norm=False,
        contextualizer=dict(
            module="lstm",
            gate=dict(
                module="sigmoid_gate",
                ln_mode="pre",
                init_value=0,
                proj=False,
                dim=192,
            ),
            input_size=1024 + 150,
            hidden_size=192,
            num_layers=6,
            dropout_p=0.1,
        )
    ),

    init_labels_bias=True,

    batch_size=24,
    use_lr_schedules=True,
    gradient_clip_val=5.,
    main_lr=1e-2,
    top_lr=1e-2,
    bert_lr=4e-5,
    warmup_rate=0.1,
    optimizer_cls="transformers.AdamW",
)

flt_format = (5, "{:.4f}".format)
trainer = pl.Trainer(
    gpus=1,
    progress_bar_refresh_rate=False,
    move_metrics_to_cpu=True,
    logger=[
#        pl.loggers.TestTubeLogger("path/to/logs", name="my_experiment"),
        RichTableLogger(key="epoch", fields={
            "epoch": {},
            "step": {},
            "train_loss":      {"goal": "lower_is_better", "format": "{:.4f}"},
            "train_f1":        {"goal": "higher_is_better", "format": "{:.4f}", "name": "train_f1"},
            "train_precision": {"goal": "higher_is_better", "format": "{:.4f}", "name": "train_p"},
            "train_recall":    {"goal": "higher_is_better", "format": "{:.4f}", "name": "train_r"},

            "val_loss":        {"goal": "lower_is_better", "format": "{:.4f}"},
            "val_f1":          {"goal": "higher_is_better", "format": "{:.4f}", "name": "val_f1"},
            "val_precision":   {"goal": "higher_is_better", "format": "{:.4f}", "name": "val_p"},
            "val_recall":      {"goal": "higher_is_better", "format": "{:.4f}", "name": "val_r"},

            "main_lr": {"format": "{:.2e}"},
            "top_lr": {"format": "{:.2e}"},
            "bert_lr": {"format": "{:.2e}"},
        }),
    ],
    max_epochs=10)
dataset = BRATDataset(
    train="path/to/brat/t3-appr",
    test="path/to/brat/t3-test",
    val=0.2, # first 20% doc will be for validation
    seed=False, # don't shuffle before splitting
)
trainer.fit(ner, dataset)
ner.save_pretrained("ner.pt")
```

### How to use
```python
from pyner import load_pretrained
from pyner.datasets import load_from_brat, export_to_brat
ner = load_pretrained("ner.pt")
export_to_brat(ner.predict(load_from_brat("path/to/brat/t3-test")), filename_prefix="path/to/exported_brat")
```

### How to search hyperparameters

```python
from pyner import NER, Vocabulary
from pyner.datasets import BRATDataset
import string
import torch
import pytorch_lightning as pl
import gc
import optuna

dataset = BRATDataset(
    train="/path/to/brat/t3-appr/",
    test=None,
    val=0.2,
    seed=False, # do not shuffle for val split, just take the first 20% docs
)

def objective(trial):
    bert_name = "camembert/camembert-large"
    ner = NER(
        seed=42,
        sentence_split_regex=r"((?:\s*\n)+\s*|(?:(?<=[a-z0-9)]\.)\s+))(?=[A-Z])",
        sentence_balance_chars=("()",),
        preprocessor=dict(
            module="preprocessor",
            bert_name=bert_name,
            vocabularies=torch.nn.ModuleDict({
                "char": Vocabulary(string.punctuation + string.ascii_letters + string.digits, with_unk=True, with_pad=True),
                "label": Vocabulary(with_unk=False, with_pad=False),
            }).train(), # keep training vocabularies on data
            word_regex='[\\w\']+|[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]',
            substitutions=(
                (r"(?<=[{}\\])(?![ ])".format(string.punctuation), r" "), # spaces around punctuation
                (r"(?<![ ])(?=[{}\\])".format(string.punctuation), r" "), # spaces around punctuation
                ("(?<=[a-zA-Z])(?=[0-9])", r" "), # spaces between letters and numbers
                ("(?<=[0-9])(?=[A-Za-z])", r" "), # spaces between letters and numbers
            )
        ),

        use_embedding_batch_norm=trial.suggest_categorical("use_embedding_batch_norm", [False, True]),
        word_encoders=[
            dict(
                module="char_cnn",
                n_chars=None, # automatically infered from data
                in_channels=8,
                out_channels=50,
                kernel_sizes=(3, 4, 5),
            ),
            dict(
                module="bert",
                path=bert_name,
                n_layers=4,
                freeze_n_layers=0, # unfreeze all
                dropout_p=trial.suggest_float("word_encoders/dropout_p", 0, 0.2, step=0.05),
            )
        ],
        decoder=dict(
            module="exhaustive_biaffine_ner",
            dim=trial.suggest_int("decoder/dim", 128, 256, 32),
            label_dim=trial.suggest_int("decoder/label_dim", 32, 128, 16),
            n_labels=None, # automatically infered from data
            dropout_p=trial.suggest_int("decoder/dropout_p", 0, 0.4, step=0.05),
            use_batch_norm=False,
            contextualizer=dict(
                module="lstm",
                gate=dict(
                    module="sigmoid_gate",
                    ln_mode=trial.suggest_categorical("decoder/contextualizer/gate/ln_mode", ["pre", "post", False]),
                    init_value=0,
                    proj=False,
                    dim=trial.params["decoder/dim"],
                ),
                input_size=1024 + 150,
                hidden_size=trial.params["decoder/dim"],
                num_layers=trial.suggest_int("decoder/contextualizer/num_layers", 1, 6),
                dropout_p=trial.suggest_float("decoder/contextualizer/dropout_p", 0, 0.4, step=0.05),
            )
        ),

        init_labels_bias=True,

        batch_size=24,
        use_lr_schedules=True,
        gradient_clip_val=5.,
        main_lr=trial.suggest_float("main_lr", 1e-4, 1e-1, log=True),
        top_lr=trial.suggest_float("top_lr", 1e-3, 1e-1, log=True),
        bert_lr=4e-5,
        warmup_rate=0.1,
        optimizer_cls="transformers.AdamW", # can be real model or serialized string
    ).train()

    trainer = pl.Trainer(
        gpus=1,
        progress_bar_refresh_rate=False,
        move_metrics_to_cpu=True,
        logger=[
            pl.loggers.TestTubeLogger("tensorboard_data", name="pyner-brat"),
        ],
        max_epochs=50)
    
    print(trial.params)
    trainer.fit(ner, dataset)
    
    res = trainer.callback_metrics["val_f1"].item()
    print("=>", res)
    
    # Clean cuda memory after trial because optuna does not
    del trainer, ner
    gc.collect()
    torch.cuda.empty_cache()
    
    return res

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.PercentilePruner(75, n_startup_trials=5, n_warmup_steps=10))
pl.seed_everything(42)
study.optimize(objective, n_trials=100, gc_after_trial=True)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
```