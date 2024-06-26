from nlstruct.recipes import AL_Simulator
import os
import warnings
import sys
warnings.filterwarnings("ignore")
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('corpus_names', type=str, nargs='+', help='names of the corpora to do the experiment on',default=[])
parser.add_argument('-r','--random_seeds', type=int, nargs='+', help='random seeds', default=[1])
parser.add_argument('-s','--strategies', type=str, nargs='+', help='strategies to apply',default=[])
parser.add_argument('-b','--bert_name', type=str, help='bert name',default="camembert/camembert-large")
parser.add_argument('-i','--entities_to_ignore', type=str, nargs='+', help='', default=[])
parser.add_argument('-d','--debug', action='store_true', help='debug mode', default=False)
args = parser.parse_args()
assert all([os.path.exists(name) for name in args.corpus_names])
for corpus in args.corpus_names:
    for strategy in args.strategies :
        for randseed in args.random_seeds if strategy!='length' else args.random_seeds[:1]:
            sim = AL_Simulator(dataset_name={"train": f"{corpus}/training","test": f"{corpus}/test"}, 
                    selection_strategy=strategy, al_seed=randseed,seed=randseed,
                    and_train_on_all_data=not any([int(os.path.basename(fn)[:-5].split('_')[-1])>10
                                           for fn in glob.glob(os.path.join('checkpoints',corpus,'*json'))]),
                    entities_to_remove_from_pool = ['papieralettre','signature','Assertion','Aspect'],
                    entities_to_ignore = args.entities_to_ignore,
                    finetune_bert=True,
                    bert_name=args.bert_name,
                    annotiter_size=10,
                    k=2,
                    debug=args.debug,
                    )
            ignore_suffix = f"_no{''.join(args.entities_to_ignore).lower()}" if len(args.entities_to_ignore) else ''
            xp_name = f"{corpus}{ignore_suffix}/{strategy}seed{randseed}"
            sim.run_simulation(num_iterations=10, max_steps=1000, xp_name=xp_name)

