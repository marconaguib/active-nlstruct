import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import glob
import pandas as pd
import json
import argparse



sns.set_theme()

parser = argparse.ArgumentParser()
parser.add_argument('-c','--corpus_name_or_names', type=str, nargs='+', help='names of the corpora to do the expetiment on',default=[])
parser.add_argument('-r','--read_logs', action='store_true', help='read from pre-existing log file', default=False)
parser.add_argument('-s','--strategies', type=str, nargs='+', help='strategies to plot',default=[])
parser.add_argument('-p','--prefix', type=str, help='global prefix to where to find checkpoints and logs',default='.')

args = parser.parse_args()
common_log_fn = f"./common_log.csv"

if not args.read_logs:
    with open(common_log_fn, 'w') as log_file:
        log_file.write('corpus;batch;score;type_f1;xp_name\n')
        for corpus in args.corpus_name_or_names :
            for xp_name_prefix in args.strategies:
                dev_scores = {}
                test_scores = {}
                for xp_name_full in glob.glob(os.path.join(f'{args.prefix}/checkpoints/{corpus}', xp_name_prefix+'*_1.json')):
                    xp_name = os.path.basename(xp_name_full.replace('_1.json',''))
    
                    for fn in glob.glob(os.path.join(f'{args.prefix}/checkpoints/{corpus}', xp_name+'_*json')):
                        batchname = fn[:-5].split('_')[-1]
                        with open(fn,'r') as f:
                            test_dico = json.load(f)
                            f1_micro = test_dico["results"]['exact']['f1']
                            other_scores = [v for k,v in test_dico['results']['exact'].items() if k.endswith('_f1')]
                            f1_macro = np.mean(other_scores) if len(other_scores) else f1_micro
                            log_file.write(f'{corpus};{batchname};{f1_micro};micro;{xp_name_prefix}\n')
                            log_file.write(f'{corpus};{batchname};{f1_macro};macro;{xp_name_prefix}\n')
    

dataset = pd.read_csv(common_log_fn,sep=';')
dataset['batch_str'] = dataset['batch'].apply(str)
dataset.sort_values(['batch','xp_name'],inplace=True)
print(dataset)



def plot_iterations(data,**kwargs):
    sns.lineplot(data=data.query('batch<=10'), **kwargs)

def plot_all(data, **kwargs):
    value_to_mean = kwargs.pop('value','score')
    plt.axhline(y=data[value_to_mean].max(), **kwargs)

g = sns.FacetGrid(data=dataset, col='corpus', row='type_f1',sharey=True)
g.map_dataframe(plot_iterations, x='batch_str',y="score", hue='xp_name', hue_order=dataset['xp_name'].unique())
g.map_dataframe(plot_all,value='score',ls='--',c='black')
g.add_legend(title='Selection strategy',)

ax = plt.gca()
ax.set_ylim(0,1)
plt.show()

