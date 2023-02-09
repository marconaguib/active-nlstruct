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
parser.add_argument('-w','--word_count', action='store_true', help='plot word count instead of batch', default=False)

args = parser.parse_args()
common_log_fn = f"./common_log.csv"

if not args.read_logs:
    with open(common_log_fn, 'w') as log_file:
        log_file.write('corpus;batch;score;type_f1;xp_name;word_count;seed\n')
        for corpus in args.corpus_name_or_names :
            for xp_name_prefix in args.strategies:
                for xp_name_full in glob.glob(os.path.join(f'{args.prefix}/checkpoints/{corpus}', xp_name_prefix+'*_1.json')):
                    xp_name = os.path.basename(xp_name_full.replace('_1.json',''))
    
                    for fn in glob.glob(os.path.join(f'{args.prefix}/checkpoints/{corpus}', xp_name+'_*json')):
                        fn_docselection = fn.replace('checkpoints','docselection').replace('.json','.txt')
                        batchname = fn[:-5].split('_')[-1]
                        seed = re.search(r'(?<=seed)\d+',fn).group(0)
                        with open(fn,'r') as f:
                            test_dico = json.load(f)
                            f1_micro = test_dico["results"]['exact']['f1']
                            other_scores = [v for k,v in test_dico['results']['exact'].items() if k.endswith('_f1')]
                            f1_macro = np.mean(other_scores) if len(other_scores) else f1_micro
                            if int(batchname) <= 10:
                                with open(fn_docselection,'r') as f:
                                    s = f.read()
                                #remove lines beginning with "====" or "---"
                                s = re.sub(r'^=+.*$','',s,flags=re.MULTILINE)
                                s = re.sub(r'^-+.*$','',s,flags=re.MULTILINE)
                                #count words
                                word_count = len(s.split())
                            else:
                                word_count = -1
                            log_file.write(f'{corpus};{batchname};{f1_micro};micro;{xp_name_prefix};{word_count};{seed}\n')
                            log_file.write(f'{corpus};{batchname};{f1_macro};macro;{xp_name_prefix};{word_count};{seed}\n')
    

dataset = pd.read_csv(common_log_fn,sep=';')
dataset.sort_values(['batch','xp_name'],inplace=True)
print(dataset.head(20))

#mean the word_count over every batch
dataset['word_count'] = dataset.groupby(['corpus','type_f1','xp_name','batch'])['word_count'].transform('mean')

#cumsum the word_count over every batch and seed
dataset['word_count'] = dataset.groupby(['corpus','type_f1','xp_name','seed'])['word_count'].transform('cumsum')


def plot_iterations(data,**kwargs):
    sns.lineplot(data=data.query('batch<=10'), **kwargs)

def plot_all(data, **kwargs):
    value_to_mean = kwargs.pop('value','score')
    plt.axhline(y=data[value_to_mean].max(), **kwargs)

g = sns.FacetGrid(data=dataset, col='corpus', row='type_f1',sharey=True, sharex=False)
g.map_dataframe(plot_iterations, x= 'batch' if not args.word_count else 'word_count', y="score", hue='xp_name', hue_order=dataset['xp_name'].unique())
g.map_dataframe(plot_all,value='score',ls='--',c='black')
g.add_legend(title='Selection strategy',)

ax = plt.gca()
ax.set_ylim(0,1)
plt.show()

