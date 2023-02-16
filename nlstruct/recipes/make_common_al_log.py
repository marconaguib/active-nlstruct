import seaborn as sns
import numpy as np
import re
import os
import glob
import json
import argparse


sns.set_theme()

parser = argparse.ArgumentParser()
parser.add_argument('-c','--corpus_name_or_names', type=str, nargs='+', help='names of the corpora to do the expetiment on',default=[])
parser.add_argument('-s','--strategies', type=str, nargs='+', help='strategies to plot',default=[])
parser.add_argument('-p','--prefix', type=str, help='global prefix to where to find checkpoints and logs',default='.')
parser.add_argument('-o','--output', type=str, help='output file',default='common_log.csv')

args = parser.parse_args()

with open(args.output, 'w') as log_file:
    log_file.write('corpus;batch;score;type_f1;xp_name;word_count;seed\n')
    for corpus in args.corpus_name_or_names :
        for xp_name_prefix in args.strategies:
            for xp_name_full in glob.glob(os.path.join(f'{args.prefix}/checkpoints/{corpus}', xp_name_prefix+'seed*_1.json')):
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
                        if int(batchname) <= 20:
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
    
