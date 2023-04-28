import argparse
from matplotlib import ticker
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":","grid.linefreq":1})
parser = argparse.ArgumentParser()
parser.add_argument('-w','--word_count', action='store_true', help='plot word count instead of batch', default=False)
parser.add_argument('-i','--input', type=str, help='input file',default='./common_log.csv')
parser.add_argument('-e','--extended', action='store_true', help='plot the random strategy until the end', default=False)

args = parser.parse_args()

dataset = pd.read_csv(args.input,sep=';')
dataset.sort_values(['batch','xp_name'],inplace=True)
print(dataset.head(20))

#mean the word_count over every batch
dataset['word_count'] = dataset.groupby(['corpus','type_f1','xp_name','batch'])['word_count'].transform('mean')

#cumsum the word_count over every batch and seed
dataset['word_count'] = dataset.groupby(['corpus','type_f1','xp_name','seed'])['word_count'].transform('cumsum')

dataset = dataset.query('type_f1!="min"')

#for each corpus, xp_name and type_f1, get the score at 1000 word count and put them in a new dataframe
rel_perfs = pd.DataFrame(columns=['corpus','type_f1','xp_name','score_at_1000'])
for c, corpus_data in dataset.groupby(['corpus']):
    max_score_by_type = corpus_data.groupby('type_f1')['score'].max()
    print(c)
    print(max_score_by_type)
    for g,xp_type_data in corpus_data.groupby(['xp_name','type_f1']):
        #get the lines where the word count is > 1000
        if xp_type_data.query('word_count>1000').empty:
            score = pd.NA
        else :
            xp_type_data['score_mean'] = xp_type_data.groupby('batch')['score'].transform('mean')
            idx = xp_type_data.query('word_count>1000').index[0]
            score = xp_type_data.loc[idx,'score_mean']
        score_ratio = round(score/max_score_by_type[g[1]],2) if score is not pd.NA else pd.NA
        rel_perfs = pd.concat([rel_perfs,pd.DataFrame({'corpus':[c],'type_f1':[g[1]],'xp_name':[g[0]],'score_at_1000':[score_ratio]})],ignore_index=True)
        
corpus_order = ['merlot', 'emea', 'medline', 'e3c_clean']
rel_perfs['corpus'] = pd.Categorical(rel_perfs['corpus'], categories=corpus_order, ordered=True)
dataset['corpus'] = pd.Categorical(dataset['corpus'], categories=corpus_order, ordered=True)
rel_perfs.sort_values(['xp_name','corpus','type_f1'], inplace=True)
print(rel_perfs)


def plot_iterations(data,**kwargs):
    sns.lineplot(data=data.query('batch<=20'), **kwargs)

def plot_all(data, **kwargs):
    value_to_mean = kwargs.pop('value','score')
    plt.axhline(y=data[value_to_mean].max(), **kwargs, label='Score obtenu en utilisant toutes les données')

hue_order = list(dataset['xp_name'].unique())
hue_order= sorted(hue_order,key=lambda x: 0 if x.startswith('random') else 1)

# #for each corpus, stop the random plot at where the other plots stop
# if args.word_count and not args.extended:
#     max_word_count = dataset.groupby(['corpus']).apply(lambda x: x.query('xp_name!="random"')['word_count'].max())
#     max_word_count = max_word_count*1.1
#     for c in max_word_count.index:
#         dataset = dataset.query('corpus!="{}" or word_count<={} or batch>20'.format(c,max_word_count[c]))

g = sns.FacetGrid(data=dataset, col='corpus', row='type_f1',sharey=True, sharex=False)
g.map_dataframe(plot_iterations, x= 'batch' if not args.word_count else 'word_count', y="score", hue='xp_name', hue_order=hue_order)
g.map_dataframe(plot_all,value='score',ls='--',c='black')
# put legend at the bottom
g.add_legend(title='Stratégie de sélection',bbox_to_anchor=(0,0,1,0.5),loc='lower center',ncol=3)
#tight layout
g.fig.tight_layout()

#set the titles of each facet
g.set_titles(template='{col_name}')

ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
for ax in g.axes.flat:
    ax.set_ylim(0,1)
    if ax in g.axes[0]:
        ax.set_ylabel('f1_micro')
        ax.set_xlabel('')
    else:
        ax.set_ylabel('f1_macro')
        ax.set_title('')
        ax.set_xlabel('Iterations' if not args.word_count else 'Nombre de mots')

#fix borders
fig = plt.gcf()
top=0.962
bottom=0.145
left=0.064
right=0.982
hspace=0.252
wspace=0.033
fig.subplots_adjust(top=top,bottom=bottom,left=left,right=right,hspace=hspace,wspace=wspace)
plt.show()

