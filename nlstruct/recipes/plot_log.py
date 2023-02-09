import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme()
parser = argparse.ArgumentParser()
parser.add_argument('-w','--word_count', action='store_true', help='plot word count instead of batch', default=False)
parser.add_argument('-i','--input', type=str, help='input file',default='./common_log.csv')

args = parser.parse_args()

dataset = pd.read_csv(args.input,sep=';')
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

