import pandas as pd
import preproc_data
import matplotlib.pyplot as plt

df, _, _ =  preproc_data.return_data(orig=True)
ax = df.boxplot(column='age', by='job', rot=45)
ax.set_title('')
ax.set_ylabel('age')
plt.savefig(r'', bbox_inches='tight')


df.groupby(['month', 'y']).size().unstack().plot(kind='bar', stacked=True)
plt.gca().set_ylabel('count')
plt.savefig(r'', bbox_inches='tight')

# df['pdays'].plot.hist(bins=50)
# plt.gca().set_ylabel('count')
# plt.savefig(r'', bbox_inches='tight')
df['pdays'].quantile(q=[0, 0.005, 0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 0.995, 1])
df.boxplot(column='pdays')