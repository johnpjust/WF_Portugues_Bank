import pandas as pd
import preproc_data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

########### general investigative plots ##############
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

###################### ANN Results ###################################
ax = pd.DataFrame(np.hstack((labels, tf.nn.sigmoid(model(df.values.astype(np.float32))).numpy())), columns=['result', 'predicted']).boxplot(column='predicted', by='result')
plt.gca().set_title('')
plt.gca().set_ylabel('predicted')
plt.savefig(r'D:\Personal\presentations\WF\pred_boxplot.png', bbox_inches='tight')

sns.set(style="whitegrid")
ax = sns.boxplot(x="result", y="predicted", data=pd.DataFrame(np.hstack((labels, tf.nn.sigmoid(model(df.values.astype(np.float32))).numpy())), columns=['result', 'predicted']), showfliers = False)
ax = sns.swarmplot(x="result", y="predicted", data=pd.DataFrame(np.hstack((labels[:1000], tf.nn.sigmoid(model(df.values.astype(np.float32))).numpy()[:1000])), columns=['result', 'predicted']), color=".25")