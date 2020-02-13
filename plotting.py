import pandas as pd
import preproc_data
import matplotlib.pyplot as plt

df, _, _ =  preproc_data.return_data(orig=True)
ax = df.boxplot(column='age', by='job', rot=45)
ax.set_title('')
ax.set_ylabel('age')
plt.savefig(r'', )

df.plot(column='job', kind='bar')