import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 

df = pd.read_csv('time_table.csv', sep = ',', )
print(df.describe())
print(df.shape)


g = sns.factorplot("# function", "time/call (s)", "nb inputs", data=df, kind="bar", palette = "gist_rainbow", legend=False, lw = 1, ec = 'black')
plt.legend(title = "nb inputs",loc="upper right", ncol=2)
plt.tight_layout()
plt.show()

#"benchmark", "nb inputs", "nb outputs", "output idx", "accuracy tree", "accuracy ABC", "nb ands", "aig depth", "sop"
df2 = pd.read_csv('training_results.csv', sep = ',', )
print(df.describe())
print(df.shape)

g = sns.scatterplot("nb ands", "aig depth", data=df2, lw = 1, ec = 'black', legend=False)
plt.tight_layout()
plt.show()

new = df2["benchmark"].str.split("_", n = 1, expand = True) 

df2["benchmark"] = new[0]

df2["total nb ands"] = df2['nb ands'].groupby(df2['benchmark']).transform('mean')

df2 = df2.sort_values(['total nb ands']).reset_index(drop=True)

g = sns.factorplot("benchmark", "nb ands", data=df2, kind = "bar", palette = "Blues", legend=False)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

g = sns.factorplot("benchmark", "aig depth", data=df2, kind = "bar", palette = "Blues", legend=False)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()