import pandas as pd 
from sklearn import datasets 
import matplotlib.pyplot as plt
import matplotlib as mpl
wine = datasets.load_wine()
print(dir(wine))

print(wine.feature_names)

df = pd.DataFrame(wine.data, columns = wine.feature_names)
print(df)
df.drop(df.iloc[:, 3:], inplace = True, axis = 1)
print(wine.target_names)
df['target'] = wine.target
print("fnl\n",df)

#df0 = df[:59]
df0 = df[df['target']==0]
#print(df0)
df1 = df[df['target']==1]
#print(df1)
df2 = df[df['target']==2]
print(df2)
'''
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.title('Wine Recognition')
plt.scatter(df0['alcohol'], df0['malic_acid'],color="green",marker='*')
plt.scatter(df1['alcohol'], df1['malic_acid'],color="tomato",marker='*')
plt.show()
'''
#mpl.rcParams['legend.fontsize'] = 10
ax = plt.axes(projection='3d')
ax.scatter3D(df0['alcohol'], df0['malic_acid'], df0['ash'], 'green',marker='*')
ax.scatter3D(df1['alcohol'], df1['malic_acid'], df1['ash'], 'lightblue',marker='*')

ax.legend()


ax.set_xlabel('Alcohol')
ax.set_ylabel('Malic Acid')
ax.set_zlabel('ash')
plt.title('Wine Recognition')

#ax.plot3D(df0['alcohol'], df0['malic_acid'], df0['ash'],'lightgreen')

plt.show()
