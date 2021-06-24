import pandas as pd 
from sklearn import datasets 
import matplotlib.pyplot as plt

wine = datasets.load_wine()
print(dir(wine))

print(wine.feature_names)

df = pd.DataFrame(wine.data, columns = wine.feature_names)
print(df)

df['target'] = wine.target
print(df)

#df0 = df[:59]
df0 = df[df['target']==0]
#print(df0)
df1 = df[df['target']==1]
#print(df1)
df2 = df[df['target']==2]
print(df2)

plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.title('Wine Recognition')
plt.scatter(df0['alcohol'], df0['malic_acid'],color="green",marker='*')
plt.scatter(df1['alcohol'], df1['malic_acid'],color="tomato",marker='*')
plt.show()

ax = plt.axes(projection='3d')
ax.scatter3D(df0['alcohol'], df0['malic_acid'], df0['proline'], 'green',marker='*')
ax.scatter3D(df1['alcohol'], df1['malic_acid'], df1['proline'], 'lightblue',marker='*')
ax.set_xlabel('Alcohol')
ax.set_ylabel('Malic Acid')
ax.set_zlabel('proline')
plt.title('Wine Recognition')
#ax.plot3D(df0['alcohol'], df0['malic_acid'], df0['proline'],'lightgreen')
plt.show()
