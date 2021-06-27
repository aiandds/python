#import necessary dataset 
import pandas as pd 
from sklearn import datasets 
import matplotlib.pyplot as plt

#loading dataset from sklearn
wine = datasets.load_wine()

#printing directory
print(dir(wine))

#printing fearure_name
print(wine.feature_names)

#converting dataset into pandas dataframe 
df = pd.DataFrame(wine.data, columns = wine.feature_names)

#printing dataframe
print(df)

#dropping unnessory column
df.drop(df.iloc[:, 3:], inplace = True, axis = 1)

#printing target name
print(wine.target_names)
#adding target to the dataframe 
df['target'] = wine.target
#printing final dataframe
print("fnl\n",df)

#separating with the target 0
df0 = df[df['target']==0]
#printing target 0 data 
print(df0)
#separating with the target 1
df1 = df[df['target']==1]
#printing target 1 data 
print(df1)
#separating with the target 2
df2 = df[df['target']==2]
#printing target 2 data 
print(df2)


#adding x and y lable
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
#adding title
plt.title('Wine Recognition')
#plotting the data 
plt.scatter(df0['alcohol'], df0['malic_acid'],color="green",marker='*',label = "type_0")
#adding ledgent to graph
plt.legend()
#show the graph
plt.show()


#2nd graph
#adding x and y lable
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
#adding title
plt.title('Wine Recognition')
#plotting the data 
plt.scatter(df1['alcohol'], df1['malic_acid'],color="tomato",marker='*',label = "type_1")
#adding ledgent to graph
plt.legend()
#show the graph
plt.show()


#3rd graph
#adding x and y lable
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
#adding title
plt.title('Wine Recognition')
#plotting the data 
plt.scatter(df0['alcohol'], df0['malic_acid'],color="green",marker='*',label = "type_0")
plt.scatter(df1['alcohol'], df1['malic_acid'],color="tomato",marker='*',label = "type_1")
#adding ledgent to graph
plt.legend()
#show the graph
plt.show()


#for '3D' graph 
ax = plt.axes(projection='3d')
#adding data for 3d graph
ax.scatter3D(df0['alcohol'], df0['malic_acid'], df0['ash'], 'green',marker='*',label = "type_0")
ax.scatter3D(df1['alcohol'], df1['malic_acid'], df1['ash'], 'lightblue',marker='*',label = "type_1")
#adding ledgent to graph
ax.legend()
#adding x , y and z lable
ax.set_xlabel('Alcohol')
ax.set_ylabel('Malic Acid')
ax.set_zlabel('Ash')
# adding title 
plt.title('Wine Recognition')
#show the graph
plt.show()
