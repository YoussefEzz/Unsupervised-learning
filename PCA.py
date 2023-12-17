from sklearn import decomposition
import pandas as pd

#read and parse the training set .csv features file 
df = pd.read_csv('A3-data.txt', delimiter = ',')

#read four input(features) columns
x = df[df.columns[0]]
y = df[df.columns[1]]
z = df[df.columns[2]]
t = df[df.columns[3]]

data = [x, y, z, t]


pca = decomposition.PCA(n_components=4)
principal_components = pca.fit_transform(data)
#print(principal_components)

principal_df = pd.DataFrame(principal_components, columns = ['PC1', 'PC2', 'PC3', 'PC4'])
print(principal_df)

print("component vectors")
print(pca.components_)

print("Explained variance")
print(pca.explained_variance_)


print("Explained variance ratio")
print(pca.explained_variance_ratio_)

print(principal_df['PC1'][0])

data_mean = data - data.mean[0]
print(data_mean.iloc[0].dot(pca.components_[0]))
