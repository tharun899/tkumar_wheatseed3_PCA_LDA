from sklearn.preprocessing import StandardScaler
import pandas as pd

# load the dataset
df = pd.read_csv('seeds_dataset.txt', delimiter='\t', header=None)
# separate the features and the labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
#Apply PCA
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
# create a PCA object with 7 components
pca = PCA(n_components=7)
# fit the PCA model to the data
X_pca = pca.fit_transform(X_std)
# plot the cumulative explained variance ratio
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance ratio')
plt.show()

# create a new PCA object with 2 components
pca = PCA(n_components=2)
# fit the PCA model to the data
X_pca = pca.fit_transform(X_std)

# plot the data in the new space
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()
#apply LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# create an LDA object with 2 components
lda = LinearDiscriminantAnalysis(n_components=2)

# fit the LDA model to the data
X_lda = lda.fit_transform(X_std, y)

# plot the data in the new space
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
plt.xlabel('First discriminant')
plt.ylabel('Second discriminant')
plt.show()

