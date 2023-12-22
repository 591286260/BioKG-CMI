import pandas as pd
from sklearn.decomposition import PCA

input_file = '9905/circRNA_similarities.csv'
data = pd.read_csv(input_file, header=None)
X = data.values

def perform_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca

n_components = 8

X_pca = perform_pca(X, n_components)

output_file = '9905/circRNA_similarities(8).csv'
df = pd.DataFrame(X_pca)
df.to_csv(output_file, index=False, header=False)

input_file = '9905/miRNA_similarities.csv'
data = pd.read_csv(input_file, header=None)
X = data.values
