from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA


def tsne (x, k=2, p=30, N=4000): 
    mytsne = TSNE(n_components=k, init='pca', perplexity=p,
                  n_iter=N)
    return mytsne.fit_transform(x)

def pca (x, k=2):
    return PCA(n_components=k).fit_transform(x)

def mdse (x, k=2, e=1e-3):
    return MDS(n_components=k, eps=e).fit_transform(x)
