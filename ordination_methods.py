import pandas as pd
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

RANDOMSEED = np.random.RandomState(seed=2)

df = pd.read_csv('irisdata.csv')
df.columns = ['sl', 'sw', 'pl', 'pw', 'label']
labels = np.sort(df['label'].unique())
N_labels = len(labels)

cmap = matplotlib.cm.get_cmap('jet')
colorset = {e: cmap(float(idx)/N_labels) for idx, e in enumerate(labels)}
labelcolors = df['label'].map(colorset).tolist()

# executing PCA
print('calculating PCA')
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.drop('label', axis=1).values)
PCA_x_scores = pca_result[:,0]
PCA_y_scores = pca_result[:,1]

# executing MDS
print('calculating MDS')
dissimilarities = euclidean_distances(df.drop('label', axis=1))
mds = MDS(n_components=2, max_iter=1000, random_state=RANDOMSEED, dissimilarity="precomputed")
mds_result = mds.fit_transform(dissimilarities)
MDS_x_scores = mds_result[:,0]
MDS_y_scores = mds_result[:,1]

# executing NMDS
print('calculating NMDS')
dissimilarities = euclidean_distances(df.drop('label', axis=1))
nmds = MDS(n_components=2, metric=False, max_iter=1000, random_state=RANDOMSEED, dissimilarity="precomputed")
nmds_result = nmds.fit_transform(dissimilarities)
NMDS_x_scores = nmds_result[:,0]
NMDS_y_scores = nmds_result[:,1]

# executing tSNE
print('calculating tSNE')
dissimilarities = euclidean_distances(df.drop('label', axis=1))
tSNE = TSNE(n_components=2, random_state=RANDOMSEED, metric='euclidean')
tSNE_result = tSNE.fit_transform(df.drop('label', axis=1).values)
tSNE_x_scores = tSNE_result[:,0]
tSNE_y_scores = tSNE_result[:,1]

# Plotting
fig = plt.figure(figsize=(24, 6))
plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.2, hspace = 0.2)
# PCA
ax1 = fig.add_subplot(1,4,1)
plt.title('PCA')
ax1.scatter(PCA_x_scores, PCA_y_scores, c=labelcolors)
plt.xlabel('X scores')
plt.ylabel('Y scores')
patches = [matplotlib.patches.Patch(color=colorset[e], label=e) for e in labels]
#patches = [matplotlib.patches.Patch(color=colorset[e]cmap(idx/N_labels), label=e) for idx, e in enumerate(labels)]
plt.legend(handles=patches)
# MDS
ax2 = fig.add_subplot(1,4,2)
plt.title('MDS')
ax2.scatter(MDS_x_scores, MDS_y_scores, c=labelcolors)
plt.xlabel('X scores')
plt.ylabel('Y scores')
# NMDS
ax3 = fig.add_subplot(1,4,3)
plt.title('NMDS')
ax3.scatter(NMDS_x_scores, NMDS_y_scores, c=labelcolors)
plt.xlabel('X scores')
plt.ylabel('Y scores')
# tSNE
ax4 = fig.add_subplot(1,4,4)
plt.title('tSNE')
ax4.scatter(tSNE_x_scores, tSNE_y_scores, c=labelcolors)
plt.xlabel('X scores')
plt.ylabel('Y scores')
# save
ax1.figure.savefig('ordination_in_Python.pdf')
