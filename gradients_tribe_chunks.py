import numpy as np
from nilearn.connectome import ConnectivityMeasure
from brainspace.gradient import GradientMaps
from utils.fmri_utils import get_masked_data, load_fmri,load_chunk_predictions
import argparse
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_surf_destrieux, load_fsaverage

from sklearn.metrics.pairwise import cosine_similarity
from nilearn.plotting import plot_matrix
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from bct import threshold_proportional
from nilearn.maskers import SurfaceLabelsMasker
from nilearn.surface import SurfaceImage

fsaverage = load_fsaverage("fsaverage5")
destrieux = fetch_atlas_surf_destrieux()


parser = argparse.ArgumentParser(description='Load and analyze fMRI data')
parser.add_argument('--basedir', type=str, default='/home/nfarrugi/Documents/git/download', help='Base directory path')
args = parser.parse_args()

basedir = args.basedir

# Load the fMRI responses

# movie names 
movies_names = ['bourne', 'wolf', 'figures', 'life']

chunk_preds = {}
for movie in movies_names:
    chunk_preds[movie] = load_chunk_predictions(f'{movie}_predictions.npz', movie)

# make a single disctionnary fmri with the four movies and their chunks predictions
fmri = {}
for movie in movies_names:
    for chunk in chunk_preds[movie].keys():
        fmri[f'{movie}_{chunk}'] = chunk_preds[movie][chunk]

# Print all available movies
print(f"Tribe model fMRI movies splits name and shape:")
for key, value in fmri.items():
    print(key + " " + str(value.shape))


## regroup fMRI time series for each movie
fmri_stack = {}
for movie in movies_names:
    keys_movie = [key for key in fmri.keys() if movie in key]
    fmri_stack[movie] = [fmri[key] for key in keys_movie]


## Compute sets of functional connectivity matrix for each movie using nilearn ConnectivityMeasure, but keeping a matrix per run. Alignement per movie
gradients = {}
eigenvalues = {}
connectivity_matrices = {}
for movie in movies_names:
    try:
        print(f"Processing {movie}...")
            # Create a ConnectivityMeasure object for correlation
        connectivity_measure = ConnectivityMeasure(kind='correlation',standardize='zscore_sample')
        # Compute the functional connectivity matrix
        connectivity_matrix = connectivity_measure.fit_transform([get_masked_data(f) for f in fmri_stack[movie]])

        # Threshold the connectivity matrix to keep only the top 10% of connections
        connectivity_matrix = np.stack([threshold_proportional(cc, 0.1) for cc in connectivity_matrix])
        connectivity_matrices[movie] = connectivity_matrix
        print(f"{movie} affinity matrix shape: {connectivity_matrix.shape}")

        ## Compute gradients for each movie using BrainSpace GradientMaps
        gm = GradientMaps(n_components=10, random_state=0,alignment='procrustes', kernel='normalized_angle')
        
        ## load the reference gradients from the fMRI data
        refgradients = np.load(f'reference_gradients_bourne_subject_1.npz')['refgradients']

        ## estimate the gradients for the current movie, using the reference gradients for alignment
        gm.fit([c for c in connectivity_matrix],reference=refgradients)
        gradients[movie] = gm.aligned_
        eigenvalues[movie] = gm.lambdas_
        print(f"{movie} gradients shapes: {[g.shape for g in gradients[movie]]}")
    except Exception as e:
        print(f"Error processing {movie}: {e}")
        continue



### visualize similarity between gradients across all movies and runs using a heatmap of the correlation between the gradients

# Compute the correlation matrix between the gradients of all movies and runs
# each gradient is a vector of shape (1000,), we will compute the correlation between all gradients across all movies and runs, resulting in a matrix of shape (number of gradients, number of gradients)
all_gradients = []
all_eigenvalues = []
for movie in movies_names:
    if movie in gradients:
        for run in range(len(gradients[movie])):
            all_gradients.append(gradients[movie][run])
            all_eigenvalues.append(eigenvalues[movie][run])
    else:
        print(f"Warning: {movie} gradients not found, skipping...")


## turn it into a numpy array
all_gradients = np.stack(all_gradients)
all_eigenvalues = np.stack(all_eigenvalues)
# check the shape of all_gradients
print(f"all_gradients shape: {all_gradients.shape}")
print(f"all_eigenvalues shape: {all_eigenvalues.shape}")
labels = []
for movie in movies_names:
    for run in range(connectivity_matrices[movie].shape[0]):
        labels.append(f"{movie}_{run}")
np.savez_compressed(f'all_gradients_tribe.npz',all_gradients=all_gradients,labels=labels,eigenvalues=all_eigenvalues)
## reshape all_gradients to be of shape (number of gradients, number of features) for correlation computation
all_gradients = all_gradients.reshape(all_gradients.shape[0], -1)    
print(f"all_gradients shape: {all_gradients.shape}")
## using sklearn pairwise cosine similarity

correlation_matrix = cosine_similarity(all_gradients)

# just convert the keys of fmri to labels for the heatmap
labels_matrix = []
for k in fmri.keys():
    labels_matrix.append(k)

plt.figure(figsize=(10, 8))
plot_matrix(correlation_matrix, figure=(10, 8), cmap='coolwarm',reorder=True,labels=labels_matrix,vmin=-1, vmax=1,title=f'Cosine Similarity between gradients - TRIBEv2 model')
plt.savefig(f'cosine_similarity_gradients_tribe.png')
plt.close()


### use scipy to compute the hierarchical clustering of the gradients based on their correlation, and plot a dendrogram 
# Compute the linkage matrix using the correlation matrix
linked = linkage(1 - correlation_matrix, 'ward')
# Plot the dendrogram
plt.figure(figsize=(10, 8))
dendrogram(linked, labels=labels_matrix, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title(f'Hierarchical Clustering of gradients - TRIBEv2 model')
plt.savefig(f'hierarchical_clustering_gradients_tribe.png')
plt.close()

## cut the dendrogram to get clusters of gradients, and print the clusters for 4 clusters

# Cut the dendrogram to get 4 clusters
clusters = fcluster(linked, 4, criterion='maxclust')
# Print the clusters
for i in range(1, 5):
    print(f"Cluster {i}:")
    for j in range(len(clusters)):
        if clusters[j] == i:
            print(labels_matrix[j])