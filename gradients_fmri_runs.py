import numpy as np
from nilearn.connectome import ConnectivityMeasure
from brainspace.gradient import GradientMaps
from utils.fmri_utils import load_fmri
import argparse
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018, load_fsaverage
from nilearn.plotting import plot_stat_map, plot_surf_stat_map, plot_surf_stat_map
from sklearn.metrics.pairwise import cosine_similarity
from nilearn.plotting import plot_matrix
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from bct import threshold_proportional

fsaverage = load_fsaverage()


parser = argparse.ArgumentParser(description='Load and analyze fMRI data')
parser.add_argument('--subject', type=int, default=1, help='Subject number (default: 1)')
parser.add_argument('--basedir', type=str, default='/home/nfarrugi/Documents/git/', help='Base directory path')
parser.add_argument('--average', action='store_true', help='Average the fMRI responses across the two repeats for each movie (default: False)')
args = parser.parse_args()

subject = args.subject
basedir = args.basedir

## prepare the masker for the Schaefer atlas
schaefer_atlas = fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7)
masker = NiftiLabelsMasker(labels_img=schaefer_atlas['maps'], standardize=True, memory='nilearn_cache')
masker.fit()

# Load the fMRI responses
fmri = load_fmri(basedir, subject, average=args.average)

# Print all available movies
print(f"Subject {subject} fMRI movies splits name and shape:")
for key, value in fmri.items():
    print(key + " " + str(value.shape))

# movie names 
movies_names = ['bourne', 'wolf', 'figures', 'life']

## regroup fMRI time series for each movie
fmri_stack = {}
for movie in movies_names:
    keys_movie = [key for key in fmri.keys() if movie in key]
    fmri_stack[movie] = [fmri[key] for key in keys_movie]


## Compute sets of functional connectivity matrix for each movie using nilearn ConnectivityMeasure, but keeping a matrix per run. Alignement per movie
gradients = {}

connectivity_matrices = {}
for movie in movies_names:
    # Create a ConnectivityMeasure object for correlation
    connectivity_measure = ConnectivityMeasure(kind='correlation',standardize='zscore_sample')
    # Compute the functional connectivity matrix
    connectivity_matrix = connectivity_measure.fit_transform(fmri_stack[movie])

    # Threshold the connectivity matrix to keep only the top 10% of connections
    connectivity_matrix = np.stack([threshold_proportional(cc, 0.1) for cc in connectivity_matrix])
    connectivity_matrices[movie] = connectivity_matrix
    print(f"{movie} affinity matrix shape: {connectivity_matrix.shape}")

    ## Compute gradients for each movie using BrainSpace GradientMaps
    gm = GradientMaps(n_components=10, random_state=0,alignment='procrustes', kernel='normalized_angle')

    # if the movie is bourne, compute its gradients and save them as reference for the other movies
    if movie == 'bourne':
        gm.fit([c for c in connectivity_matrix])
        gradients[movie] = gm.aligned_
        refgradients = gm.aligned_[0] # take the first run as reference for alignment
        print(f"Reference {movie} gradients shapes: {refgradients.shape}")
    else:
        gm.fit([c for c in connectivity_matrix],reference=refgradients)
        gradients[movie] = gm.aligned_
    print(f"{movie} gradients shapes: {[g.shape for g in gradients[movie]]}")



### visualize similarity between gradients across all movies and runs using a heatmap of the correlation between the gradients

# Compute the correlation matrix between the gradients of all movies and runs
# each gradient is a vector of shape (1000,), we will compute the correlation between all gradients across all movies and runs, resulting in a matrix of shape (number of gradients, number of gradients)
all_gradients = []
for movie in movies_names:
    for run in range(connectivity_matrices[movie].shape[0]):
        all_gradients.append(gradients[movie][run])


## turn it into a numpy array
all_gradients = np.stack(all_gradients)
# check the shape of all_gradients
print(f"all_gradients shape: {all_gradients.shape}")

## reshape all_gradients to be of shape (number of gradients, number of features) for correlation computation
all_gradients = all_gradients.reshape(all_gradients.shape[0], -1)
## using sklearn pairwise cosine similarity

correlation_matrix = cosine_similarity(all_gradients)

# just convert the keys of fmri to labels for the heatmap
labels_matrix = []
for k in fmri.keys():
    labels_matrix.append(k)

plt.figure(figsize=(10, 8))
plot_matrix(correlation_matrix, figure=(10, 8), cmap='coolwarm',reorder=True,labels=labels_matrix,vmin=-1, vmax=1,title=f'Cosine Similarity between gradients - subject {subject}')
plt.savefig(f'cosine_similarity_gradients_subject_{subject}.png')
plt.close()


### use scipy to compute the hierarchical clustering of the gradients based on their correlation, and plot a dendrogram 
# Compute the linkage matrix using the correlation matrix
linked = linkage(1 - correlation_matrix, 'ward')
# Plot the dendrogram
plt.figure(figsize=(10, 8))
dendrogram(linked, labels=labels_matrix, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title(f'Hierarchical Clustering of gradients - subject {subject}')
plt.savefig(f'hierarchical_clustering_gradients_subject_{subject}.png')
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