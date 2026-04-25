import numpy as np
from nilearn.connectome import ConnectivityMeasure
from brainspace.gradient import GradientMaps
from utils.fmri_utils import load_fmri
import argparse
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018, load_fsaverage
from nilearn.plotting import plot_stat_map, plot_surf_stat_map, plot_surf_stat_map
import matplotlib.pyplot as plt


fsaverage = load_fsaverage()


parser = argparse.ArgumentParser(description='Load and analyze fMRI data')
parser.add_argument('--subject', type=int, default=1, help='Subject number (default: 1)')
parser.add_argument('--basedir', type=str, default='/home/nfarrugi/Documents/git/', help='Base directory path')
args = parser.parse_args()

subject = args.subject
basedir = args.basedir

## prepare the masker for the Schaefer atlas
schaefer_atlas = fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7)
masker = NiftiLabelsMasker(labels_img=schaefer_atlas['maps'], standardize=True, memory='nilearn_cache')
masker.fit()

# Load the fMRI responses
fmri = load_fmri(basedir, subject)

# Print all available movies
print(f"Subject {subject} fMRI movies splits name and shape:")
for key, value in fmri.items():
    print(key + " " + str(value.shape))

# movie names 
movies_names = ['bourne', 'wolf', 'figures', 'life']

## concatenate fMRI time series for each movie
fmri_concat = {}
for movie in movies_names:
    keys_movie = [key for key in fmri.keys() if movie in key]
    fmri_concat[movie] = np.concatenate([fmri[key] for key in keys_movie], axis=0)
    print(f"{movie} concatenated shape: {fmri_concat[movie].shape}")


## Compute a functional connectivity matrix for each movie using nilearn ConnectivityMeasure
connectivity_matrices = {}
for movie in movies_names:
    # Create a ConnectivityMeasure object for correlation
    connectivity_measure = ConnectivityMeasure(kind='correlation')
    # Compute the functional connectivity matrix
    connectivity_matrix = connectivity_measure.fit_transform([fmri_concat[movie]])[0]
    connectivity_matrices[movie] = connectivity_matrix
    print(f"{movie} connectivity matrix shape: {connectivity_matrix.shape}")


## Compute gradients for each movie using BrainSpace GradientMaps

gm = GradientMaps(n_components=2, random_state=0,alignment='procrustes', kernel='normalized_angle')
gm.fit([connectivity_matrices[m] for m in movies_names])

gradients = {}
for i, movie in enumerate(movies_names):    
    gradients[movie] = gm.aligned_[i]
    print(f"{movie} gradients shape: {gradients[movie].shape}")


## Visualize the gradients for each movie in a large figure with subplots

fig,ax = plt.subplots(len(movies_names), 8, figsize=(24, 8 * len(movies_names)),subplot_kw={'projection': '3d'})
from nilearn.surface import SurfaceImage

for i, movie in enumerate(movies_names):
    gradient_img1 = SurfaceImage.from_volume(mesh=fsaverage["pial"], volume_img=masker.inverse_transform(gradients[movie][:, 0]),radius=5)
    gradient_img2 = SurfaceImage.from_volume(mesh=fsaverage["pial"], volume_img=masker.inverse_transform(gradients[movie][:, 1]),radius=5)
    curcmap = "viridis_r"
    plot_surf_stat_map(
        stat_map=gradient_img1,
        view="lateral",
        hemi='left',
        threshold=None,
        colorbar = False,
        title=f"{movie} - Gradient 1",
        cmap=curcmap,
        axes=ax[i, 0],
        figure=fig
    )
    plot_surf_stat_map(
        stat_map=gradient_img1,
        view="medial",
        hemi='left',
        threshold=None,
        colorbar = False,
        title=f"{movie} - Gradient 1",
        cmap=curcmap,
        axes=ax[i, 1],
        figure=fig
    )

    plot_surf_stat_map(
        stat_map=gradient_img1,
        view="lateral",
        hemi='right',
        threshold=None,
        colorbar = False,
        title=f"{movie} - Gradient 1",
        cmap=curcmap,
        axes=ax[i, 2],
        figure=fig
    )
    plot_surf_stat_map(
        stat_map=gradient_img1,
        view="medial",
        hemi='right',
        threshold=None,
        title=f"{movie} - Gradient 1",
        cmap=curcmap,
        colorbar = False,
        axes=ax[i, 3],
        figure=fig
    )

    
    plot_surf_stat_map(
        stat_map=gradient_img2,
        view="lateral",
        hemi='left',
        threshold=None,
        colorbar = False,
        title=f"{movie} - Gradient 2",
        cmap=curcmap,
        axes=ax[i, 4],
        figure=fig
    )
    plot_surf_stat_map(
        stat_map=gradient_img2,
        view="medial",
        hemi='left',
        threshold=None,
        title=f"{movie} - Gradient 2",
        cmap=curcmap,
        colorbar = False,
        axes=ax[i, 5],
        figure=fig
    )

    plot_surf_stat_map(
        stat_map=gradient_img2,
        view="lateral",
        hemi='right',
        threshold=None,
        title=f"{movie} - Gradient 2",
        cmap=curcmap,
        colorbar = False,
        axes=ax[i, 6],
        figure=fig
    )
    plot_surf_stat_map(
        stat_map=gradient_img2,
        view="medial",
        hemi='right',
        threshold=None,
        title=f"{movie} - Gradient 2",
        cmap=curcmap,
        colorbar = False,
        axes=ax[i, 7],
        figure=fig
    )


plt.suptitle(f"Subject {subject}", fontsize=16)
    





plt.savefig(f'gradients_fmri_subject_{subject}.png', dpi=300)
plt.close()
