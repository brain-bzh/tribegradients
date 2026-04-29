# %%
import pandas as pd
import numpy as np
import argparse
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import iter_img
from nilearn.datasets import fetch_atlas_schaefer_2018, load_fsaverage
from nilearn.plotting import  plot_surf_stat_map, plot_surf_stat_map
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.surface import SurfaceImage
fsaverage = load_fsaverage()

parser = argparse.ArgumentParser(description='Visualize gradients for a specific subject')
parser.add_argument('--subject', type=int, default=2, help='Subject number (default: 2)')
args = parser.parse_args()

subject = args.subject

## if select is a number, then subject = sub-0{subject}, else subject = 'tribe'
if subject > 0:
    subject = f'subject_{subject}'
else:
    subject = 'tribe'

## prepare the masker for the Schaefer atlas
schaefer_atlas = fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7)
labelsmasker = NiftiLabelsMasker(labels_img=schaefer_atlas['maps'], standardize=True, memory='nilearn_cache')
labelsmasker.fit()

labels = schaefer_atlas.labels  # This is a list of byte strings

# Convert byte strings to regular strings and extract the network name
# Example: '7Networks_LH_Vis_1' -> 'Vis'
network_names = [label.split('_')[2] for label in labels[1:]]  # Skip the first label which is usually 'Background'
assert len(network_names) == 1000  # Should be 1000
print(pd.Series(network_names).unique())  # Check the format of the split labels
# Create a mapping DataFrame
mapping_df = pd.DataFrame({
    'parcel_index': range(1000),
    'network': network_names
})


gm=np.load(f'all_gradients_{subject}.npz',allow_pickle=True)['all_gradients']
movielabels=np.load(f'all_gradients_{subject}.npz',allow_pickle=True)['labels']
eigenvalues=np.load(f'all_gradients_{subject}.npz',allow_pickle=True)['eigenvalues']
print(gm.shape)  # (number of gradients, number of features)


### eigenvalues visualization as a boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(data=eigenvalues, orient="h")
plt.yticks(ticks=np.arange(eigenvalues.shape[1]), labels=[f'Gradient {i+1}' for i in range(eigenvalues.shape[1])])
plt.xlabel('Eigenvalue')
plt.title(f'Distribution of Eigenvalues across Movies and Runs for {subject}')
plt.savefig(f'eigenvalues_boxplot_{subject}.png')
plt.show()

average = np.mean(gm, axis=0)
# Visualize this as 10 brain stat maps, one for each gradient, using nilearn's plotting functions

avg_img = labelsmasker.inverse_transform(average.T)
gradient_img = SurfaceImage.from_volume(mesh=fsaverage["pial"], volume_img=avg_img)

fig,ax = plt.subplots(4,4,figsize=(10,10),subplot_kw={'projection': '3d'})
curcmap = "viridis_r"
for i,cur_img in enumerate(iter_img(gradient_img)):
    if i==4:
        break
    

    plot_surf_stat_map(
        stat_map=cur_img,
        view="lateral",
        hemi='left',
        threshold=None,
        colorbar = False,
        cmap=curcmap,
        axes=ax[i, 0],
        figure=fig
    )
    plot_surf_stat_map(
        stat_map=cur_img,
        view="medial",
        hemi='left',
        threshold=None,
        colorbar = False,
        cmap=curcmap,
        axes=ax[i, 1],
        figure=fig
    )

    plot_surf_stat_map(
        stat_map=cur_img,
        view="lateral",
        hemi='right',
        threshold=None,
        colorbar = False,
        cmap=curcmap,
        axes=ax[i, 2],
        figure=fig
    )
    plot_surf_stat_map(
        stat_map=cur_img,
        view="medial",
        hemi='right',
        threshold=None,
        cmap=curcmap,
        colorbar = False,
        axes=ax[i, 3],
        figure=fig
    )

plt.suptitle(f'Average across all runs and movies for {subject}')
plt.savefig(f'average_gradient_{subject}.png')
plt.close()


# for each gradient (axis = 2), compute the variance across samples (axis = 0)
variance = np.var(gm, axis=0)
# Visualize this as 10 brain stat maps, one for each gradient, using nilearn's plotting functions

var_img = labelsmasker.inverse_transform(variance.T)
gradient_img = SurfaceImage.from_volume(mesh=fsaverage["pial"], volume_img=var_img)

## vmin and vmax for the variance maps, to have a common color scale across gradients
vmin = np.min(variance)
vmax = np.max(variance)

fig,ax = plt.subplots(4,4,figsize=(10,10),subplot_kw={'projection': '3d'})
curcmap = "hot"
for i,cur_img in enumerate(iter_img(gradient_img)):
    if i==4:
        break
    

    plot_surf_stat_map(
        stat_map=cur_img,
        view="lateral",
        hemi='left',
        vmin=vmin,
        vmax=vmax,
        threshold=None,
        colorbar = False,
        cmap=curcmap,
        axes=ax[i, 0],
        figure=fig
    )
    plot_surf_stat_map(
        stat_map=cur_img,
        view="medial",
        hemi='left',
        threshold=None,
        vmin=vmin,
        vmax=vmax,
        colorbar = False,
        cmap=curcmap,
        axes=ax[i, 1],
        figure=fig
    )

    plot_surf_stat_map(
        stat_map=cur_img,
        view="lateral",
        hemi='right',
        vmin=vmin,
        vmax=vmax,
        threshold=None,
        colorbar = False,
        cmap=curcmap,
        axes=ax[i, 2],
        figure=fig
    )
    plot_surf_stat_map(
        stat_map=cur_img,
        view="medial",
        hemi='right',
        threshold=None,
        vmin=vmin,
        vmax=vmax,
        cmap=curcmap,
        colorbar = True,
        axes=ax[i, 3],
        figure=fig
    )

plt.suptitle(f'Variance across all runs and movies for {subject}')
plt.savefig(f'variance_gradient_{subject}.png')
plt.close()





