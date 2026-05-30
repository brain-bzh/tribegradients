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

# parser = argparse.ArgumentParser(description='Visualize gradients for a specific subject')
# parser.add_argument('--subject', type=int, default=2, help='Subject number (default: 2)')
# args = parser.parse_args()

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

ind_average = mapping_df.groupby('network')['parcel_index'].apply(list)

all_gradients = {}
for subject in ['subject_1', 'subject_2', 'subject_3', 'subject_5', 'tribe']:
    gm=np.load(f'data/npz/all_gradients_{subject}.npz',allow_pickle=True)['all_gradients']
    movielabels=np.load(f'data/npz/all_gradients_{subject}.npz',allow_pickle=True)['labels']
    eigenvalues=np.load(f'data/npz/all_gradients_{subject}.npz',allow_pickle=True)['eigenvalues']
    all_gradients[subject] = []
    for network in ind_average.index:
        indices = ind_average[network]
        network_gradients = gm[:,indices, :].mean(axis=1)
        print(network_gradients.shape)
        all_gradients[subject].append(network_gradients)
    
    all_gradients[subject] = np.stack(all_gradients[subject])



# concatenate all gradients in a single numpy array of shape (1000, 4*5) 
all_gradients_concat = np.stack([all_gradients[subject] for subject in['subject_1', 'subject_2', 'subject_3', 'subject_5']])

tribe_gradients = all_gradients['tribe']

all_gradients_avg = np.swapaxes(all_gradients_concat, 0,1)
## shape is 7 networks, 4 subjects, 61 runs, 4 gradients, regroup subjects and runs 
all_gradients_avg = all_gradients_avg.reshape(7, -1, 4)  


print(f"Tribe gradients shape: {tribe_gradients.shape}") # 7 networks, 44 runs, 4 gradients
print(f"Average gradients shape: {all_gradients_avg.shape}") # 7 networks, 244 runs (4 subjects * 61 runs), 4 gradients

## we want to do a boxplot of the 7 networks for each gradient putting tribe and average side by side
fig, axes = plt.subplots(1, 4, figsize=(20, 6))

for g in range(4):
    data = []
    for n in range(7):
        for r in range(all_gradients_avg.shape[1]):
            data.append({
                'network': ind_average.index[n],
                'gradient': f'Gradient {g+1}',
                'value': all_gradients_avg[n, r, g],
                'type': 'Average'
            })
        for r in range(tribe_gradients.shape[1]):
            data.append({
                'network': ind_average.index[n],
                'gradient': f'Gradient {g+1}',
                'value': tribe_gradients[n, r, g],
                'type': 'Tribe'
            })
    df = pd.DataFrame(data)
    sns.boxplot(y='network', x='value', hue='type', data=df, ax=axes[g])
    axes[g].set_title(f'Gradient {g+1} scores')
    axes[g].set_yticklabels(axes[g].get_yticklabels(), rotation=45)
    axes[g].legend(title='Type')

plt.tight_layout()
plt.show()