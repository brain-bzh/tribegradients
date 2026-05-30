
import os
import numpy as np
import h5py
import re
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import (
    fetch_atlas_surf_destrieux,
    load_fsaverage,
)
from nilearn.maskers import SurfaceLabelsMasker
from nilearn.surface import SurfaceImage

from nilearn.datasets import load_fsaverage_data
from nilearn.surface import load_surf_data
from nilearn.datasets import load_fsaverage

schaefer_1000_surf = load_surf_data('./data/annotations/lh.Schaefer2018_1000Parcels_7Networks_order.annot')
fsaverage = load_fsaverage("fsaverage5")

labels_img = SurfaceImage(
    mesh=fsaverage["pial"],
    data={
        "left": load_surf_data('./data/annotations/lh.Schaefer2018_1000Parcels_7Networks_order.annot'),
        "right": load_surf_data('./data/annotations/rh.Schaefer2018_1000Parcels_7Networks_order.annot'),
    },
)

labels_masker = SurfaceLabelsMasker(
    labels_img=labels_img,  verbose=1
).fit()



def load_fmri(root_data_dir, subject,average=True):
    """
    Load the fMRI responses for the selected subject.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    subject : int
        Subject used to train and validate the encoding model.

    Returns
    -------
    fmri : dict
        Dictionary containing the  fMRI responses.

    """

    fmri = {}

    ### Load the fMRI responses for Movie10 ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the the fMRI responses
    fmri_movie10 = h5py.File(fmri_dir, 'r')
    for key, val in fmri_movie10.items():
        fmri[key[13:]] = val[:].astype(np.float32)
    del fmri_movie10
    if average:
        # Average the fMRI responses across the two repeats for 'figures'
        keys_all = fmri.keys()
        figures_splits = 12
        for s in range(figures_splits):
            movie = 'figures' + format(s+1, '02')
            keys_movie = [rep for rep in keys_all if movie in rep]
            fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
            del fmri[keys_movie[0]]
            del fmri[keys_movie[1]]
        # Average the fMRI responses across the two repeats for 'life'
        keys_all = fmri.keys()
        life_splits = 5
        for s in range(life_splits):
            movie = 'life' + format(s+1, '02')
            keys_movie = [rep for rep in keys_all if movie in rep]
            fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
            del fmri[keys_movie[0]]
            del fmri[keys_movie[1]]

    ### Output ###
    return fmri

def load_chunk_predictions(npz_path, movie_name):
    data = np.load(npz_path, allow_pickle=True)
    preds = data['preds']
    segments = data['segments']

    
    chunk_predictions = {}
    for i, segment in enumerate(segments):
        timeline = segment.timeline
        match = re.search(r'chunk=(\d+)', timeline)
        if not match:
            raise ValueError(f'No chunk number found in timeline: {timeline}')
        chunk_number = int(match.group(1))
        chunk_predictions.setdefault(chunk_number, []).append(preds[i])

    chunk_predictions = {k: np.array(v) for k, v in chunk_predictions.items()}
    return {f'{movie_name}{k:02d}': v for k, v in chunk_predictions.items()}

def get_masked_data(surfdata):
    fsaverage_data = load_fsaverage_data()
    data_dict = {
        "left": surfdata[:, :10242].T,
        "right": surfdata[:, 10242:20484].T,
    }

    surfimg = SurfaceImage(fsaverage_data.mesh, data_dict)

    fsaverage = load_fsaverage("fsaverage5")

    rh_data = load_surf_data('./data/annotations/rh.Schaefer2018_1000Parcels_7Networks_order.annot')
    lh_data = load_surf_data('./data/annotations/lh.Schaefer2018_1000Parcels_7Networks_order.annot')

    ## add 500 to non zero values in rh_data to make them distinct from lh_data
    rh_data[rh_data != 0] += 500

    labels_img = SurfaceImage(
        mesh=fsaverage["pial"],
        data={
            "left": lh_data,
            "right": rh_data,
        },
    )

    labels_masker = SurfaceLabelsMasker(
        labels_img=labels_img,  verbose=1
    ).fit()

    return labels_masker.transform(surfimg)