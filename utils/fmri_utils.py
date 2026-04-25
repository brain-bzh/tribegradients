
import os
import numpy as np
import h5py




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