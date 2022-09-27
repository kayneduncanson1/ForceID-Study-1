import numpy as np
import pandas as pd
import random


# Load raw dataset by channel:
def load_channel_raw(filepath):

    # Take columns five onward to include only the measurements.
    # Expand dims to include a channel dim (i.e., to form shape (N, C, L)):
    sigs = np.expand_dims(pd.read_excel(filepath).fillna(0).values[:, 5:], axis=1)

    return sigs


# Load pre-processed dataset by channel:
def load_channel_pro(filepath):

    # Expand dims to include a channel dim (i.e., to form shape (N, C, L)):
    sigs = np.expand_dims(pd.read_excel(filepath, usecols='F:KS').values, axis=1)

    return sigs


# Split each ds by ID and session no. and then shuffle samples from each ID:
def split_trial_names_by_ID_and_sess(ids, trial_names):

    # Separate dataset by ID and session number:
    trial_names_s1 = [[name for name in trial_names if int(name[:3]) == id_ and name[4] == '1'] for id_ in ids]
    trial_names_s2 = [[name for name in trial_names if int(name[:3]) == id_ and name[4] == '2'] for id_ in ids]

    # Shuffle samples from each ID:
    trial_names_s1_shuffled = np.array([random.sample(id_, k=len(id_)) for id_ in trial_names_s1], dtype=object)
    trial_names_s2_shuffled = np.array([random.sample(id_, k=len(id_)) for id_ in trial_names_s2], dtype=object)

    return trial_names_s1_shuffled, trial_names_s2_shuffled


# Define train, val, and test sets for seven-fold cross-validation given a set of trial names:
def generate_cv_folds(trial_names_s1, trial_names_s2, n_IDs_tr, n_IDs_te):

    trial_names_s1_tr = [trial_names_s1[:n_IDs_tr],
                         trial_names_s1[n_IDs_te:n_IDs_tr + n_IDs_te],
                         trial_names_s1[n_IDs_te * 2:],
                         np.concatenate((trial_names_s1[:n_IDs_te], trial_names_s1[n_IDs_te * 3:])),
                         np.concatenate((trial_names_s1[:n_IDs_te * 2], trial_names_s1[n_IDs_te * 4:])),
                         np.concatenate((trial_names_s1[:n_IDs_te * 3], trial_names_s1[n_IDs_te * 5:])),
                         np.concatenate((trial_names_s1[:n_IDs_te * 4], trial_names_s1[n_IDs_tr:n_IDs_te * 5],
                                         trial_names_s1[n_IDs_tr + n_IDs_te:]))]

    trial_names_s2_tr = [trial_names_s2[:n_IDs_tr],
                         trial_names_s2[n_IDs_te:n_IDs_tr + n_IDs_te],
                         trial_names_s2[n_IDs_te * 2:],
                         np.concatenate((trial_names_s2[:n_IDs_te], trial_names_s2[n_IDs_te * 3:])),
                         np.concatenate((trial_names_s2[:n_IDs_te * 2], trial_names_s2[n_IDs_te * 4:])),
                         np.concatenate((trial_names_s2[:n_IDs_te * 3], trial_names_s2[n_IDs_te * 5:])),
                         np.concatenate((trial_names_s2[:n_IDs_te * 4], trial_names_s2[n_IDs_tr:n_IDs_te * 5],
                                         trial_names_s2[n_IDs_tr + n_IDs_te:]))]

    trial_names_s1_va = [trial_names_s1[n_IDs_tr:n_IDs_tr + n_IDs_te],
                         trial_names_s1[n_IDs_tr + n_IDs_te:],
                         trial_names_s1[:n_IDs_te],
                         trial_names_s1[n_IDs_te:n_IDs_te * 2],
                         trial_names_s1[n_IDs_te * 2:n_IDs_te * 3],
                         trial_names_s1[n_IDs_te * 3:n_IDs_te * 4],
                         trial_names_s1[n_IDs_te * 4:n_IDs_tr]]

    trial_names_s2_va = [trial_names_s2[n_IDs_tr:n_IDs_tr + n_IDs_te],
                         trial_names_s2[n_IDs_tr + n_IDs_te:],
                         trial_names_s2[:n_IDs_te],
                         trial_names_s2[n_IDs_te:n_IDs_te * 2],
                         trial_names_s2[n_IDs_te * 2:n_IDs_te * 3],
                         trial_names_s2[n_IDs_te * 3:n_IDs_te * 4],
                         trial_names_s2[n_IDs_te * 4:n_IDs_tr]]

    trial_names_s1_te = [trial_names_s1[n_IDs_tr + n_IDs_te:],
                         trial_names_s1[:n_IDs_te],
                         trial_names_s1[n_IDs_te:n_IDs_te * 2],
                         trial_names_s1[n_IDs_te * 2:n_IDs_te * 3],
                         trial_names_s1[n_IDs_te * 3:n_IDs_te * 4],
                         trial_names_s1[n_IDs_te * 4:n_IDs_te * 5],
                         trial_names_s1[n_IDs_te * 5:n_IDs_tr + n_IDs_te]]

    trial_names_s2_te = [trial_names_s2[n_IDs_tr + n_IDs_te:],
                         trial_names_s2[:n_IDs_te],
                         trial_names_s2[n_IDs_te:n_IDs_te * 2],
                         trial_names_s2[n_IDs_te * 2:n_IDs_te * 3],
                         trial_names_s2[n_IDs_te * 3:n_IDs_te * 4],
                         trial_names_s2[n_IDs_te * 4:n_IDs_te * 5],
                         trial_names_s2[n_IDs_te * 5:n_IDs_tr + n_IDs_te]]

    return trial_names_s1_tr, trial_names_s1_va, trial_names_s1_te, \
        trial_names_s2_tr, trial_names_s2_va, trial_names_s2_te


# Get the min no. of samples for any given session across all IDs in val and test sets:
def get_min_sample_count(trial_names_s1_va, trial_names_s1_te, trial_names_s2_va, trial_names_s2_te, fold):

    sample_count_va_min = np.min([len(item) for item in np.concatenate((trial_names_s1_va[fold],
                                                                        trial_names_s2_va[fold]))])
    sample_count_te_min = np.min([len(item) for item in np.concatenate((trial_names_s1_te[fold],
                                                                        trial_names_s2_te[fold]))])

    return sample_count_va_min, sample_count_te_min
