import os
import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, f1_score
from Utils import get_min_sample_count
from Losses import get_dist_and_masks

"""This script implements the analysis for the manuscript including computation of (test set) F1-scores, 
per-ID performance analysis, and breakdown of accuracy according to walking speed and footwear comparison. """

# Get relevant footwear metadata from metadata spreadsheet:
same_footwear_bw_sessions = pd.read_excel('./Metadata.xlsx', usecols=[10]).values.squeeze()
ids_same_footwear = np.asarray(same_footwear_bw_sessions == 'Y').nonzero()[0]
ids_cross_footwear = np.asarray(same_footwear_bw_sessions == 'N').nonzero()[0]

le = preprocessing.LabelEncoder()

# Pick a data subset (ds) on which to run the analysis:
ds = 'D_AS_AF'

get_acc_and_f1 = True
ID_analysis = True
speed_footwear_analysis = True

# This can be used to get the metrics that were included in Tables III, V (in the manuscript) and VI (in Supplemental
# Material).
if get_acc_and_f1:

    accs_all = []
    f1_scores_all = []

    for bs in ['32', '64', '128', '256', '512']: # bs = mini-batch size

        # Get file names for test embeddings from main experiments. Change the 'All' condition to a specific
        # component name (e.g., 'Fz') if you wish to run the analysis for an ablation condition:
        fnames_embs_te_all = np.array([name for name in os.listdir('./Results')
                                       if name.__contains__('embs_te_%s' % ds)
                                       and name.__contains__('%s' % bs)
                                       and name.__contains__('All')])

        # Load associated embeddings:
        embs_te_all = [torch.load('./Results/%s' % name) for name in fnames_embs_te_all]

        # Loading both val and test file names because the get_min_sample_count function below takes both as input args:
        trial_names_s1_va = np.load('./Results/trial_names_%s_s1_va.npy' % ds, allow_pickle=True)
        trial_names_s1_te = np.load('./Results/trial_names_%s_s1_te.npy' % ds, allow_pickle=True)

        trial_names_s2_va = np.load('./Results/trial_names_%s_s2_va.npy' % ds, allow_pickle=True)
        trial_names_s2_te = np.load('./Results/trial_names_%s_s2_te.npy' % ds, allow_pickle=True)

        for i in range(len(embs_te_all)):

            accs = []
            f1_scores = []

            for fold in range(7):

                # Get the min no. of samples for any given session across all IDs in val and test sets:
                sample_count_va_min, sample_count_te_min = get_min_sample_count(trial_names_s1_va, trial_names_s1_te,
                                                                                trial_names_s2_va, trial_names_s2_te,
                                                                                fold)

                # Define test set:
                trial_names_te = np.concatenate([np.concatenate((trial_names_s1_te[fold][j][:sample_count_te_min],
                                                                 trial_names_s2_te[fold][j][:sample_count_te_min]))
                                                 for j in range(trial_names_s1_te[fold].shape[0])])

                labels_te = np.array([int(name[:3]) for name in trial_names_te])

                le.fit(labels_te)
                # Transform labels to start from zero:
                labels_te = torch.tensor(le.transform(labels_te)).long()

                # Get pair-wise distance matrix and masks:
                dist, mask_pos, mask_neg = get_dist_and_masks(embs_te_all[i][fold], labels_te)

                # Reduce distance matrix to contain only session two samples as anchors (row-wise) and S1 samples as
                # positives or negatives (column-wise):
                chunk = torch.cat(torch.split(dist, int(sample_count_te_min))[1::2], dim=0)
                dist_red = torch.cat(torch.split(chunk, int(sample_count_te_min), dim=1)[::2], dim=1)

                # Split up the distance matrix according to IDs:
                dist_by_ID = torch.stack(torch.split(dist_red, int(sample_count_te_min), dim=1))

                # Get the predicted ID labels and then flatten the object:
                labels_pred = torch.argmin(dist_by_ID, dim=0).transpose(1, 0).reshape(-1)

                # Reduce the labels object to contain only session one samples and then repeat it to reflect the
                # total number of predictions made:
                labels_te_red = torch.cat(torch.split(labels_te,
                                                      int(sample_count_te_min))[::2]).repeat(sample_count_te_min)

                # Calculate mean rank-1 accuracy across all predictions:
                accs.append((labels_te_red == labels_pred).float().mean().item())

                # Binarize the ground-truth labels (i.e., labels_te) and predicted labels (i.e., labels_pred) and
                # then calculate F1-score:
                labels_te_bin = preprocessing.label_binarize(labels_te_red, classes=np.unique(labels_te))
                labels_pred_bin = preprocessing.label_binarize(labels_pred, classes=np.unique(labels_te))
                f1_scores.append(f1_score(labels_te_bin, labels_pred_bin, average='macro'))

            accs_all.append(accs)
            f1_scores_all.append(f1_scores)

    accs_all = np.array(accs_all)
    f1_scores_all = np.array(f1_scores_all)

    # Split accs and F1-scores according to bs:
    accs_bs_32 = accs_all[:len(fnames_embs_te_all)]
    accs_bs_64 = accs_all[len(fnames_embs_te_all):2 * len(fnames_embs_te_all)]
    accs_bs_128 = accs_all[2 * len(fnames_embs_te_all):3 * len(fnames_embs_te_all)]
    accs_bs_256 = accs_all[3 * len(fnames_embs_te_all):4 * len(fnames_embs_te_all)]
    accs_bs_512 = accs_all[4 * len(fnames_embs_te_all):]

    f1_scores_bs_32 = f1_scores_all[:len(fnames_embs_te_all)]
    f1_scores_bs_64 = f1_scores_all[len(fnames_embs_te_all):2 * len(fnames_embs_te_all)]
    f1_scores_bs_128 = f1_scores_all[2 * len(fnames_embs_te_all):3 * len(fnames_embs_te_all)]
    f1_scores_bs_256 = f1_scores_all[3 * len(fnames_embs_te_all):4 * len(fnames_embs_te_all)]
    f1_scores_bs_512 = f1_scores_all[4 * len(fnames_embs_te_all):]

    # Calculate the mean, min, and max for each bs (express as a percentage and round to two decimal places):
    accs_bs_32_mean = np.round(np.mean(accs_bs_32) * 100, decimals=2)
    accs_bs_64_mean = np.round(np.mean(accs_bs_64) * 100, decimals=2)
    accs_bs_128_mean = np.round(np.mean(accs_bs_128) * 100, decimals=2)
    accs_bs_256_mean = np.round(np.mean(accs_bs_256) * 100, decimals=2)
    accs_bs_512_mean = np.round(np.mean(accs_bs_512) * 100, decimals=2)

    f1_scores_bs_32_mean = np.round(np.mean(f1_scores_bs_32) * 100, decimals=2)
    f1_scores_bs_64_mean = np.round(np.mean(f1_scores_bs_64) * 100, decimals=2)
    f1_scores_bs_128_mean = np.round(np.mean(f1_scores_bs_128) * 100, decimals=2)
    f1_scores_bs_256_mean = np.round(np.mean(f1_scores_bs_256) * 100, decimals=2)
    f1_scores_bs_512_mean = np.round(np.mean(f1_scores_bs_512) * 100, decimals=2)

    accs_bs_32_min = np.round(np.min(accs_bs_32) * 100, decimals=2)
    accs_bs_64_min = np.round(np.min(accs_bs_64) * 100, decimals=2)
    accs_bs_128_min = np.round(np.min(accs_bs_128) * 100, decimals=2)
    accs_bs_256_min = np.round(np.min(accs_bs_256) * 100, decimals=2)
    accs_bs_512_min = np.round(np.min(accs_bs_512) * 100, decimals=2)

    f1_scores_bs_32_min = np.round(np.min(f1_scores_bs_32) * 100, decimals=2)
    f1_scores_bs_64_min = np.round(np.min(f1_scores_bs_64) * 100, decimals=2)
    f1_scores_bs_128_min = np.round(np.min(f1_scores_bs_128) * 100, decimals=2)
    f1_scores_bs_256_min = np.round(np.min(f1_scores_bs_256) * 100, decimals=2)
    f1_scores_bs_512_min = np.round(np.min(f1_scores_bs_512) * 100, decimals=2)

    accs_bs_32_max = np.round(np.max(accs_bs_32) * 100, decimals=2)
    accs_bs_64_max = np.round(np.max(accs_bs_64) * 100, decimals=2)
    accs_bs_128_max = np.round(np.max(accs_bs_128) * 100, decimals=2)
    accs_bs_256_max = np.round(np.max(accs_bs_256) * 100, decimals=2)
    accs_bs_512_max = np.round(np.max(accs_bs_512) * 100, decimals=2)

    f1_scores_bs_32_max = np.round(np.max(f1_scores_bs_32) * 100, decimals=2)
    f1_scores_bs_64_max = np.round(np.max(f1_scores_bs_64) * 100, decimals=2)
    f1_scores_bs_128_max = np.round(np.max(f1_scores_bs_128) * 100, decimals=2)
    f1_scores_bs_256_max = np.round(np.max(f1_scores_bs_256) * 100, decimals=2)
    f1_scores_bs_512_max = np.round(np.max(f1_scores_bs_512) * 100, decimals=2)

# ID_analysis can be used to get the IDs and accuracy values shown in Table VII (in Supplemental Material).
# Speed_footwear_analysis can be used to get the results shown in Table IV.
if ID_analysis or speed_footwear_analysis:

    # These analyses were only done on the results from mini-batch size 512 as it led to the best performance:
    bs = 512

    ids_te_all = []
    cms_all = []
    cms_norm_all = []
    counts_correct_all = []
    counts_incorrect_all = []

    num_preds_s_s_same_footwear_all = []
    num_preds_s_p_same_footwear_all = []
    num_preds_s_f_same_footwear_all = []
    num_preds_p_s_same_footwear_all = []
    num_preds_p_p_same_footwear_all = []
    num_preds_p_f_same_footwear_all = []
    num_preds_f_s_same_footwear_all = []
    num_preds_f_p_same_footwear_all = []
    num_preds_f_f_same_footwear_all = []

    num_preds_s_s_cross_footwear_all = []
    num_preds_s_p_cross_footwear_all = []
    num_preds_s_f_cross_footwear_all = []
    num_preds_p_s_cross_footwear_all = []
    num_preds_p_p_cross_footwear_all = []
    num_preds_p_f_cross_footwear_all = []
    num_preds_f_s_cross_footwear_all = []
    num_preds_f_p_cross_footwear_all = []
    num_preds_f_f_cross_footwear_all = []

    corrects_s_s_same_footwear_all = []
    corrects_s_p_same_footwear_all = []
    corrects_s_f_same_footwear_all = []
    corrects_p_s_same_footwear_all = []
    corrects_p_p_same_footwear_all = []
    corrects_p_f_same_footwear_all = []
    corrects_f_s_same_footwear_all = []
    corrects_f_p_same_footwear_all = []
    corrects_f_f_same_footwear_all = []

    corrects_s_s_cross_footwear_all = []
    corrects_s_p_cross_footwear_all = []
    corrects_s_f_cross_footwear_all = []
    corrects_p_s_cross_footwear_all = []
    corrects_p_p_cross_footwear_all = []
    corrects_p_f_cross_footwear_all = []
    corrects_f_s_cross_footwear_all = []
    corrects_f_p_cross_footwear_all = []
    corrects_f_f_cross_footwear_all = []

    # Get file names for test embeddings from main experiments. Change the 'All' condition to a specific
    # component name (e.g., 'Fz') if you wish to run the analysis for an ablation condition:
    fnames_embs_te_all = np.array([name for name in os.listdir('./Results')
                                   if name.__contains__('embs_te_%s' % ds)
                                   and name.__contains__('%s' % bs)
                                   and name.__contains__('All')])

    # Load associated embeddings:
    embs_te_all = [torch.load('./Results/%s' % name) for name in fnames_embs_te_all]

    # Loading both val and test file names because the get_min_sample_count function below takes both as input args:
    trial_names_s1_va = np.load('./Results/trial_names_%s_s1_va.npy' % ds, allow_pickle=True)
    trial_names_s1_te = np.load('./Results/trial_names_%s_s1_te.npy' % ds, allow_pickle=True)

    trial_names_s2_va = np.load('./Results/trial_names_%s_s2_va.npy' % ds, allow_pickle=True)
    trial_names_s2_te = np.load('./Results/trial_names_%s_s2_te.npy' % ds, allow_pickle=True)

    for i in range(len(embs_te_all)):

        cms = []
        cms_norm = []
        counts_correct = []
        counts_incorrect = []

        num_preds_s_s_same_footwear = 0
        num_preds_s_p_same_footwear = 0
        num_preds_s_f_same_footwear = 0
        num_preds_p_s_same_footwear = 0
        num_preds_p_p_same_footwear = 0
        num_preds_p_f_same_footwear = 0
        num_preds_f_s_same_footwear = 0
        num_preds_f_p_same_footwear = 0
        num_preds_f_f_same_footwear = 0

        num_preds_s_s_cross_footwear = 0
        num_preds_s_p_cross_footwear = 0
        num_preds_s_f_cross_footwear = 0
        num_preds_p_s_cross_footwear = 0
        num_preds_p_p_cross_footwear = 0
        num_preds_p_f_cross_footwear = 0
        num_preds_f_s_cross_footwear = 0
        num_preds_f_p_cross_footwear = 0
        num_preds_f_f_cross_footwear = 0

        corrects_s_s_same_footwear = 0
        corrects_s_p_same_footwear = 0
        corrects_s_f_same_footwear = 0
        corrects_p_s_same_footwear = 0
        corrects_p_p_same_footwear = 0
        corrects_p_f_same_footwear = 0
        corrects_f_s_same_footwear = 0
        corrects_f_p_same_footwear = 0
        corrects_f_f_same_footwear = 0

        corrects_s_s_cross_footwear = 0
        corrects_s_p_cross_footwear = 0
        corrects_s_f_cross_footwear = 0
        corrects_p_s_cross_footwear = 0
        corrects_p_p_cross_footwear = 0
        corrects_p_f_cross_footwear = 0
        corrects_f_s_cross_footwear = 0
        corrects_f_p_cross_footwear = 0
        corrects_f_f_cross_footwear = 0

        for fold in range(7):

            # Get the min no. of samples for any given session across all IDs in val and test sets:
            sample_count_va_min, sample_count_te_min = get_min_sample_count(trial_names_s1_va, trial_names_s1_te,
                                                                            trial_names_s2_va, trial_names_s2_te, fold)

            # Define test set:
            trial_names_te = np.concatenate([np.concatenate((trial_names_s1_te[fold][j][:sample_count_te_min],
                                                             trial_names_s2_te[fold][j][:sample_count_te_min]))
                                             for j in range(trial_names_s1_te[fold].shape[0])])

            labels_te = np.array([int(name[:3]) for name in trial_names_te])

            le.fit(labels_te)

            if i == 0:

                # This is needed later:
                ids_te_all.append(le.classes_)

            # Transform labels to start from zero:
            labels_te = torch.tensor(le.transform(labels_te)).long()

            # Get pair-wise distance matrix and masks:
            dist, mask_pos, mask_neg = get_dist_and_masks(embs_te_all[i][fold], labels_te)

            # Reduce distance matrix to contain only session two samples as anchors (row-wise) and S1 samples as
            # positives or negatives (column-wise):
            chunk = torch.cat(torch.split(dist, int(sample_count_te_min))[1::2], dim=0)
            dist_red = torch.cat(torch.split(chunk, int(sample_count_te_min), dim=1)[::2], dim=1)

            # Split up the distance matrix according to IDs:
            dist_by_ID = torch.stack(torch.split(dist_red, int(sample_count_te_min), dim=1))

            # Get the predicted ID labels. As opposed to the 'get_acc_and_f1' implementation above, the object is NOT
            # flattened here because of how it is indexed into in the speed_footwear_analysis below:
            labels_pred = torch.argmin(dist_by_ID, dim=0)

            # Reduce the labels object to contain only session one samples and then repeat it to reflect the total
            # number of predictions made:
            labels_te_red = torch.cat(torch.split(labels_te,
                                                  int(sample_count_te_min))[::2]).repeat(sample_count_te_min)

            if ID_analysis:

                # Generate non-normalized and normalized confusion matrices:
                cm = np.array(confusion_matrix(labels_te_red, labels_pred.transpose(1, 0).reshape(-1)))
                cms.append(cm)

                cm_norm = cm / np.sum(cm, axis=1)
                cms_norm.append(cm_norm)

                # Get the number of correct counts per ID (diagonal elements):
                counts_correct.append(cm.diagonal())
                copy = cm.copy()
                np.fill_diagonal(copy, 0)

                # Get the number of incorrect counts per ID (sum with diagonal = 0):
                counts_incorrect.append(np.sum(copy, axis=1))

            if speed_footwear_analysis:

                # Initialize matrix with same shape as dist containing -infinity:
                neg_infinity = torch.ones_like(dist) * float('-inf')

                dist_ap = torch.where(mask_pos, dist, neg_infinity)
                dist_an = torch.where(mask_neg, dist, neg_infinity)

                # Reduce ap and an distance matrices to contain only session two samples as anchors (row-wise) and
                # session one samples as positives or negatives (column-wise). These matrices are organised such that
                # there are sample_count_min samples from session one and then sample_count_min samples from session two
                # for each ID. As such, they are split into subsets of sample_count_min, and then every second subset is
                # taken to get all samples from a given session ([1::2] is a slice containing all session two samples,
                # [::2] is a slice containing all session one samples):
                chunk = torch.cat(torch.split(dist_ap, int(sample_count_te_min))[1::2], dim=0)
                dist_ap_red = torch.cat(torch.split(chunk, int(sample_count_te_min), dim=1)[::2],
                                        dim=1)  # red means reduced

                chunk = torch.cat(torch.split(dist_an, int(sample_count_te_min))[1::2], dim=0)
                dist_an_red = torch.cat(torch.split(chunk, int(sample_count_te_min), dim=1)[::2], dim=1)

                # Remove data from original distance matrices to save memory (not used again):
                dist_ap = None
                dist_an = None

                # Get indices of valid positives and negatives for each anchor:
                indices_dist_ap_valid = torch.nonzero(dist_ap_red != float('-inf'), as_tuple=False)
                indices_dist_an_valid = torch.nonzero(dist_an_red != float('-inf'), as_tuple=False)

                # Get the trial names that were used as anchors vs. positives / negatives:
                trial_names_a = np.concatenate((np.split(trial_names_te,
                                                         trial_names_te.shape[0] / int(sample_count_te_min))[1::2]))
                trial_names_pn = np.concatenate((np.split(trial_names_te,
                                                          trial_names_te.shape[0] / int(sample_count_te_min))[::2]))

                # Include only valid positives:
                trial_names_p_valid = trial_names_pn[indices_dist_ap_valid[:, 1]]

                # Split trial_names_p_valid according to ID such that its shape matches that of labels_pred:
                trial_names_p_valid_split = np.array(np.split(trial_names_p_valid,
                                                              trial_names_p_valid.shape[0] / int(sample_count_te_min)))

                # Split labels_te_red according to ID and then transpose it such that its shape matches that of
                # labels_pred and trial_names_p_valid_split:
                labels_te_red = torch.stack(torch.split(labels_te_red, dist_ap_red.size(0))).transpose(1, 0)

                # Loop through anchors:
                for k in range(trial_names_a.shape[0]):

                    # Loop through the indices of each positive per anchor:
                    for l in range(sample_count_te_min):

                        # Check whether a given anchor is from an ID who wore the same footwear between sessions:
                        if int(trial_names_a[k][:3]) in ids_same_footwear:

                            # Assign total number of predictions (num_preds) and correct predictions based on the
                            # walking speed of each anchor compared to each valid positive:
                            if trial_names_a[k][-1] == 'S' and trial_names_p_valid_split[k, l][-1] == 'S' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_s_s_same_footwear += 1
                                corrects_s_s_same_footwear += 1

                            elif trial_names_a[k][-1] == 'S' and trial_names_p_valid_split[k, l][-1] == 'S' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_s_s_same_footwear += 1

                            elif trial_names_a[k][-1] == 'S' and trial_names_p_valid_split[k, l][-1] == 'P' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_s_p_same_footwear += 1
                                corrects_s_p_same_footwear += 1

                            elif trial_names_a[k][-1] == 'S' and trial_names_p_valid_split[k, l][-1] == 'P' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_s_p_same_footwear += 1

                            elif trial_names_a[k][-1] == 'S' and trial_names_p_valid_split[k, l][-1] == 'F' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_s_f_same_footwear += 1
                                corrects_s_f_same_footwear += 1

                            elif trial_names_a[k][-1] == 'S' and trial_names_p_valid_split[k, l][-1] == 'F' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_s_f_same_footwear += 1

                            elif trial_names_a[k][-1] == 'P' and trial_names_p_valid_split[k, l][-1] == 'S' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_p_s_same_footwear += 1
                                corrects_p_s_same_footwear += 1

                            elif trial_names_a[k][-1] == 'P' and trial_names_p_valid_split[k, l][-1] == 'S' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_p_s_same_footwear += 1

                            elif trial_names_a[k][-1] == 'P' and trial_names_p_valid_split[k, l][-1] == 'P' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_p_p_same_footwear += 1
                                corrects_p_p_same_footwear += 1

                            elif trial_names_a[k][-1] == 'P' and trial_names_p_valid_split[k, l][-1] == 'P' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_p_p_same_footwear += 1

                            elif trial_names_a[k][-1] == 'P' and trial_names_p_valid_split[k, l][-1] == 'F' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_p_f_same_footwear += 1
                                corrects_p_f_same_footwear += 1

                            elif trial_names_a[k][-1] == 'P' and trial_names_p_valid_split[k, l][-1] == 'F' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_p_f_same_footwear += 1

                            elif trial_names_a[k][-1] == 'F' and trial_names_p_valid_split[k, l][-1] == 'S' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_f_s_same_footwear += 1
                                corrects_f_s_same_footwear += 1

                            elif trial_names_a[k][-1] == 'F' and trial_names_p_valid_split[k, l][-1] == 'S' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_f_s_same_footwear += 1

                            elif trial_names_a[k][-1] == 'F' and trial_names_p_valid_split[k, l][-1] == 'P' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_f_p_same_footwear += 1
                                corrects_f_p_same_footwear += 1

                            elif trial_names_a[k][-1] == 'F' and trial_names_p_valid_split[k, l][-1] == 'P' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_f_p_same_footwear += 1

                            elif trial_names_a[k][-1] == 'F' and trial_names_p_valid_split[k, l][-1] == 'F' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_f_f_same_footwear += 1
                                corrects_f_f_same_footwear += 1

                            elif trial_names_a[k][-1] == 'F' and trial_names_p_valid_split[k, l][-1] == 'F' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_f_f_same_footwear += 1

                        # Check whether a given anchor is from an ID who wore different footwear between sessions:
                        elif int(trial_names_a[k][:3]) in ids_cross_footwear:

                            if trial_names_a[k][-1] == 'S' and trial_names_p_valid_split[k, l][-1] == 'S' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_s_s_cross_footwear += 1
                                corrects_s_s_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'S' and trial_names_p_valid_split[k, l][-1] == 'S' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_s_s_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'S' and trial_names_p_valid_split[k, l][-1] == 'P' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_s_p_cross_footwear += 1
                                corrects_s_p_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'S' and trial_names_p_valid_split[k, l][-1] == 'P' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_s_p_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'S' and trial_names_p_valid_split[k, l][-1] == 'F' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_s_f_cross_footwear += 1
                                corrects_s_f_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'S' and trial_names_p_valid_split[k, l][-1] == 'F' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_s_f_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'P' and trial_names_p_valid_split[k, l][-1] == 'S' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_p_s_cross_footwear += 1
                                corrects_p_s_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'P' and trial_names_p_valid_split[k, l][-1] == 'S' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_p_s_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'P' and trial_names_p_valid_split[k, l][-1] == 'P' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_p_p_cross_footwear += 1
                                corrects_p_p_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'P' and trial_names_p_valid_split[k, l][-1] == 'P' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_p_p_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'P' and trial_names_p_valid_split[k, l][-1] == 'F' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_p_f_cross_footwear += 1
                                corrects_p_f_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'P' and trial_names_p_valid_split[k, l][-1] == 'F' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_p_f_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'F' and trial_names_p_valid_split[k, l][-1] == 'S' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_f_s_cross_footwear += 1
                                corrects_f_s_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'F' and trial_names_p_valid_split[k, l][-1] == 'S' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_f_s_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'F' and trial_names_p_valid_split[k, l][-1] == 'P' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_f_p_cross_footwear += 1
                                corrects_f_p_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'F' and trial_names_p_valid_split[k, l][-1] == 'P' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_f_p_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'F' and trial_names_p_valid_split[k, l][-1] == 'F' and \
                                    labels_pred[k, l] == labels_te_red[k, l]:

                                num_preds_f_f_cross_footwear += 1
                                corrects_f_f_cross_footwear += 1

                            elif trial_names_a[k][-1] == 'F' and trial_names_p_valid_split[k, l][-1] == 'F' and \
                                    labels_pred[k, l] != labels_te_red[k, l]:

                                num_preds_f_f_cross_footwear += 1

        cms_all.append(cms)
        cms_norm_all.append(cms_norm)
        counts_correct_all.append(counts_correct)
        counts_incorrect_all.append(counts_incorrect)

        num_preds_s_s_same_footwear_all.append(num_preds_s_s_same_footwear)
        num_preds_s_p_same_footwear_all.append(num_preds_s_p_same_footwear)
        num_preds_s_f_same_footwear_all.append(num_preds_s_f_same_footwear)
        num_preds_p_s_same_footwear_all.append(num_preds_p_s_same_footwear)
        num_preds_p_p_same_footwear_all.append(num_preds_p_p_same_footwear)
        num_preds_p_f_same_footwear_all.append(num_preds_p_f_same_footwear)
        num_preds_f_s_same_footwear_all.append(num_preds_f_s_same_footwear)
        num_preds_f_p_same_footwear_all.append(num_preds_f_p_same_footwear)
        num_preds_f_f_same_footwear_all.append(num_preds_f_f_same_footwear)

        num_preds_s_s_cross_footwear_all.append(num_preds_s_s_cross_footwear)
        num_preds_s_p_cross_footwear_all.append(num_preds_s_p_cross_footwear)
        num_preds_s_f_cross_footwear_all.append(num_preds_s_f_cross_footwear)
        num_preds_p_s_cross_footwear_all.append(num_preds_p_s_cross_footwear)
        num_preds_p_p_cross_footwear_all.append(num_preds_p_p_cross_footwear)
        num_preds_p_f_cross_footwear_all.append(num_preds_p_f_cross_footwear)
        num_preds_f_s_cross_footwear_all.append(num_preds_f_s_cross_footwear)
        num_preds_f_p_cross_footwear_all.append(num_preds_f_p_cross_footwear)
        num_preds_f_f_cross_footwear_all.append(num_preds_f_f_cross_footwear)

        corrects_s_s_same_footwear_all.append(corrects_s_s_same_footwear)
        corrects_s_p_same_footwear_all.append(corrects_s_p_same_footwear)
        corrects_s_f_same_footwear_all.append(corrects_s_f_same_footwear)
        corrects_p_s_same_footwear_all.append(corrects_p_s_same_footwear)
        corrects_p_p_same_footwear_all.append(corrects_p_p_same_footwear)
        corrects_p_f_same_footwear_all.append(corrects_p_f_same_footwear)
        corrects_f_s_same_footwear_all.append(corrects_f_s_same_footwear)
        corrects_f_p_same_footwear_all.append(corrects_f_p_same_footwear)
        corrects_f_f_same_footwear_all.append(corrects_f_f_same_footwear)

        corrects_s_s_cross_footwear_all.append(corrects_s_s_cross_footwear)
        corrects_s_p_cross_footwear_all.append(corrects_s_p_cross_footwear)
        corrects_s_f_cross_footwear_all.append(corrects_s_f_cross_footwear)
        corrects_p_s_cross_footwear_all.append(corrects_p_s_cross_footwear)
        corrects_p_p_cross_footwear_all.append(corrects_p_p_cross_footwear)
        corrects_p_f_cross_footwear_all.append(corrects_p_f_cross_footwear)
        corrects_f_s_cross_footwear_all.append(corrects_f_s_cross_footwear)
        corrects_f_p_cross_footwear_all.append(corrects_f_p_cross_footwear)
        corrects_f_f_cross_footwear_all.append(corrects_f_f_cross_footwear)

    # Get the 20 IDs with the most incorrect predictions:
    if ID_analysis:

        # Because the same test sets were used in each experiment, the counts for a given fold (i.e., counts_..._sub
        # here) can be summed between experiments. The resultant objects contain the number of correct and incorrect
        # predictions per ID:
        counts_correct_sum_bw_expts = []
        counts_incorrect_sum_bw_expts = []

        for fold in range(7):

            counts_correct_sub = []
            counts_incorrect_sub = []

            for i in range(len(embs_te_all)):

                counts_correct_sub.append(counts_correct_all[i][fold])
                counts_incorrect_sub.append(counts_incorrect_all[i][fold])

            counts_correct_sum_bw_expts.append(np.sum(np.array(counts_correct_sub), axis=0))
            counts_incorrect_sum_bw_expts.append(np.sum(np.array(counts_incorrect_sub), axis=0))

        # Concatenate objects containing IDs and counts per ID such that each element represents an ID:
        ids_te_all = np.concatenate(ids_te_all)
        counts_correct_sum_bw_expts = np.concatenate(counts_correct_sum_bw_expts)
        counts_incorrect_sum_bw_expts = np.concatenate(counts_incorrect_sum_bw_expts)

        # Get indices that sort counts per ID in descending order and use them to sort the objects containing IDs and
        # counts per ID:
        indices_sort_counts_incorrect = np.argsort(counts_incorrect_sum_bw_expts)[::-1]
        ids_te_all_sorted = ids_te_all[indices_sort_counts_incorrect]
        counts_correct_sorted = counts_correct_sum_bw_expts[indices_sort_counts_incorrect]
        counts_incorrect_sorted = counts_incorrect_sum_bw_expts[indices_sort_counts_incorrect]

        # Get the 20 IDs with the most incorrect predictions:
        ids_top_20 = ids_te_all_sorted[:20]
        accs_top_20 = counts_correct_sorted[:20] / (counts_correct_sorted[:20] + counts_incorrect_sorted[:20])

        indices_sort_by_acc = np.argsort(accs_top_20)

        # Sort acc values and express as a percentage to two decimal places:
        accs_top_20_sorted = np.round(accs_top_20[indices_sort_by_acc] * 100, decimals=2)

        # Sort 'top 20' most incorrectly predicted IDs by acc:
        ids_top_20_sorted = ids_top_20[indices_sort_by_acc]

    if speed_footwear_analysis:

        # Get acc for each comparison:
        acc_s_s_same_footwear = np.sum(corrects_s_s_same_footwear_all) / np.sum(num_preds_s_s_same_footwear_all) * 100
        acc_p_p_same_footwear = np.sum(corrects_p_p_same_footwear_all) / np.sum(num_preds_p_p_same_footwear_all) * 100
        acc_f_f_same_footwear = np.sum(corrects_f_f_same_footwear_all) / np.sum(num_preds_f_f_same_footwear_all) * 100

        acc_s_s_cross_footwear = np.sum(corrects_s_s_cross_footwear_all) / np.sum(num_preds_s_s_cross_footwear_all)\
            * 100
        acc_p_p_cross_footwear = np.sum(corrects_p_p_cross_footwear_all) / np.sum(num_preds_p_p_cross_footwear_all)\
            * 100
        acc_f_f_cross_footwear = np.sum(corrects_f_f_cross_footwear_all) / np.sum(num_preds_f_f_cross_footwear_all)\
            * 100

        acc_s_p_same_footwear = np.sum(np.concatenate((corrects_s_p_same_footwear_all,
                                                       corrects_p_s_same_footwear_all))) /\
                                np.sum(np.concatenate((num_preds_s_p_same_footwear_all,
                                                       num_preds_p_s_same_footwear_all))) * 100
        acc_s_f_same_footwear = np.sum(np.concatenate((corrects_s_f_same_footwear_all,
                                                       corrects_f_s_same_footwear_all))) /\
                                np.sum(np.concatenate((num_preds_s_f_same_footwear_all,
                                                       num_preds_f_s_same_footwear_all))) * 100
        acc_p_f_same_footwear = np.sum(np.concatenate((corrects_p_f_same_footwear_all,
                                                       corrects_f_p_same_footwear_all))) /\
                                np.sum(np.concatenate((num_preds_p_f_same_footwear_all,
                                                       num_preds_f_p_same_footwear_all))) * 100

        acc_s_p_cross_footwear = np.sum(np.concatenate((corrects_s_p_cross_footwear_all,
                                                        corrects_p_s_cross_footwear_all))) /\
            np.sum(np.concatenate((num_preds_s_p_cross_footwear_all, num_preds_p_s_cross_footwear_all))) * 100
        acc_s_f_cross_footwear = np.sum(np.concatenate((corrects_s_f_cross_footwear_all,
                                                        corrects_f_s_cross_footwear_all))) /\
            np.sum(np.concatenate((num_preds_s_f_cross_footwear_all, num_preds_f_s_cross_footwear_all))) * 100
        acc_p_f_cross_footwear = np.sum(np.concatenate((corrects_p_f_cross_footwear_all,
                                                        corrects_f_p_cross_footwear_all))) /\
            np.sum(np.concatenate((num_preds_p_f_cross_footwear_all, num_preds_f_p_cross_footwear_all))) * 100

        # Get acc for preferred-preferred speed comparisons (regardless of footwear condition):
        acc_p_p = np.sum(np.concatenate((corrects_p_p_same_footwear_all, corrects_p_p_cross_footwear_all))) /\
            np.sum(np.concatenate((num_preds_p_p_same_footwear_all, num_preds_p_p_cross_footwear_all))) * 100

        # Get acc on same footwear comparisons:
        acc_same_footwear = np.round(np.sum(np.concatenate((corrects_s_s_same_footwear_all,
                                                            corrects_p_p_same_footwear_all,
                                                            corrects_f_f_same_footwear_all,
                                                            corrects_s_p_same_footwear_all,
                                                            corrects_p_s_same_footwear_all,
                                                            corrects_s_f_same_footwear_all,
                                                            corrects_f_s_same_footwear_all,
                                                            corrects_p_f_same_footwear_all,
                                                            corrects_f_p_same_footwear_all))) /
                                      np.sum(np.concatenate((num_preds_s_s_same_footwear_all,
                                                             num_preds_p_p_same_footwear_all,
                                                             num_preds_f_f_same_footwear_all,
                                                             num_preds_s_p_same_footwear_all,
                                                             num_preds_p_s_same_footwear_all,
                                                             num_preds_s_f_same_footwear_all,
                                                             num_preds_f_s_same_footwear_all,
                                                             num_preds_p_f_same_footwear_all,
                                                             num_preds_f_p_same_footwear_all))) * 100, decimals=2)
        # Get acc on cross footwear comparisons:
        acc_cross_footwear = np.round(np.sum(np.concatenate((corrects_s_s_cross_footwear_all,
                                                             corrects_p_p_cross_footwear_all,
                                                             corrects_f_f_cross_footwear_all,
                                                             corrects_s_p_cross_footwear_all,
                                                             corrects_p_s_cross_footwear_all,
                                                             corrects_s_f_cross_footwear_all,
                                                             corrects_f_s_cross_footwear_all,
                                                             corrects_p_f_cross_footwear_all,
                                                             corrects_f_p_cross_footwear_all))) /
                                      np.sum(np.concatenate((num_preds_s_s_cross_footwear_all,
                                                             num_preds_p_p_cross_footwear_all,
                                                             num_preds_f_f_cross_footwear_all,
                                                             num_preds_s_p_cross_footwear_all,
                                                             num_preds_p_s_cross_footwear_all,
                                                             num_preds_s_f_cross_footwear_all,
                                                             num_preds_f_s_cross_footwear_all,
                                                             num_preds_p_f_cross_footwear_all,
                                                             num_preds_f_p_cross_footwear_all))) * 100, decimals=2)

        # Get the total number of predictions for each comparison:
        num_preds_total_f_f_same_footwear = np.sum(num_preds_f_f_same_footwear_all)
        num_preds_total_s_s_same_footwear = np.sum(num_preds_s_s_same_footwear_all)
        num_preds_total_p_p_same_footwear = np.sum(num_preds_p_p_same_footwear_all)

        num_preds_total_f_f_cross_footwear = np.sum(num_preds_f_f_cross_footwear_all)
        num_preds_total_s_s_cross_footwear = np.sum(num_preds_s_s_cross_footwear_all)
        num_preds_total_p_p_cross_footwear = np.sum(num_preds_p_p_cross_footwear_all)

        num_preds_total_s_p_same_footwear = np.sum(num_preds_s_p_same_footwear_all)\
            + np.sum(num_preds_p_s_same_footwear_all)
        num_preds_total_p_f_same_footwear = np.sum(num_preds_p_f_same_footwear_all)\
            + np.sum(num_preds_f_p_same_footwear_all)
        num_preds_total_s_f_same_footwear = np.sum(num_preds_s_f_same_footwear_all)\
            + np.sum(num_preds_f_s_same_footwear_all)

        num_preds_total_s_p_cross_footwear = np.sum(num_preds_s_p_cross_footwear_all)\
            + np.sum(num_preds_p_s_cross_footwear_all)
        num_preds_total_p_f_cross_footwear = np.sum(num_preds_p_f_cross_footwear_all)\
            + np.sum(num_preds_f_p_cross_footwear_all)
        num_preds_total_s_f_cross_footwear = np.sum(num_preds_s_f_cross_footwear_all)\
            + np.sum(num_preds_f_s_cross_footwear_all)

        # Get the total number of predictions across all comparisons:
        num_preds_total = np.sum([num_preds_total_f_f_same_footwear,
                                 num_preds_total_s_s_same_footwear,
                                 num_preds_total_p_p_same_footwear,
                                 num_preds_total_s_p_same_footwear,
                                 num_preds_total_p_f_same_footwear,
                                 num_preds_total_s_f_same_footwear,
                                 num_preds_total_f_f_cross_footwear,
                                 num_preds_total_s_s_cross_footwear,
                                 num_preds_total_p_p_cross_footwear,
                                 num_preds_total_s_p_cross_footwear,
                                 num_preds_total_p_f_cross_footwear,
                                 num_preds_total_s_f_cross_footwear])
