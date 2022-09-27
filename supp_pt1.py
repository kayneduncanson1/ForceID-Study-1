import numpy as np
import itertools
import os
import pandas as pd
from sklearn import preprocessing
import torch
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
from PrePro import sep_eight_channels, standardize
from DataLoaders import DataLoaderLabels, DataLoaderNoLabels, init_data_loaders
from Models import FCNN, CNN, CLSTMNN, CBiLSTMNN, CTNN
from Losses import BatchHardTripletSuppPt1
from TrainEval import train_val_supp_pt1, test_supp_pt1
from Utils import load_channel_pro, split_trial_names_by_ID_and_sess, generate_cv_folds, get_min_sample_count

"""This script shows how to run the experiments included in Supp Material - Section I (except the 'All' condition).
Please see '..._supp_pt1' funcs within the TrainEval package and the 'BatchHardTripletSuppPt1' class within the Losses
package for further details."""

# Get trial metadata from sample spreadsheet:
labels = pd.read_excel('./Cx_FP1_pro.xlsx', usecols=[0]).values.squeeze()
session_nos = pd.read_excel('./Cx_FP1_pro.xlsx', usecols=[1]).values.squeeze()
trial_nos = pd.read_excel('./Cx_FP1_pro.xlsx', usecols=[2]).values.squeeze()
speeds = pd.read_excel('./Cx_FP1_pro.xlsx', usecols=[3]).values.squeeze()

# Load pre-processed dataset by channel:
# fx_fp1 = load_channel_pro('./Fx_FP1_pro.xlsx')
# fy_fp1 = load_channel_pro('./Fy_FP1_pro.xlsx')
# fz_fp1 = load_channel_pro('./Fz_FP1_pro.xlsx')
# mx_fp1 = load_channel_pro('./Mx_FP1_pro.xlsx')
# my_fp1 = load_channel_pro('./My_FP1_pro.xlsx')
# mz_fp1 = load_channel_pro('./Mz_FP1_pro.xlsx')
# cx_fp1 = load_channel_pro('./Cx_FP1_pro.xlsx')
# cy_fp1 = load_channel_pro('./Cy_FP1_pro.xlsx')
#
# fx_fp2 = load_channel_pro('./Fx_FP2_pro.xlsx')
# fy_fp2 = load_channel_pro('./Fy_FP2_pro.xlsx')
# fz_fp2 = load_channel_pro('./Fz_FP2_pro.xlsx')
# mx_fp2 = load_channel_pro('./Mx_FP2_pro.xlsx')
# my_fp2 = load_channel_pro('./My_FP2_pro.xlsx')
# mz_fp2 = load_channel_pro('./Mz_FP2_pro.xlsx')
# cx_fp2 = load_channel_pro('./Cx_FP2_pro.xlsx')
# cy_fp2 = load_channel_pro('./Cy_FP2_pro.xlsx')

# Concatenate channels:
# sigs_all_fp1 = np.concatenate((fx_fp1,
#                                fy_fp1,
#                                fz_fp1,
#                                mx_fp1,
#                                my_fp1,
#                                mz_fp1,
#                                cx_fp1,
#                                cy_fp1), axis=1)
#
# sigs_all_fp2 = np.concatenate((fx_fp2,
#                                fy_fp2,
#                                fz_fp2,
#                                mx_fp2,
#                                my_fp2,
#                                mz_fp2,
#                                cx_fp2,
#                                cy_fp2), axis=1)

# Instead of loading the channels from Excel every time as above, we suggest saving the ds objects to numpy arrays for
# for future use:
# np.save('./sigs_all_fp1_pro.npy', sigs_all_fp1)
# np.save('./sigs_all_fp2_pro.npy', sigs_all_fp2)

# And to load them if previously saved:
sigs_all_fp1 = np.load('./sigs_all_fp1_pro.npy', allow_pickle=True)
sigs_all_fp2 = np.load('./sigs_all_fp2_pro.npy', allow_pickle=True)

# Concatenate sigs from each platform length-wise:
sigs_all_cat = np.concatenate((sigs_all_fp1, sigs_all_fp2), axis=2)

# Generate unique names for each trial according to label, session, trial, and speed:
trial_names = np.array(['_'.join([format(labels[i], '03d'), str(session_nos[i]),
                                  format(trial_nos[i], '02d'), speeds[i]]) for i in range(labels.shape[0])])

# Load train, val, and test sets for each cross-validation fold. These were previously defined in main.py and saved
# to ./Results:
# # D_AS_AF:
trial_names_D_AS_AF_s1_tr = np.load('./Results/trial_names_D_AS_AF_s1_tr.npy', allow_pickle=True)
trial_names_D_AS_AF_s1_va = np.load('./Results/trial_names_D_AS_AF_s1_va.npy', allow_pickle=True)
trial_names_D_AS_AF_s1_te = np.load('./Results/trial_names_D_AS_AF_s1_te.npy', allow_pickle=True)
trial_names_D_AS_AF_s2_tr = np.load('./Results/trial_names_D_AS_AF_s2_tr.npy', allow_pickle=True)
trial_names_D_AS_AF_s2_va = np.load('./Results/trial_names_D_AS_AF_s2_va.npy', allow_pickle=True)
trial_names_D_AS_AF_s2_te = np.load('./Results/trial_names_D_AS_AF_s2_te.npy', allow_pickle=True)

# # D_AS_SF:
trial_names_D_AS_SF_s1_tr = np.load('./Results/trial_names_D_AS_SF_s1_tr.npy', allow_pickle=True)
trial_names_D_AS_SF_s1_va = np.load('./Results/trial_names_D_AS_SF_s1_va.npy', allow_pickle=True)
trial_names_D_AS_SF_s1_te = np.load('./Results/trial_names_D_AS_SF_s1_te.npy', allow_pickle=True)
trial_names_D_AS_SF_s2_tr = np.load('./Results/trial_names_D_AS_SF_s2_tr.npy', allow_pickle=True)
trial_names_D_AS_SF_s2_va = np.load('./Results/trial_names_D_AS_SF_s2_va.npy', allow_pickle=True)
trial_names_D_AS_SF_s2_te = np.load('./Results/trial_names_D_AS_SF_s2_te.npy', allow_pickle=True)

# # D_PS_AF:
trial_names_D_PS_AF_s1_tr = np.load('./Results/trial_names_D_PS_AF_s1_tr.npy', allow_pickle=True)
trial_names_D_PS_AF_s1_va = np.load('./Results/trial_names_D_PS_AF_s1_va.npy', allow_pickle=True)
trial_names_D_PS_AF_s1_te = np.load('./Results/trial_names_D_PS_AF_s1_te.npy', allow_pickle=True)
trial_names_D_PS_AF_s2_tr = np.load('./Results/trial_names_D_PS_AF_s2_tr.npy', allow_pickle=True)
trial_names_D_PS_AF_s2_va = np.load('./Results/trial_names_D_PS_AF_s2_va.npy', allow_pickle=True)
trial_names_D_PS_AF_s2_te = np.load('./Results/trial_names_D_PS_AF_s2_te.npy', allow_pickle=True)

# # D_PS_SF:
trial_names_D_PS_SF_s1_tr = np.load('./Results/trial_names_D_PS_SF_s1_tr.npy', allow_pickle=True)
trial_names_D_PS_SF_s1_va = np.load('./Results/trial_names_D_PS_SF_s1_va.npy', allow_pickle=True)
trial_names_D_PS_SF_s1_te = np.load('./Results/trial_names_D_PS_SF_s1_te.npy', allow_pickle=True)
trial_names_D_PS_SF_s2_tr = np.load('./Results/trial_names_D_PS_SF_s2_tr.npy', allow_pickle=True)
trial_names_D_PS_SF_s2_va = np.load('./Results/trial_names_D_PS_SF_s2_va.npy', allow_pickle=True)
trial_names_D_PS_SF_s2_te = np.load('./Results/trial_names_D_PS_SF_s2_te.npy', allow_pickle=True)

le = preprocessing.LabelEncoder()

# Set your desired results path:
results_path = './Results_supp'

# Define fixed experimental settings:
m = 0.3
criterion = BatchHardTripletSuppPt1(margin=m)
epochs = 1000
interp_len = 300

# The benchmark set-up from the main experiments was applied on D_AS_AF for these experiments:
datasets = ['D_AS_AF']
archs = ['FCNN']
batch_sizes = [512]
input_channels = ['All']

# Get all possible experimental settings:
expt_settings = list(itertools.product(*[datasets, archs, batch_sizes, input_channels]))

# Loop through expts:
for setting in expt_settings:

    # Print expt details:
    print('ds = %s' % setting[0])
    print('arch = %s' % setting[1])
    print('batch size = %s' % setting[2])
    print('input channel/s = %s' % setting[3])

    for n_samples_per_ID in np.arange(1, 9):

        # Initialize results objects to be appended each run and then and saved at the end:
        times = []
        mod_params = []
        losses_tr_all = []
        losses_va_all = []
        losses_te_all = []
        accs_tr_all = []
        accs_va_all = []
        accs_te_all = []
        embs_va_all = []
        embs_te_all = []

        # Loop through each CV fold
        for fold in range(7):

            print('Fold %s:' % (fold + 1))

            # Initialize generic trial_name variables depending on the data subset being used (to reduce repetition):
            if setting[0] == 'D_AS_AF':

                trial_names_s1_tr = trial_names_D_AS_AF_s1_tr
                trial_names_s1_va = trial_names_D_AS_AF_s1_va
                trial_names_s1_te = trial_names_D_AS_AF_s1_te
                trial_names_s2_tr = trial_names_D_AS_AF_s2_tr
                trial_names_s2_va = trial_names_D_AS_AF_s2_va
                trial_names_s2_te = trial_names_D_AS_AF_s2_te

            elif setting[0] == 'D_AS_SF':

                trial_names_s1_tr = trial_names_D_AS_SF_s1_tr
                trial_names_s1_va = trial_names_D_AS_SF_s1_va
                trial_names_s1_te = trial_names_D_AS_SF_s1_te
                trial_names_s2_tr = trial_names_D_AS_SF_s2_tr
                trial_names_s2_va = trial_names_D_AS_SF_s2_va
                trial_names_s2_te = trial_names_D_AS_SF_s2_te

            elif setting[0] == 'D_PS_AF':

                trial_names_s1_tr = trial_names_D_PS_AF_s1_tr
                trial_names_s1_va = trial_names_D_PS_AF_s1_va
                trial_names_s1_te = trial_names_D_PS_AF_s1_te
                trial_names_s2_tr = trial_names_D_PS_AF_s2_tr
                trial_names_s2_va = trial_names_D_PS_AF_s2_va
                trial_names_s2_te = trial_names_D_PS_AF_s2_te

            else:

                trial_names_s1_tr = trial_names_D_PS_SF_s1_tr
                trial_names_s1_va = trial_names_D_PS_SF_s1_va
                trial_names_s1_te = trial_names_D_PS_SF_s1_te
                trial_names_s2_tr = trial_names_D_PS_SF_s2_tr
                trial_names_s2_va = trial_names_D_PS_SF_s2_va
                trial_names_s2_te = trial_names_D_PS_SF_s2_te

            # Get the min no. of samples for any given session across all IDs in val and test sets:
            sample_count_va_min, sample_count_te_min = get_min_sample_count(trial_names_s1_va, trial_names_s1_te,
                                                                            trial_names_s2_va, trial_names_s2_te, fold)

            # Define train, val and test sets:
            trial_names_tr = np.concatenate([np.concatenate((trial_names_s1_tr[fold][i], trial_names_s2_tr[fold][i]))
                                             for i in range(trial_names_s1_tr[fold].shape[0])])

            trial_names_va = np.concatenate([np.concatenate((trial_names_s1_va[fold][i][:sample_count_va_min],
                                                             trial_names_s2_va[fold][i][:sample_count_va_min]))
                                             for i in range(trial_names_s2_va[fold].shape[0])])

            trial_names_te = np.concatenate([np.concatenate((trial_names_s1_te[fold][i][:sample_count_te_min],
                                                             trial_names_s2_te[fold][i][:sample_count_te_min]))
                                             for i in range(trial_names_s1_te[fold].shape[0])])

            labels_tr = np.array([int(name[:3]) for name in trial_names_tr])
            labels_va = np.array([int(name[:3]) for name in trial_names_va])
            labels_te = np.array([int(name[:3]) for name in trial_names_te])

            le.fit(labels_tr)
            # Transform labels to start from 0:
            labels_tr = torch.tensor(le.transform(labels_tr)).long()

            le.fit(labels_va)
            labels_va = torch.tensor(le.transform(labels_va)).long()

            le.fit(labels_te)
            labels_te = torch.tensor(le.transform(labels_te)).long()

            # Define weighted random sampler for training (with the weights being probabilities of sampling from each
            # ID):
            sample_counts_tr = np.unique(labels_tr, return_counts=True)[1]
            ID_weights_tr = 1. / torch.tensor(sample_counts_tr, dtype=torch.float32)
            sample_weights_tr = list(torch.cat([torch.linspace(ID_weights_tr[i], ID_weights_tr[i], sample_counts_tr[i])
                                                for i in range(ID_weights_tr.size(0))]))
            sampler_tr = WeightedRandomSampler(sample_weights_tr, labels_tr.size(0))

            # Obtain the indices of train, val, and test samples in the original trial names object:
            indices_tr = [np.asarray(trial_names == name).nonzero()[0] for name in trial_names_tr]
            indices_va = [np.asarray(trial_names == name).nonzero()[0] for name in trial_names_va]
            indices_te = [np.asarray(trial_names == name).nonzero()[0] for name in trial_names_te]

            # Use the above indices to return objects containing train, val, and test signals.
            sigs_tr = np.squeeze(sigs_all_cat[indices_tr], axis=1)
            sigs_va = np.squeeze(sigs_all_cat[indices_va], axis=1)
            sigs_te = np.squeeze(sigs_all_cat[indices_te], axis=1)

            # Separate each channel for standardization:
            fx_tr, fy_tr, fz_tr, mx_tr, my_tr, mz_tr, cx_tr, cy_tr = sep_eight_channels(sigs_tr)
            fx_va, fy_va, fz_va, mx_va, my_va, mz_va, cx_va, cy_va = sep_eight_channels(sigs_va)
            fx_te, fy_te, fz_te, mx_te, my_te, mz_te, cx_te, cy_te = sep_eight_channels(sigs_te)

            # Standardize each channel separately:
            fx_tr_stsd, fx_va_stsd, fx_te_stsd = standardize(fx_tr, fx_va, fx_te)
            fy_tr_stsd, fy_va_stsd, fy_te_stsd = standardize(fy_tr, fy_va, fy_te)
            fz_tr_stsd, fz_va_stsd, fz_te_stsd = standardize(fz_tr, fz_va, fz_te)
            mx_tr_stsd, mx_va_stsd, mx_te_stsd = standardize(mx_tr, mx_va, mx_te)
            my_tr_stsd, my_va_stsd, my_te_stsd = standardize(my_tr, my_va, my_te)
            mz_tr_stsd, mz_va_stsd, mz_te_stsd = standardize(mz_tr, mz_va, mz_te)
            cx_tr_stsd, cx_va_stsd, cx_te_stsd = standardize(cx_tr, cx_va, cx_te)
            cy_tr_stsd, cy_va_stsd, cy_te_stsd = standardize(cy_tr, cy_va, cy_te)

            # Define data loader and model if including 'all' channels as input (i.e., main experiments):
            if setting[3] == 'All':

                # Overwriting sigs_tr, sigs_va, and sigs_te here to save memory:
                sigs_tr = torch.cat((fx_tr_stsd, fy_tr_stsd, fz_tr_stsd, mx_tr_stsd,
                                    my_tr_stsd, mz_tr_stsd, cx_tr_stsd, cy_tr_stsd), dim=1)
                sigs_va = torch.cat((fx_va_stsd, fy_va_stsd, fz_va_stsd, mx_va_stsd,
                                    my_va_stsd, mz_va_stsd, cx_va_stsd, cy_va_stsd), dim=1)
                sigs_te = torch.cat((fx_te_stsd, fy_te_stsd, fz_te_stsd, mx_te_stsd,
                                    my_te_stsd, mz_te_stsd, cx_te_stsd, cy_te_stsd), dim=1)

                data_loader_tr, data_loader_va, data_loader_te = init_data_loaders(DataLoaderLabels,
                                                                                   DataLoaderNoLabels,
                                                                                   sigs_tr, labels_tr,
                                                                                   sigs_va, sigs_te,
                                                                                   batch_size=setting[2],
                                                                                   sampler_tr=sampler_tr)

                if setting[1] == 'FCNN':

                    mod = FCNN(in_features=4800, fc1_out=1600, fc2_out=800)

                elif setting[1] == 'CNN':

                    # nc0 = no. input channels:
                    mod = CNN(nc0=8, nc1=32, nc2=64, nc3=128, nc4=256, fc1_out=2133, fc2_out=711)

                elif setting[1] == 'CLSTMNN':

                    mod = CLSTMNN(nc0=8, nc1=32, nc2=64, nc3=128, nc4=256, lstm_layers=1, fc1_out=2133, fc2_out=711)

                # Note - we ran training, validation, and testing on CPU when using CTNN or C-Bi-LSTMNN with mini-batch
                # size 512. Depending on your GPU, you may have to do this by removing .cuda() calls in funcs within the
                # TrainEval package. In the case of C-Bi-LSTMNN, also remove the .cuda() call in the init_hidden func in
                # the model class:
                elif setting[1] == 'CTNN':

                    mod = CTNN(nc0=8, nc1=32, nc2=64, nc3=128, nc4=256, tf_layers=1, fc1_out=2133, fc2_out=711)

                else:

                    mod = CBiLSTMNN(nc0=8, nc1=32, nc2=64, nc3=128, nc4=256, lstm_layers=1, fc1_out=3200, fc2_out=800)

            # Define data loader if including single channel as input (i.e., ablations):
            elif setting[3] == 'Fx':

                data_loader_tr, data_loader_va, data_loader_te = init_data_loaders(DataLoaderLabels,
                                                                                   DataLoaderNoLabels,
                                                                                   fx_tr_stsd, labels_tr,
                                                                                   fx_va_stsd, fx_te_stsd,
                                                                                   batch_size=setting[2],
                                                                                   sampler_tr=sampler_tr)

            elif setting[3] == 'Fy':

                data_loader_tr, data_loader_va, data_loader_te = init_data_loaders(DataLoaderLabels,
                                                                                   DataLoaderNoLabels,
                                                                                   fy_tr_stsd, labels_tr,
                                                                                   fy_va_stsd, fy_te_stsd,
                                                                                   batch_size=setting[2],
                                                                                   sampler_tr=sampler_tr)

            elif setting[3] == 'Fz':

                data_loader_tr, data_loader_va, data_loader_te = init_data_loaders(DataLoaderLabels,
                                                                                   DataLoaderNoLabels,
                                                                                   fz_tr_stsd, labels_tr,
                                                                                   fz_va_stsd, fz_te_stsd,
                                                                                   batch_size=setting[2],
                                                                                   sampler_tr=sampler_tr)

            elif setting[3] == 'Mx':

                data_loader_tr, data_loader_va, data_loader_te = init_data_loaders(DataLoaderLabels,
                                                                                   DataLoaderNoLabels,
                                                                                   mx_tr_stsd, labels_tr,
                                                                                   mx_va_stsd, mx_te_stsd,
                                                                                   batch_size=setting[2],
                                                                                   sampler_tr=sampler_tr)

            elif setting[3] == 'My':

                data_loader_tr, data_loader_va, data_loader_te = init_data_loaders(DataLoaderLabels,
                                                                                   DataLoaderNoLabels,
                                                                                   my_tr_stsd, labels_tr,
                                                                                   my_va_stsd, my_te_stsd,
                                                                                   batch_size=setting[2],
                                                                                   sampler_tr=sampler_tr)

            elif setting[3] == 'Mz':

                data_loader_tr, data_loader_va, data_loader_te = init_data_loaders(DataLoaderLabels,
                                                                                   DataLoaderNoLabels,
                                                                                   mz_tr_stsd, labels_tr,
                                                                                   mz_va_stsd, mz_te_stsd,
                                                                                   batch_size=setting[2],
                                                                                   sampler_tr=sampler_tr)

            elif setting[3] == 'Cx':

                data_loader_tr, data_loader_va, data_loader_te = init_data_loaders(DataLoaderLabels,
                                                                                   DataLoaderNoLabels,
                                                                                   cx_tr_stsd, labels_tr,
                                                                                   cx_va_stsd, cx_te_stsd,
                                                                                   batch_size=setting[2],
                                                                                   sampler_tr=sampler_tr)

            else:

                data_loader_tr, data_loader_va, data_loader_te = init_data_loaders(DataLoaderLabels,
                                                                                   DataLoaderNoLabels,
                                                                                   cy_tr_stsd, labels_tr,
                                                                                   cy_va_stsd, cy_te_stsd,
                                                                                   batch_size=setting[2],
                                                                                   sampler_tr=sampler_tr)

            # Define model if including single channel as input (i.e., ablations):
            if setting[3] in ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Cx', 'Cy']:

                if setting[1] == 'FCNN':

                    mod = FCNN(in_features=600, fc1_out=300, fc2_out=150)

                # (Only thing that's different in these models compared to their counterparts above is nc0):
                elif setting[1] == 'CNN':

                    # nc0 = no. input channels:
                    mod = CNN(nc0=1, nc1=32, nc2=64, nc3=128, nc4=256, fc1_out=2133, fc2_out=711)

                elif setting[1] == 'CLSTMNN':

                    mod = CLSTMNN(nc0=1, nc1=32, nc2=64, nc3=128, nc4=256, lstm_layers=1, fc1_out=2133, fc2_out=711)

                # Note - we ran training, validation, and testing on CPU when using CTNN or C-Bi-LSTMNN with mini-batch
                # size 512. Depending on your GPU, you may have to do this by removing .cuda() calls in funcs within the
                # TrainEval package. In the case of C-Bi-LSTMNN, also remove the .cuda() call in the init_hidden func in
                # the model class:
                elif setting[1] == 'CTNN':

                    mod = CTNN(nc0=1, nc1=32, nc2=64, nc3=128, nc4=256, tf_layers=1, fc1_out=2133, fc2_out=711)

                else:

                    mod = CBiLSTMNN(nc0=1, nc1=32, nc2=64, nc3=128, nc4=256, lstm_layers=1, fc1_out=3200, fc2_out=800)

            opt = optim.Adam(mod.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

            time_tr_va, accs_tr, losses_tr, accs_va, losses_va, mod_state_dict, opt_state_dict, embs_va =\
                train_val_supp_pt1(mod, data_loader_tr, data_loader_va, labels_va, criterion, sample_count_va_min,
                                   n_samples_per_ID, opt, epochs)

            # Append train and val results to associated objects:
            times.append(time_tr_va)
            accs_tr_all.append(accs_tr)
            losses_tr_all.append(losses_tr)
            accs_va_all.append(accs_va)
            losses_va_all.append(losses_va)
            mod_params.append((mod_state_dict, opt_state_dict))
            embs_va_all.append(embs_va.cpu())
            # Wipe embs_va to save memory:
            embs_va = None

            # Load model weights that led to highest val accuracy:
            mod.load_state_dict(mod_state_dict)

            # Evaluate model on test set:
            loss_te, acc_te, embs_te = test_supp_pt1(mod, opt, data_loader_te, labels_te, criterion,
                                                     sample_count_te_min, n_samples_per_ID)

            # Append test results to associated objects:
            losses_te_all.append(loss_te)
            accs_te_all.append(acc_te)
            embs_te_all.append(embs_te.cpu())
            # Wipe embs_te to save memory:
            embs_te = None

        # Save results objects (note that the filenames now contain an extra part for n_samples_per_ID):
        np.save(os.path.join(results_path, 'times_%s_%s_%s_%s_n%s.npy' % (setting[0], setting[1], setting[2],
                                                                          setting[3], n_samples_per_ID)), times)
        np.save(os.path.join(results_path, 'losses_tr_%s_%s_%s_%s_n%s.npy' % (setting[0], setting[1], setting[2],
                                                                              setting[3], n_samples_per_ID)),
                losses_tr_all)
        np.save(os.path.join(results_path, 'losses_va_%s_%s_%s_%s_n%s.npy' % (setting[0], setting[1], setting[2],
                                                                              setting[3], n_samples_per_ID)),
                losses_va_all)
        np.save(os.path.join(results_path, 'losses_te_%s_%s_%s_%s_n%s.npy' % (setting[0], setting[1], setting[2],
                                                                              setting[3], n_samples_per_ID)),
                losses_te_all)
        np.save(os.path.join(results_path, 'accs_tr_%s_%s_%s_%s_n%s.npy' % (setting[0], setting[1], setting[2],
                                                                            setting[3], n_samples_per_ID)),
                accs_tr_all)
        np.save(os.path.join(results_path, 'accs_va_%s_%s_%s_%s_n%s.npy' % (setting[0], setting[1], setting[2],
                                                                            setting[3], n_samples_per_ID)),
                accs_va_all)
        np.save(os.path.join(results_path, 'accs_te_%s_%s_%s_%s_n%s.npy' % (setting[0], setting[1], setting[2],
                                                                            setting[3], n_samples_per_ID)),
                accs_te_all)
        torch.save(embs_va_all, os.path.join(results_path, 'embs_va_%s_%s_%s_%s_n%s.pth' % (setting[0], setting[1],
                                                                                            setting[2], setting[3],
                                                                                            n_samples_per_ID)))
        torch.save(embs_te_all, os.path.join(results_path, 'embs_te_%s_%s_%s_%s_n%s.pth' % (setting[0], setting[1],
                                                                                            setting[2], setting[3],
                                                                                            n_samples_per_ID)))
        # If you want to save model params (not recommended for every experiment due to memory requirement):
        # torch.save(mod_params, os.path.join(results_path, 'mod_params_%s_%s_%s_%s_n%s.pth' % (setting[0], setting[1],
        #                                                                                       setting[2], setting[3],
        #                                                                                       n_samples_per_ID)))
