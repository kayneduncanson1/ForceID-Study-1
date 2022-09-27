import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import torch
from sklearn.preprocessing import StandardScaler


# Clip, filter, and time normalize all sigs in the dataset from one force platform.
# Sigs object must be of shape (N, C, L):
def clip_filt_norm(sigs, fz_thresh, cutoff_freq, sampling_rate, interp_len):

    sigs_pro = []

    for i in range(sigs.shape[0]):

        # Define initial clip indices for GRFs and GRMs using -50N threshold (noting that Vicon outputs are applied
        # forces not reaction forces, hence the negative sign and the 'less than -50N condition below):
        start_1 = np.asarray(sigs[i, 2, :] < fz_thresh).nonzero()[0][0]
        end_1 = np.asarray(sigs[i, 2, :] < fz_thresh).nonzero()[0][-1]
        # Define additional margin to clip COPs (to avoid inaccuracies at low force values):
        margin = int(np.round(0.05 * (end_1 - start_1)))

        # Filter clipped GRFs and GRMs (with 20 frame buffer) and clipped COP coordinates:
        grf_grm_filt = butterworth_lowpass(sigs[i, :6, start_1 - 20:end_1 + 20], order=4,
                                           normal_cutoff=cutoff_freq / (0.5 * sampling_rate), pad_len=300)
        cop_filt = butterworth_lowpass(sigs[i, 6:, start_1 + margin:end_1 - margin], order=4,
                                       normal_cutoff=cutoff_freq / (0.5 * sampling_rate), pad_len=300)

        # Define indices where filtered Fz exceeds the threshold:
        start_2 = np.asarray(grf_grm_filt[2, :] < fz_thresh).nonzero()[0][0]
        end_2 = np.asarray(grf_grm_filt[2, :] < fz_thresh).nonzero()[0][-1]

        # Interpolate between the time points before and after Fz crosses the threshold to determine the exact time
        # points where Fz = -50N:
        interp_start = interp1d([grf_grm_filt[2, start_2 - 1], grf_grm_filt[2, start_2]], [start_2 - 1, start_2],
                                axis=0)  # TODO: Confirm that axis=0 is correct
        t_start = interp_start(fz_thresh)
        interp_end = interp1d([grf_grm_filt[2, end_2], grf_grm_filt[2, end_2 + 1]], [end_2, end_2 + 1], axis=0)
        t_end = interp_end(fz_thresh)

        # Interpolate GRFs and GRMs to contain interp_len no. of frames, using the time points defined above:
        t_grf_grm = np.arange(0, grf_grm_filt.shape[1], 1)
        interp_func = interp1d(t_grf_grm, grf_grm_filt, axis=1)
        t_new_grf_grm = np.linspace(t_start, t_end, interp_len)
        grf_grm_interp = interp_func(t_new_grf_grm)

        # Interpolate COP coordinates to contain interp_len no. of frames:
        t_cop = np.linspace(0, interp_len, cop_filt.shape[1])
        interp_func = interp1d(t_cop, cop_filt, axis=1)
        t_new_cop = np.linspace(0, interp_len, interp_len)
        cop_interp = interp_func(t_new_cop)

        # Set COP coordinates to start at (0,0):
        cop_zeroed = cop_interp - np.expand_dims(cop_interp[:, 0], axis=1)

        # Concatenate all sigs into single object:
        sigs_pro.append(np.concatenate((grf_grm_interp, cop_zeroed), axis=0))

    return np.array(sigs_pro)


# Input must be of shape (C, L):
def butterworth_lowpass(input, order, normal_cutoff, pad_len):

    # Get filter coefficients:
    b, a = butter(order, normal_cutoff)
    # Filter:
    input_filtered = filtfilt(b, a, input, axis=1, padlen=pad_len)

    return input_filtered


# Input must be of shape (N, C, L):
def sep_eight_channels(input):

    fx = input[:, 0, :]
    fy = input[:, 1, :]
    fz = input[:, 2, :]
    mx = input[:, 3, :]
    my = input[:, 4, :]
    mz = input[:, 5, :]
    cx = input[:, 6, :]
    cy = input[:, 7, :]

    return fx, fy, fz, mx, my, mz, cx, cy


# Each set must be of shape (N, L) where L is the no. of features to be standardized.
# Each output is a tensor of shape (N, C, L):
def standardize(tr_set, va_set, te_set):

    s_scaler = StandardScaler()

    s_scaler.fit(tr_set)

    # Transform each set using the mean and std of the train set:
    tr_set_standardized = torch.tensor(np.expand_dims(s_scaler.transform(tr_set),
                                                      axis=1), dtype=torch.float32)
    va_set_standardized = torch.tensor(np.expand_dims(s_scaler.transform(va_set),
                                                      axis=1), dtype=torch.float32)
    te_set_standardized = torch.tensor(np.expand_dims(s_scaler.transform(te_set),
                                                      axis=1), dtype=torch.float32)

    return tr_set_standardized, va_set_standardized, te_set_standardized
