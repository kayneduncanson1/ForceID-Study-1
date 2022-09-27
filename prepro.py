import numpy as np
from Utils import load_channel_raw
from PrePro import clip_filt_norm

"""This script does all of the pre-processing except for concatenation and standardization, which is done in main.py."""

# Load raw sigs of each component and each force platform (load_channel_raw returns object of shape (N, C, L):
# fx_fp1_raw = load_channel_raw('./Fx_FP1_raw.xlsx')
# fy_fp1_raw = load_channel_raw('./Fy_FP1_raw.xlsx')
# fz_fp1_raw = load_channel_raw('./Fz_FP1_raw.xlsx')
# mx_fp1_raw = load_channel_raw('./Mx_FP1_raw.xlsx')
# my_fp1_raw = load_channel_raw('./My_FP1_raw.xlsx')
# mz_fp1_raw = load_channel_raw('./Mz_FP1_raw.xlsx')
# cx_fp1_raw = load_channel_raw('./Cx_FP1_raw.xlsx')
# cy_fp1_raw = load_channel_raw('./Cy_FP1_raw.xlsx')
#
# fx_fp2_raw = load_channel_raw('./Fx_FP2_raw.xlsx')
# fy_fp2_raw = load_channel_raw('./Fy_FP2_raw.xlsx')
# fz_fp2_raw = load_channel_raw('./Fz_FP2_raw.xlsx')
# mx_fp2_raw = load_channel_raw('./Mx_FP2_raw.xlsx')
# my_fp2_raw = load_channel_raw('./My_FP2_raw.xlsx')
# mz_fp2_raw = load_channel_raw('./Mz_FP2_raw.xlsx')
# cx_fp2_raw = load_channel_raw('./Cx_FP2_raw.xlsx')
# cy_fp2_raw = load_channel_raw('./Cy_FP2_raw.xlsx')

# Concatenate components along the channel axis:
# sigs_all_fp1_raw = np.concatenate((fx_fp1_raw, fy_fp1_raw, fz_fp1_raw, mx_fp1_raw, my_fp1_raw, mz_fp1_raw,
#                                    cx_fp1_raw, cy_fp1_raw), axis=1)
# sigs_all_fp2_raw = np.concatenate((fx_fp2_raw, fy_fp2_raw, fz_fp2_raw, mx_fp2_raw, my_fp2_raw, mz_fp2_raw,
#                                    cx_fp2_raw, cy_fp2_raw), axis=1)

# Instead of loading the channels from Excel every time as above, we suggest saving the ds objects to numpy arrays for
# for future use:
# np.save('./sigs_all_fp1_raw.npy', sigs_all_fp1_raw)
# np.save('./sigs_all_fp2_raw.npy', sigs_all_fp2_raw)

# And to load them if previously saved:
sigs_all_fp1_raw = np.load('./sigs_all_fp1_raw.npy', allow_pickle=True)
sigs_all_fp2_raw = np.load('./sigs_all_fp2_raw.npy', allow_pickle=True)

# Clip, filter, and time normalize all sigs from each force platform. Note that Vicon outputs are applied forces not
# reaction forces. That is why the threshold is set to -50N (N = Newtons) instead of 50N:
fz_thresh = -50 # N.
interp_len = 300
cutoff_freq = 30 # Hz
sampling_rate = 2000 # Hz

sigs_all_fp1 = clip_filt_norm(sigs_all_fp1_raw, fz_thresh, cutoff_freq, sampling_rate, interp_len)
sigs_all_fp2 = clip_filt_norm(sigs_all_fp2_raw, fz_thresh, cutoff_freq, sampling_rate, interp_len)

# Save the ds as objects to load into main.py:
np.save('./sigs_all_fp1_pro.npy', sigs_all_fp1)
np.save('./sigs_all_fp2_pro.npy', sigs_all_fp2)



