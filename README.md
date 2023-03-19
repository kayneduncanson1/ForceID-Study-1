# ForceID-Study-1
Repository for the article, 'Deep Metric Learning for Scalable Gait Based Person Re-identification Using Force Platform Data'.


Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
## Dataset
This code base is set up to work with the public force platform dataset, 'ForceID Dataset A'. The dataset and its
description can be found at (https://adelaide.figshare.com/articles/dataset/ForceID_Dataset/14482980). Note that a
private version of the dataset (containing nine more participants than ForceID Dataset A) was used for most of the
experiments in the study. However, the best performing model was also applied on ForceID Dataset A and the results
will be reported in the article upon publication for the purpose of benchmarking.

## Navigation
Each package contains functions and/or classes relevant to the component of the software indicated by the package name.
Each script file utilizes relevant packages to complete part of the study workflow. Comments are included within
package __init__.py files and script files to provide additional information where necessary. Below is a list of the script files
and their overall purposes:
- LICENSE.txt contains additional information on the software license.
- prepro.py is for reading, pre-processing and then saving force platform data (in the form of time series signals) from
Excel spreadsheets. Currently, the dataset contains .xlsx files and the code is set up to read this file type. An
update to the dataset and code is forthcoming to work with .csv files because these are more universal and more
efficient to read.
- main.py contains the code to read relevant data and metadata and then complete experiments as per the methods to be
outlined in the article upon its publication. This script is set up to be somewhat modular, such that different subsets
of experiments can be completed efficiently (e.g., using certain subset(s) of the dataset, neural network
architecture(s), batch size(s), and input(s))
- supp_pt1.py is similar to main.py, but contains changes to the code where necessary to run experiments included in
Section 1 of the Supplementary Material (except the 'All' condition).
- supp_pt2.py is similar to main.py, but contains changes to the code where necessary to run the experiment for the 'All'
condition in Section 1 of the Supplementary Material.
- analysis.py contains the code to run the analyses that were conducted for the study and will be reported on in the
publication. These analyses include getting overall accuracy and F1-score metrics, accuracy per identity (ID) in the
dataset, and accuracy for different speed and footwear conditions.
- conda_env.yml contains the conda environment that was used to complete the study and are required to run the code in
this repository.

## Python packages
For reference, the packages included in the interpreter (many of which were not utilized in the above scripts)
are as follows (formatted as name, version, build, channel):  
_tflow_select             2.2.0                     eigen  
absl-py                   0.15.0             pyhd3eb1b0_0  
aiohttp                   3.8.1            py37h2bbff1b_1  
aiosignal                 1.2.0              pyhd3eb1b0_0  
astor                     0.8.1            py37haa95532_0  
async-timeout             4.0.1              pyhd3eb1b0_0  
asynctest                 0.13.0                     py_0  
attrs                     21.4.0             pyhd3eb1b0_0  
backcall                  0.2.0              pyhd3eb1b0_0  
blas                      1.0                         mkl  
blinker                   1.4              py37haa95532_0  
bottleneck                1.3.5            py37h080aedc_0  
brotli                    1.0.9                h2bbff1b_7  
brotli-bin                1.0.9                h2bbff1b_7  
brotlipy                  0.7.0           py37h2bbff1b_1003  
ca-certificates           2022.07.19           haa95532_0  
cachetools                4.2.2              pyhd3eb1b0_0  
certifi                   2022.9.14        py37haa95532_0  
cffi                      1.15.1           py37h2bbff1b_0  
chardet                   4.0.0           py37haa95532_1003  
charset-normalizer        2.0.4              pyhd3eb1b0_0  
click                     8.0.4            py37haa95532_0  
cloudpickle               2.0.0              pyhd3eb1b0_0  
colorama                  0.4.5            py37haa95532_0  
cryptography              37.0.1           py37h21b164f_0  
cudatoolkit               11.3.1               h59b6b97_2  
cycler                    0.11.0             pyhd3eb1b0_0  
cytoolz                   0.11.0           py37he774522_0  
dask-core                 2021.10.0          pyhd3eb1b0_0  
decorator                 5.1.1              pyhd3eb1b0_0  
enum34                    1.1.10                   pypi_0    pypi  
et_xmlfile                1.1.0            py37haa95532_0  
fancycompleter            0.9.1           py37h03978a9_1003    conda-forge  
fftw                      3.3.9                h2bbff1b_1  
fonttools                 4.25.0             pyhd3eb1b0_0  
freetype                  2.10.4               hd328e21_0  
frozenlist                1.2.0            py37h2bbff1b_0  
fsspec                    2022.7.1         py37haa95532_0  
gast                      0.2.2                    py37_0  
google-auth               2.6.0              pyhd3eb1b0_0  
google-auth-oauthlib      0.4.1                      py_2  
google-pasta              0.2.0              pyhd3eb1b0_0  
grpcio                    1.42.0           py37hc60d5dd_0  
h5py                      3.7.0            py37h3de5c98_0  
hdf5                      1.10.6               h1756f20_1  
icc_rt                    2019.0.0             h0cc432a_1  
icu                       58.2                 ha925a31_3  
idna                      3.3                pyhd3eb1b0_0  
imageio                   2.19.3           py37haa95532_0  
importlib-metadata        4.11.3           py37haa95532_0  
intel-openmp              2021.4.0          haa95532_3556  
ipython                   7.31.1           py37haa95532_1  
ipython_genutils          0.2.0              pyhd3eb1b0_1  
jedi                      0.18.1           py37haa95532_1  
joblib                    1.1.0              pyhd3eb1b0_0  
jpeg                      9e                   h2bbff1b_0  
keras                     2.3.1                         0  
keras-applications        1.0.8                      py_1  
keras-base                2.3.1                    py37_0  
keras-preprocessing       1.1.2              pyhd3eb1b0_0  
kiwisolver                1.4.2            py37hd77b12b_0  
lerc                      3.0                  hd77b12b_0  
libbrotlicommon           1.0.9                h2bbff1b_7  
libbrotlidec              1.0.9                h2bbff1b_7  
libbrotlienc              1.0.9                h2bbff1b_7  
libdeflate                1.8                  h2bbff1b_5  
libpng                    1.6.37               h2a8f88b_0  
libprotobuf               3.20.1               h23ce68f_0  
libtiff                   4.4.0                h8a3f274_0  
libuv                     1.40.0               he774522_0  
libwebp                   1.2.2                h2bbff1b_0  
locket                    1.0.0            py37haa95532_0  
lz4-c                     1.9.3                h2bbff1b_1  
markdown                  3.3.4            py37haa95532_0  
mat73                     0.52                     pypi_0    pypi  
matplotlib                3.5.2            py37haa95532_0  
matplotlib-base           3.5.2            py37hd77b12b_0  
matplotlib-inline         0.1.6            py37haa95532_0  
mkl                       2021.4.0           haa95532_640  
mkl-service               2.4.0            py37h2bbff1b_0  
mkl_fft                   1.3.1            py37h277e83a_0  
mkl_random                1.2.2            py37hf11a4ad_0  
multidict                 5.2.0            py37h2bbff1b_3  
munkres                   1.1.4                      py_0  
networkx                  2.6.3              pyhd3eb1b0_0  
ninja                     1.10.2               haa95532_5  
ninja-base                1.10.2               h6d14046_5  
numexpr                   2.8.3            py37hb80d3ca_0  
numpy                     1.21.5           py37h7a0a035_3  
numpy-base                1.21.5           py37hca35cd5_3  
oauthlib                  3.2.0              pyhd3eb1b0_1  
olefile                   0.46                     py37_0  
openpyxl                  3.0.10           py37h2bbff1b_0  
openssl                   1.1.1q               h2bbff1b_0  
opt_einsum                3.3.0              pyhd3eb1b0_1  
packaging                 21.3               pyhd3eb1b0_0  
pandas                    1.3.5            py37h6214cd6_0  
parso                     0.8.3              pyhd3eb1b0_0  
partd                     1.2.0              pyhd3eb1b0_1  
patsy                     0.5.2            py37haa95532_1  
pdbpp                     0.10.3             pyhd8ed1ab_0    conda-forge  
pickleshare               0.7.5           pyhd3eb1b0_1003  
pillow                    9.2.0            py37hdc2b20a_1  
pip                       22.1.2           py37haa95532_0  
prompt-toolkit            3.0.20             pyhd3eb1b0_0  
prompt_toolkit            3.0.20               hd3eb1b0_0  
protobuf                  3.20.1           py37hd77b12b_0  
pyasn1                    0.4.8              pyhd3eb1b0_0  
pyasn1-modules            0.2.8                      py_0  
pycparser                 2.21               pyhd3eb1b0_0  
pygments                  2.11.2             pyhd3eb1b0_0  
pyjwt                     2.4.0            py37haa95532_0  
pyopenssl                 22.0.0             pyhd3eb1b0_0  
pyparsing                 3.0.9            py37haa95532_0  
pyqt                      5.9.2            py37hd77b12b_6  
pyreadline                2.1                      py37_1  
pysocks                   1.7.1                    py37_1  
python                    3.7.13               h6244533_0  
python-dateutil           2.8.2              pyhd3eb1b0_0  
python_abi                3.7                     2_cp37m    conda-forge  
pytorch                   1.11.0          py3.7_cuda11.3_cudnn8_0    pytorch  
pytorch-metric-learning   1.6.0              pyh39e3cac_0    metric-learning  
pytorch-mutex             1.0                        cuda    pytorch  
pytz                      2022.1           py37haa95532_0  
pywavelets                1.3.0            py37h2bbff1b_0  
pyyaml                    6.0              py37h2bbff1b_1  
qt                        5.9.7            vc14h73c81de_0  
requests                  2.28.1           py37haa95532_0  
requests-oauthlib         1.3.0                      py_0  
rsa                       4.7.2              pyhd3eb1b0_1  
scikit-image              0.19.2           py37hf11a4ad_0  
scikit-learn              1.0.2            py37hf11a4ad_1  
scipy                     1.7.3            py37h7a0a035_2  
seaborn                   0.11.2             pyhd3eb1b0_0  
setuptools                63.4.1           py37haa95532_0  
sip                       4.19.13          py37hd77b12b_0  
six                       1.16.0             pyhd3eb1b0_1  
sqlite                    3.39.2               h2bbff1b_0  
statsmodels               0.13.2           py37h2bbff1b_0  
tensorboard               2.6.0                      py_1  
tensorboard-data-server   0.6.0            py37haa95532_0  
tensorboard-plugin-wit    1.8.1            py37haa95532_0  
tensorflow                2.1.0           eigen_py37hd727fc0_0  
tensorflow-addons         0.9.1                    pypi_0    pypi  
tensorflow-base           2.1.0           eigen_py37h49b2757_0  
tensorflow-estimator      2.6.0              pyh7b7c402_0  
termcolor                 1.1.0            py37haa95532_1  
threadpoolctl             2.2.0              pyh0d69192_0  
tifffile                  2020.10.1        py37h8c2d366_2  
tk                        8.6.12               h2bbff1b_0  
toolz                     0.11.2             pyhd3eb1b0_0  
torchaudio                0.11.0               py37_cu113    pytorch  
torchvision               0.12.0               py37_cu113    pytorch  
tornado                   6.2              py37h2bbff1b_0  
tqdm                      4.64.0           py37haa95532_0  
traitlets                 5.1.1              pyhd3eb1b0_0  
typeguard                 2.7.1                    pypi_0    pypi  
typing-extensions         4.3.0            py37haa95532_0  
typing_extensions         4.3.0            py37haa95532_0  
urllib3                   1.26.11          py37haa95532_0  
vc                        14.2                 h21ff451_1  
vicon-dssdk               1.11.0                   pypi_0    pypi  
vs2015_runtime            14.27.29016          h5e58377_2  
wcwidth                   0.2.5              pyhd3eb1b0_0  
werkzeug                  0.16.1                     py_0  
wheel                     0.37.1             pyhd3eb1b0_0  
win_inet_pton             1.1.0            py37haa95532_0  
wincertstore              0.2              py37haa95532_2  
wmctrl                    0.4                pyhd8ed1ab_0    conda-forge  
wrapt                     1.14.1           py37h2bbff1b_0  
xlrd                      2.0.1              pyhd3eb1b0_0  
xlsxwriter                3.0.3              pyhd3eb1b0_0  
xz                        5.2.5                h8cc25b3_1  
yaml                      0.2.5                he774522_0  
yarl                      1.8.1            py37h2bbff1b_0  
zipp                      3.8.0            py37haa95532_0  
zlib                      1.2.12               h8cc25b3_2  
zstd                      1.5.2                h19a0ad4_0  
