# EEG-brain age prediction model trainning
“ Dense_train.py” is a data loading and training code for EEG_BAI.
Preprocessed EEG data were converted into scalograms based on the continuous wavelet transform method using complex Morlet wavelet function.
We chose to use 16 frequency bands of the center frequency C, which were determined to be 0.5, 0.7, 0.9, 1.2, 1.6, 2.1, 2.8, 3.8, 5.0, 6.7, 8.9, 11.9, 15.8, 21.1, 28.1, and 37.5 Hz using the log scale distancing. Each band width B was determined as 1.5 times of the corresponding center frequency. 
Finally, the input data shape is (2160 (time bin), 16 (frequency bands), 6 (electrode channels).

## Hardware Requirements
We tested this code in the following hardware environment.
CPU: AMD - Ryzen 7 5800X
GPU: RTX 3090
RAM: 32GB

## Software Requirements
### OS Requirements
This code is tested on Linux operating systems. The developmental version of the package has been tested on the following systems:
- Linux: Ubuntu 18.04

### other software versions that we tested.
- grpcio==1.27.2
- h5py==2.10.0
- Keras==2.3.1
- Keras-Applications==1.0.8
- Keras-Preprocessing==1.1.0
- numpy==1.18.1
- pandas==1.0.4
- pyflakes==2.2.0
- pyOpenSSL==19.1.0
- python-dateutil==2.8.1
- python-jsonrpc-server==0.3.4
- python-language-server==0.31.9
- scikit-learn==0.23.1
- scipy==1.4.1
- sklearn==0.0
- spyder==4.1.3
- spyder-kernels==1.9.1
- tensorboard==1.14.0
- tensorflow==1.14.0
- tensorflow-estimator==1.14.0

# Basic demographics 
“basic demographics_BAI.xlxs” included chronological age, sex, and brain age index. 
