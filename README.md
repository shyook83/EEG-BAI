# EEG-brain age prediction model trainning
“ Dense_train.py” is a data loading and training code for EEG_BAI.
Preprocessed EEG data were converted into scalograms based on the continuous wavelet transform method using complex Morlet wavelet function.
We chose to use 16 frequency bands of the center frequency C, which were determined to be 0.5, 0.7, 0.9, 1.2, 1.6, 2.1, 2.8, 3.8, 5.0, 6.7, 8.9, 11.9, 15.8, 21.1, 28.1, and 37.5 Hz using the log scale distancing. Each band width B was determined as 1.5 times of the corresponding center frequency. 
Finally, the input data shape is (2160 (time bin), 16 (frequency bands), 6 (electrode channels).

# Basic demographics 
“basic demographics_BAI.xlxs” included chronological age, sex, and brain age index. 

