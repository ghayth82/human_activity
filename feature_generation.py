# -----------------------------------------------------------------
# Data preprocessing for human activity recognition
# -----------------------------------------------------------------

#%% ----------------------------------------------------------------
# Feature generation

import scipy.signal as signal

def butter_highpass_filter(signal_data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
    filtered_signal = signal.filtfilt(b, a, signal_data)
    return filtered_signal

def butter_lowpass_filter(signal_data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
    filtered_signal = signal.filtfilt(b, a, signal_data)
    return filtered_signal

def butter_bandpass_filter(signal_data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass', analog=False)
    filtered_signal = signal.filtfilt(b, a, signal_data)
    return filtered_signal

def preprocess_data(data_raw):
    sampling_rate = 20 # Hz

    data = data_raw.copy()                     
    data = data.drop_duplicates()
    
    data['idx'] = list(range(len(data)))
    data = data.set_index('idx')

    for axis in ['x', 'y', 'z']:
        outliers = abs(data[axis])<1e-6
        data.loc[outliers, axis] = 1e-6
        outliers = data[axis] > data[axis].quantile(0.999)
        data.loc[outliers, axis] = data[axis].quantile(0.999)
        outliers = data[axis] < data[axis].quantile(0.001)
        data.loc[outliers, axis] = data[axis].quantile(0.001)   
    
    data['magnitude'] = (data['x']**2 + data['y']**2 + data['z']**2)**0.5
    
    data['filt_x'] = butter_lowpass_filter(data['x'], 0.75, sampling_rate)
    data['filt_y'] = butter_lowpass_filter(data['y'], 0.75, sampling_rate)
    data['filt_z'] = butter_lowpass_filter(data['z'], 0.75, sampling_rate)

    return data

def generate_moments(window_data, features, feat_list):
    
    for axis in ['x', 'y', 'z', 'filt_x', 'filt_y', 'filt_z', 'magnitude']:
        feat_name = 'mean_' + axis
        features[feat_name] = window_data[axis].mean() 
        feat_list.append(feat_name)

    for axis in ['x', 'y', 'z', 'filt_x', 'filt_y', 'filt_z', 'magnitude']:
        feat_name = 'std_' + axis
        features[feat_name] = window_data[axis].std()
        feat_list.append(feat_name)
        
    for axis in ['x', 'y', 'z', 'filt_x', 'filt_y', 'filt_z', 'magnitude']:
        feat_name = 'skew_' + axis
        features[feat_name] = window_data[axis].skew()
        feat_list.append(feat_name)
        
    for axis in ['x', 'y', 'z', 'filt_x', 'filt_y', 'filt_z', 'magnitude']:
        feat_name = 'kurt_' + axis
        features[feat_name] = window_data[axis].kurtosis()
        feat_list.append(feat_name)
        
    return features, feat_list


def generate_quantiles(window_data, features, feat_list):

    for axis in ['x', 'y', 'z', 'filt_x', 'filt_y', 'filt_z', 'magnitude']:
        feat_name = 'q25_' + axis
        features[feat_name] = window_data[axis].quantile(0.25)
        feat_list.append(feat_name)
        
    for axis in ['x', 'y', 'z', 'filt_x', 'filt_y', 'filt_z', 'magnitude']:
        feat_name = 'q50_' + axis
        features[feat_name] = window_data[axis].quantile(0.50)
        feat_list.append(feat_name)

    for axis in ['x', 'y', 'z', 'filt_x', 'filt_y', 'filt_z', 'magnitude']:
        feat_name = 'q75_' + axis
        features[feat_name] = window_data[axis].quantile(0.75)
        feat_list.append(feat_name)

    return features, feat_list

    
def generate_psd(window_data, features, feat_list, sampling_rate):
    import numpy as np

    freq_bands = [0.01, 0.75, 2.50, 5.0, 7.5, 10]
    n_freq_bands = len(freq_bands)-1
    
    for axis in ['x', 'y', 'z', 'filt_x', 'filt_y', 'filt_z', 'magnitude']:
        freq, PSD = signal.periodogram(window_data[axis], sampling_rate)
        for j in range(n_freq_bands):
            feat_name = 'psd' + str(j) + '_' + axis
            if PSD.sum() > 1e-6:
                freq_filter = np.logical_and(freq >= freq_bands[j], freq < freq_bands[j+1])
                features[feat_name] = PSD[freq_filter].sum()/PSD.sum()
            else:
                features[feat_name] = 0

            feat_list.append(feat_name)
    
    return features, feat_list
    

def generate_features(window_data, sampling_rate, d_type):
    
    activities = {"Jogging"  : 0, "LyingDown": 1, "Sitting"  : 2,
                  "Stairs"   : 3, "Standing" : 4, "Walking"  : 5}		 
    
    features = dict()
    feat_list = list()
    
    if d_type == 'Train':
        # sets user_id to id with most rows in the window
        grouped = window_data.groupby('id').size()
        features['user_id'] = grouped.sort_values(ascending=False).index[0]
        feat_list.append('user_id')
        
        # sets activity_id to activity with most rows in the window
        grouped = window_data.groupby('activity').size()
        activity = grouped.sort_values(ascending=False).index[0]
        features['activity_id'] = activities[activity]
        feat_list.append('activity_id')

    features['win_begin_idx'] = window_data.iloc[0,:].name
    features['win_end_idx'] = window_data.iloc[len(window_data)-1,:].name
    feat_list.extend(['win_begin_idx', 'win_end_idx'])
    
    features, feat_list = generate_moments(window_data, features, feat_list)
    features, feat_list = generate_quantiles(window_data, features, feat_list)
    features, feat_list = generate_psd(window_data, features, feat_list, sampling_rate)
    
    return features, feat_list


def generate_samples(data, d_type):
    sampling_rate = 20 # Hz
    window_length = 2  # seconds
    window_size = int(sampling_rate * window_length)
    
    features = []
    begin_row = 0
    end_row = window_size
    while(end_row <= data.shape[0]):
        window_data = data[begin_row:end_row]
        feat_sample, feat_list = generate_features(window_data, sampling_rate, d_type)
        features.append(feat_sample)
        begin_row = end_row
        end_row = end_row + window_size
                    
    samples = pd.DataFrame(features, columns = feat_list)

    return samples



#%% ---------------------------------------------------------------------------
# Train and Test files generation

import time
import pandas as pd

start_time = time.time()

root_dir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train_raw = pd.read_csv(root_dir + "train_raw.csv")   
train_data = preprocess_data(train_raw) 
train = generate_samples(train_data, 'Train')
train.to_csv(root_dir + 'train.csv', index = False)
'''
test_raw = pd.read_csv(root_dir + "test_raw.csv")
test_data = preprocess_data(test_raw) 
test = generate_samples(test_data, 'Test')
test.to_csv(root_dir + 'test.csv', index = False)
'''
print("Total processing time: {:.2f} minutes".format((time.time()-start_time)/60))

                
#%% ---------------------------------------------------------------------------
# Filter tests

import matplotlib.pyplot as plt

user_id = 293
activity = 'Sitting'
user_data = train_data[train_data.id == user_id]
activity_data = user_data[user_data.activity == activity]
signal_data = activity_data['y'][:400]

sampling_rate = 20 # Hz
filtered_signal = butter_lowpass_filter(signal_data, 0.25, sampling_rate)
#filtered_signal = butter_highpass_filter(signal_data, 0.5, sampling_rate)
#filtered_signal = butter_bandpass_filter(signal_data, 0.5, 2.5, sampling_rate)
         

plt.figure(figsize=(10,10))
plt.subplot(211)
plt.plot(range(len(signal_data)),signal_data)
plt.title('generated signal')
plt.subplot(212)
plt.plot(range(len(filtered_signal)),filtered_signal)
plt.title('filtered signal')
plt.show()

