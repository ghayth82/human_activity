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

def preprocess_data(train_raw):
    sampling_rate = 20 # Hz

    train_data = train_raw.copy()                     
    train_data = train_data.drop_duplicates()
    
    train_data['idx'] = list(range(len(train_data)))
    train_data = train_data.set_index('idx')
    
    outliers = train_data['x'] > train_data['x'].quantile(0.999)
    train_data.loc[outliers, 'x'] = train_data['x'].quantile(0.999)
    outliers = train_data['x'] < train_data['x'].quantile(0.001)
    train_data.loc[outliers, 'x'] = train_data['x'].quantile(0.001)   
    
    outliers = train_data['y'] > train_data['y'].quantile(0.999)
    train_data.loc[outliers, 'y'] = train_data['y'].quantile(0.999)
    outliers = train_data['y'] < train_data['y'].quantile(0.001)
    train_data.loc[outliers, 'y'] = train_data['y'].quantile(0.001)    

    outliers = train_data['z'] > train_data['z'].quantile(0.999)
    train_data.loc[outliers, 'z'] = train_data['z'].quantile(0.999)
    outliers = train_data['z'] < train_data['z'].quantile(0.001)
    train_data.loc[outliers, 'z'] = train_data['z'].quantile(0.001)    
    
    train_data['magnitude'] = (train_data['x']**2 + train_data['y']**2 + train_data['z']**2)**0.5
    
    train_data['filt_x'] = butter_lowpass_filter(train_data['x'], 0.25, sampling_rate)
    train_data['filt_y'] = butter_lowpass_filter(train_data['y'], 0.25, sampling_rate)
    train_data['filt_z'] = butter_lowpass_filter(train_data['z'], 0.25, sampling_rate)

    return train_data

def generate_basic_stats(window_data, features, feat_list):
    features['mean_x'] = window_data['x'].mean() 
    features['mean_y'] = window_data['y'].mean() 
    features['mean_z'] = window_data['z'].mean() 
    features['mean_mag'] = window_data['magnitude'].mean() 
    feat_list.extend(['mean_x', 'mean_y', 'mean_z', 'mean_mag'])
    
    features['mean_filt_x'] = window_data['filt_x'].mean()
    features['mean_filt_y'] = window_data['filt_y'].mean() 
    features['mean_filt_z'] = window_data['filt_z'].mean() 
    feat_list.extend(['mean_filt_x', 'mean_filt_y', 'mean_filt_z'])
    
    features['std_x'] = window_data['x'].std()
    features['std_y'] = window_data['y'].std()
    features['std_z'] = window_data['z'].std()
    features['std_mag'] = window_data['magnitude'].std()
    feat_list.extend(['std_x', 'std_y', 'std_z', 'std_mag'])
                                                
    features['std_filt_x'] = window_data['filt_x'].std()
    features['std_filt_y'] = window_data['filt_y'].std()
    features['std_filt_z'] = window_data['filt_z'].std()
    feat_list.extend(['std_filt_x', 'std_filt_y', 'std_filt_z'])

    return features, feat_list

def generate_quantiles(window_data, features, feat_list):
    features['q25_x'] = window_data['x'].quantile(0.25)
    features['q25_y'] = window_data['y'].quantile(0.25)
    features['q25_z'] = window_data['z'].quantile(0.25)
    features['q25_mag'] = window_data['magnitude'].quantile(0.25)
    feat_list.extend(['q25_x', 'q25_y', 'q25_z', 'q25_mag'])
    
    features['q50_x'] = window_data['x'].quantile(0.50)
    features['q50_y'] = window_data['y'].quantile(0.50)
    features['q50_z'] = window_data['z'].quantile(0.50)
    features['q50_mag'] = window_data['magnitude'].quantile(0.50)
    feat_list.extend(['q50_x', 'q50_y', 'q50_z', 'q50_mag'])
    
    features['q75_x'] = window_data['x'].quantile(0.75)
    features['q75_y'] = window_data['y'].quantile(0.75)
    features['q75_z'] = window_data['z'].quantile(0.75)
    features['q75_mag'] = window_data['magnitude'].quantile(0.75)
    feat_list.extend(['q75_x', 'q75_y', 'q75_z', 'q75_mag'])
    
    return features, feat_list
    
def generate_psd(window_data, features, feat_list, sampling_rate):
    import numpy as np

    freq_bands = [0.01, 0.75, 2.50, 5.0, 7.5, 10]
    n_freq_bands = len(freq_bands)-1
    
    for axis in ['x', 'y', 'z']:
        freq, PSD = signal.periodogram(window_data[axis], sampling_rate)
        for j in range(n_freq_bands):
            freq_filter = np.logical_and(freq >= freq_bands[j], freq < freq_bands[j+1])
            features['psd' + str(j) + '_' + axis] = PSD[freq_filter].sum()/PSD.sum()
            feat_list.append('psd' + str(j) + '_' + axis)
    
    return features, feat_list
    
    

def generate_features(window_data, sampling_rate, d_type):
    
    activities = {"Jogging"  : 0, "LyingDown": 1, "Sitting"  : 2,
                  "Stairs"   : 3, "Standing" : 4, "Walking"  : 5}		 
    
    features = dict()
    feat_list = []
    
    if d_type == 'Train':
        # sets user_id to id with most rows in the window
        grouped = window_data.groupby('id').size()
        features['user_id'] = grouped.sort_values(ascending=False).index[0]
        
        # sets activity_id to activity with most rows in the window
        grouped = window_data.groupby('activity').size()
        activity = grouped.sort_values(ascending=False).index[0]
        features['activity_id'] = activities[activity]

        feat_list.extend(['user_id', 'activity_id'])
        
    features['win_begin_idx'] = window_data.iloc[0,:].name
    features['win_end_idx'] = window_data.iloc[len(window_data)-1,:].name
    feat_list.extend(['win_begin_idx', 'win_end_idx'])
    
    features, feat_list = generate_basic_stats(window_data, features, feat_list)
    features, feat_list = generate_quantiles(window_data, features, feat_list)
    features, feat_list = generate_psd(window_data, features, feat_list, sampling_rate)
    
    return features, feat_list


def generate_samples(train_data):
    sampling_rate = 20 # Hz
    window_length = 2  # seconds
    window_size = int(sampling_rate * window_length)
    
    features = []
    begin_row = 0
    end_row = window_size
    while(end_row <= train_data.shape[0]):
        window_data = train_data[begin_row:end_row]
        feat_sample, feat_list = generate_features(window_data, sampling_rate, 'Train')
        features.append(feat_sample)
        begin_row = end_row
        end_row = end_row + window_size
                    
    samples = pd.DataFrame(features, columns = feat_list)

    return samples



#%%
import time
import pandas as pd

start_time = time.time()

root_dir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train_raw = pd.read_csv(root_dir + "train_raw.csv")   
train_data = preprocess_data(train_raw) 
train = generate_samples(train_data)
train.to_csv(root_dir + 'train.csv', index = False)

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

