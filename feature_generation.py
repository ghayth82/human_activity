# -----------------------------------------------------------------
# Data preprocessing for human activity recognition
# -----------------------------------------------------------------

#%% -----------------------------------------------------------------
# Definition of Butterworth filters

from scipy import signal

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

#%% ----------------------------------------------------------------
# Feature generation

import pandas as pd

activities = {
                "Jogging"  : 0, 
                "LyingDown": 1,
                "Sitting"  : 2,
                "Stairs"   : 3,
                "Standing" : 4,
                "Walking"  : 5
              }		 

feat_list = [
                'id'           ,
                'activity_id'  ,
                'win_begin_idx',
                'win_end_idx'  ,
                'median_x'     ,
                'median_y'     ,
                'median_z'     ,
                'median_mag'   ,
                'q25_x'        ,
                'q25_y'        ,
                'q25_z'        ,
                'q25_mag'      ,
                'q75_x'        ,
                'q75_y'        ,
                'q75_z'        ,
                'q75_mag'      ,
                'mean_x'       ,
                'mean_y'       ,
                'mean_z'       ,
                'mean_mag'     ,
                'mean_filt_x'  , 
                'mean_filt_y'  , 
                'mean_filt_z'  , 
                'std_x'        ,
                'std_y'        ,
                'std_z'        ,
                'std_mag'      ,
                'std_filt_x'   , 
                'std_filt_y'   , 
                'std_filt_z'   , 
            ]

def generate_features(window_data):
    window_size = len(window_data)
    
    # sets user_id to id with most rows in the window
    grouped = window_data.groupby('id').size()
    user_id = grouped.sort_values(ascending=False).index[0]
    
    # sets activity_id to activity with most rows in the window
    grouped = window_data.groupby('activity').size()
    activity = grouped.sort_values(ascending=False).index[0]
    activity_id = activities[activity]
    
    features = {
                'id'               : user_id,
                'activity_id'      : activity_id,
                'win_begin_idx'    : window_data.iloc[0,:].name, 
                'win_end_idx'      : window_data.iloc[window_size-1,:].name,
                'median_x'         : window_data['x'].median(),
                'median_y'         : window_data['y'].median(),
                'median_z'         : window_data['z'].median(),
                'median_mag'       : window_data['magnitude'].median(),
                'q25_x'            : window_data['x'].quantile(0.25),
                'q25_y'            : window_data['y'].quantile(0.25),
                'q25_z'            : window_data['z'].quantile(0.25),
                'q25_mag'          : window_data['magnitude'].quantile(0.25),
                'q75_x'            : window_data['x'].quantile(0.75),
                'q75_y'            : window_data['y'].quantile(0.75),
                'q75_z'            : window_data['z'].quantile(0.75),
                'q75_mag'          : window_data['magnitude'].quantile(0.75),
                'mean_x'           : window_data['x'].mean(), 
                'mean_y'           : window_data['y'].mean(), 
                'mean_z'           : window_data['z'].mean(), 
                'mean_mag'         : window_data['magnitude'].mean(), 
                'mean_filt_x'      : window_data['filt_x'].mean(), 
                'mean_filt_y'      : window_data['filt_y'].mean(), 
                'mean_filt_z'      : window_data['filt_z'].mean(), 
                'std_x'            : window_data['x'].std(), 
                'std_y'            : window_data['y'].std(), 
                'std_z'            : window_data['z'].std(), 
                'std_mag'          : window_data['magnitude'].std(),
                'std_filt_x'       : window_data['filt_x'].std(), 
                'std_filt_y'       : window_data['filt_y'].std(), 
                'std_filt_z'       : window_data['filt_z'].std(), 
			     }
    
    return features


def generate_pure_samples(train_data, window_size):
    user_ids = train_data.id.unique()
    features = []
    for user_id in user_ids:
        user_data = train_data[train_data.id == user_id]
        for activity in activities.keys():
            activity_data = user_data[user_data.activity == activity]
            begin_row = 0
            end_row = window_size
            while(end_row <= activity_data.shape[0]):
                window_data = activity_data[begin_row:end_row]
                features.append(generate_features(window_data))
                begin_row = end_row
                end_row = end_row + window_size
                    
    samples = pd.DataFrame(features, columns = feat_list)

    return samples
    

def generate_hybrid_samples(train_data, window_size):
    features = []
    begin_row = 0
    end_row = window_size
    while(end_row <= train_data.shape[0]):
        window_data = train_data[begin_row:end_row]
        features.append(generate_features(window_data))
        begin_row = end_row
        end_row = end_row + window_size
                    
    samples = pd.DataFrame(features, columns = feat_list)

    return samples


def preprocess_train_data(train_raw, sampling_rate):
    train_data = train_raw.drop_duplicates().copy() 
    
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
    
    '''              
    train_data['filt_x'] = butter_lowpass_filter(train_data['x'], 0.25, sampling_rate)
    train_data['filt_y'] = butter_lowpass_filter(train_data['y'], 0.25, sampling_rate)
    train_data['filt_z'] = butter_lowpass_filter(train_data['z'], 0.25, sampling_rate)

    train_data['filt_x'] = butter_highpass_filter(train_data['x'], 0.25, sampling_rate)
    train_data['filt_y'] = butter_highpass_filter(train_data['y'], 0.25, sampling_rate)
    train_data['filt_z'] = butter_highpass_filter(train_data['z'], 0.25, sampling_rate)
    '''
    train_data['filt_x'] = butter_bandpass_filter(train_data['x'], 1.25, 2.50, sampling_rate)
    train_data['filt_y'] = butter_bandpass_filter(train_data['y'], 1.25, 2.50, sampling_rate)
    train_data['filt_z'] = butter_bandpass_filter(train_data['z'], 1.25, 2.50, sampling_rate)
    
    return train_data

    


#%%
import time
#import pandas as pd

start_time = time.time()

root_dir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train_raw = pd.read_csv(root_dir + "train_raw.csv")   

sampling_rate = 20 # Hz
window_length = 2 # seconds
window_size = int(sampling_rate * window_length)

train_data = preprocess_train_data(train_raw, sampling_rate) 

train = generate_hybrid_samples(train_data, window_size)

train.to_csv(root_dir + 'train.csv', index = False)

print("Total processing time: {:.2f} minutes".format((time.time()-start_time)/60))




                
#%%
# ---------------------------------------------------------------------------
# Filter tests

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

def sine_generator(fs, sinefreq, duration):
    T = duration
    nsamples = fs * T
    w = 2. * np.pi * sinefreq
    t_sine = np.linspace(0, T, nsamples, endpoint=False)
    y_sine = np.sin(w * t_sine)
    result = pd.DataFrame({ 
        'data' : y_sine} ,index=t_sine)
    return result

fps = 30
sine_fq = 10 #Hz
duration = 5 #seconds
sine_5Hz = sine_generator(fps,sine_fq,duration)
sine_fq = 1 #Hz
duration = 5 #seconds
sine_1Hz = sine_generator(fps,sine_fq,duration)

sine = sine_5Hz + sine_1Hz

filtered_sine = butter_highpass_filter(sine.data,10,fps)

plt.figure(figsize=(10,10))
plt.subplot(211)
plt.plot(range(len(sine)),sine)
plt.title('generated signal')
plt.subplot(212)
plt.plot(range(len(filtered_sine)),filtered_sine)
plt.title('filtered signal')
plt.show()
#%%

import matplotlib.pyplot as plt

train_data = preprocess_train_data(train_raw)

user_id = 358
activity = 'Stairs'
user_data = train_data[train_data.id == user_id]
activity_data = user_data[user_data.activity == activity]
signal_data = activity_data['y'][:400]

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


outliers = abs(train_data[337540:337560]['x']) < 1e-6
train_data.loc[outliers, 'x'] = 0

outliers = train_raw['timestamp'] < 0

train_raw[outliers]



















