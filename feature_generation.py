# -----------------------------------------------------------------
# Data preprocessing for human activity recognition

#%%
import time
import pandas as pd

start_time = time.time()

root_dir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train_raw = pd.read_csv(root_dir + "train_raw.csv")    
train_raw = train_raw.drop_duplicates() 
train_raw.loc[:, 'idx'] = list(range(len(train_raw)))
train_raw = train_raw.set_index('idx')

train_raw['x'] = butter_highpass_filter(train_raw['x'], 1, 20)
train_raw['y'] = butter_highpass_filter(train_raw['y'], 1, 20)
train_raw['z'] = butter_highpass_filter(train_raw['z'], 1, 20)

train_raw.loc[:,'magnitude'] = (train_raw['x']**2 + train_raw['y']**2 + train_raw['z']**2)**0.5

sampling_rate = 20 # Hz
window_length = 2.5 # seconds
window_size = int(sampling_rate * window_length)

activities = [{"act_type": "Jogging"    , "act_id": 0},
              {"act_type": "LyingDown"  , "act_id": 1},
              {"act_type": "Sitting"    , "act_id": 2},
              {"act_type": "Stairs"     , "act_id": 3},
              {"act_type": "Standing"   , "act_id": 4},
              {"act_type": "Walking"    , "act_id": 5}]

user_ids = train_raw.id.unique()

columns = ['id', 'win_begin_idx', 'win_end_idx', 'activity_id',
           'median_x', 'mean_x', 'std_x', 
           'median_y', 'mean_y', 'std_y', 
           'median_z', 'mean_z', 'std_z', 
           'median_mag', 'mean_mag', 'std_mag']

features = []
for user_id in user_ids:
    user_data = train_raw[train_raw.id == user_id]
    for activity in activities:
        activity_data = user_data[user_data.activity == activity['act_type']]
        begin_row = 0
        end_row = window_size
        while(end_row <= activity_data.shape[0]):
            window_data = activity_data[begin_row:end_row]
            
            features.append({'id'               : user_id,
                             'win_begin_idx'    : window_data.iloc[0,:].name, 
                             'win_end_idx'      : window_data.iloc[window_size-1,:].name,
                             'activity_id'      : activity['act_id'],
                             'median_x'         : window_data['x'].median(),
                             'mean_x'           : window_data['x'].mean(), 
                             'std_x'            : window_data['x'].std(), 
                             'median_y'         : window_data['y'].median(),
                             'mean_y'           : window_data['y'].mean(), 
                             'std_y'            : window_data['y'].std(), 
                             'median_z'         : window_data['z'].median(),
                             'mean_z'           : window_data['z'].mean(), 
                             'std_z'            : window_data['z'].std(), 
                             'median_mag'       : window_data['magnitude'].median(),
                             'mean_mag'         : window_data['magnitude'].mean(), 
                             'std_mag'          : window_data['magnitude'].std()})

            begin_row = end_row
            end_row = end_row + window_size
                
train = pd.DataFrame(features, columns = columns)
train.to_csv(root_dir + 'train.csv', index = False)

print("Total processing time: {:.2f} minutes".format((time.time()-start_time)/60))

                
#%%
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

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

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

sine = train_raw['x'][:200]
filtered_sine = butter_highpass_filter(sine, 1, 20)


plt.figure(figsize=(10,10))
plt.subplot(211)
plt.plot(range(len(sine)),sine)
plt.title('generated signal')
plt.subplot(212)
plt.plot(range(len(filtered_sine)),filtered_sine)
plt.title('filtered signal')
plt.show()










