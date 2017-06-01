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


def normalize_sampling(data):
    
    curr_idx = [10*i for i in range(len(data))]
    data['idx'] = curr_idx
    data = data.set_index('idx')
        
    timestamps = data['timestamp'].values
    norm_idx = list()                       
    previous_ts = timestamps[0]
    previous_idx = curr_idx[0]

    for j in range(len(timestamps)):
        if(((timestamps[j] - previous_ts)>60) & ((timestamps[j] - previous_ts)<360)):
            previous_ts = previous_ts + 50
            previous_idx = previous_idx + 1
            while (timestamps[j] - previous_ts) > 40:
                norm_idx.append(previous_idx)
                previous_ts = previous_ts + 50
                previous_idx = previous_idx + 1
           
        norm_idx.append(curr_idx[j])
        previous_ts = timestamps[j]
        previous_idx = curr_idx[j]
                
    data = data.reindex(index = norm_idx)                       
    
    data[['id', 'activity']] = data[['id', 'activity']].ffill()   
    data[['timestamp', 'x', 'y', 'z']] = data[['timestamp', 'x', 'y', 'z']].interpolate()                       
    data.id = data.id.astype(int)
    data.timestamp = data.timestamp.astype('int64')
    
    
    return data


def preprocess_data(data_raw):

    data = data_raw.copy()               
    data = data.drop_duplicates()
    #data = normalize_sampling(data)
    
    data['idx'] = list(range(len(data)))
    data = data.set_index('idx')

    for axis in ['x', 'y', 'z']:
        outliers = abs(data[axis])<1e-6
        data.loc[outliers, axis] = 1e-6
        outliers = data[axis] > data[axis].quantile(0.999)
        data.loc[outliers, axis] = data[axis].quantile(0.999)
        outliers = data[axis] < data[axis].quantile(0.001)
        data.loc[outliers, axis] = data[axis].quantile(0.001)   
    
    data['mag'] = (data['x']**2 + data['y']**2 + data['z']**2)**0.5
        
    for axis in ['x', 'y', 'z', 'mag']:
        col_name = 'jerk_' + axis
        data[col_name] = data[axis].diff() 
        
    return data


def postprocess_data(data):
    data = data[data.mean_mag > 5]
    return data
    

def generate_moments(window_data, features, feat_list):
    
    for axis in ['x', 'y', 'z', 'mag', 'jerk_x', 'jerk_y', 'jerk_z', 'jerk_mag']:
        feat_name = 'mean_' + axis
        features[feat_name] = window_data[axis].mean() 
        feat_list.append(feat_name)

    for axis in ['x', 'y', 'z', 'mag', 'jerk_x', 'jerk_y', 'jerk_z', 'jerk_mag']:
        feat_name = 'std_' + axis
        features[feat_name] = window_data[axis].std()
        feat_list.append(feat_name)
        
    for axis in ['x', 'y', 'z', 'mag', 'jerk_x', 'jerk_y', 'jerk_z', 'jerk_mag']:
        feat_name = 'skew_' + axis
        features[feat_name] = window_data[axis].skew()
        feat_list.append(feat_name)
        
    for axis in ['x', 'y', 'z', 'mag', 'jerk_x', 'jerk_y', 'jerk_z', 'jerk_mag']:
        feat_name = 'kurt_' + axis
        features[feat_name] = window_data[axis].kurtosis()
        feat_list.append(feat_name)
        
    return features, feat_list


def generate_correlations(window_data, features, feat_list):
    import numpy as np

    corr_matrix = window_data.loc[:,'x':'z'].corr()
    
    features['corr_xy'] = corr_matrix.loc['x', 'y']
    feat_list.append('corr_xy')
    features['corr_xz'] = corr_matrix.loc['x', 'z']
    feat_list.append('corr_xz')
    features['corr_yz'] = corr_matrix.loc['y', 'z']
    feat_list.append('corr_yz')
    
    # Real part of the sorted eigenvalues of correlation matrix are included
    try:
        w,v = np.linalg.eig(corr_matrix)
        w = np.sort(np.real(w))
        features['corr_eig0'] = w[0]
        features['corr_eig1'] = w[1]
        features['corr_eig2'] = w[2]

    except:
        features['corr_eig0'] = 0
        features['corr_eig1'] = 0
        features['corr_eig2'] = 0
        
    feat_list.append('corr_eig0')
    feat_list.append('corr_eig1')
    feat_list.append('corr_eig2')
    
    return features, feat_list


def generate_psd(window_data, features, feat_list, sampling_rate):
    import numpy as np

    freq_bands = [0.01, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    n_freq_bands = len(freq_bands)-1
    
    for axis in ['x', 'y', 'z', 'mag']:
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

def generate_spectral_edge(window_data, features, feat_list, sampling_rate):
    import scipy.signal as signal
    import numpy as np

    # Calculate the spectral edge at 65% power below 10Hz
    min_freq = 0.01
    max_freq = 10 
    pct_power = 0.65 
    
    for axis in ['x', 'y', 'z', 'mag']:
        feat_name = 'spec_edge_' + axis
        feat_list.append(feat_name)

        freq, PSD = signal.periodogram(window_data[axis], sampling_rate)
        if PSD.sum() > 1e-6:
            PSD = PSD/PSD.sum()
            freq_filter = np.logical_and(freq >= min_freq, freq <= max_freq)
            target_power = pct_power*PSD[freq_filter].sum()    
            cum_PSD = np.cumsum(PSD[freq_filter])
            # The spectral edge frequency corresponds to the frequency at 
            # the point on the freq axis where cum_PSD = target_power            
            freq_edge = freq[np.argmin(np.abs(cum_PSD - target_power))]
            features[feat_name] = freq_edge
        else:
            features[feat_name] = 0
            
    return features, feat_list


def generate_shannon_entropy(window_data, features, feat_list, sampling_rate):
    import scipy.signal as signal
    import numpy as np

    freq_bands = [0.01, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    n_freq_bands = len(freq_bands)-1
    
    for axis in ['x', 'y', 'z', 'mag']:
        feat_name = 'entropy_' + axis
        feat_list.append(feat_name)

        freq, PSD = signal.periodogram(window_data[axis], sampling_rate)
        if PSD.sum() > 1e-6:
            freq_bin_density = np.zeros(n_freq_bands)
            for j in range(n_freq_bands):
                freq_filter = np.logical_and(freq >= freq_bands[j], freq < freq_bands[j+1])
                freq_bin_density[j] = PSD[freq_filter].sum()/PSD.sum()
                
            features[feat_name] = -freq_bin_density.dot(np.log2(freq_bin_density)) 
        else:
            features[feat_name] = 0
            
    return features, feat_list


def generate_hjorth_parameters(window_data, features, feat_list):
    import numpy as np

    for axis in ['x', 'y', 'z', 'mag']:
        feat_name = 'activity_' + axis
        feat_list.append(feat_name)
        features[feat_name] = window_data[axis].var(axis=0)

    for axis in ['x', 'y', 'z', 'mag']:
        feat_name = 'mobility_' + axis
        feat_list.append(feat_name)
        if window_data[axis].std() > 1e-6:
            features[feat_name] = np.diff(window_data[axis]).std()/window_data[axis].std()
        else:
            features[feat_name] = 0
        
    for axis in ['x', 'y', 'z', 'mag']:
        feat_name = 'complexity_' + axis
        feat_list.append(feat_name)
        if window_data[axis].std() > 1e-6:
            mobility = window_data[axis].diff().std()/window_data[axis].std()
            if abs(mobility) > 1e-6:
                complexity = (np.diff(window_data[axis], n=2).std()/np.diff(window_data[axis]).std())/mobility
                features[feat_name] = complexity
            else:
                features[feat_name] = 0
        else:
            features[feat_name] = 0
        
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
    features, feat_list = generate_correlations(window_data, features, feat_list)
    #features, feat_list = generate_psd(window_data, features, feat_list, sampling_rate)
    #features, feat_list = generate_spectral_edge(window_data, features, feat_list, sampling_rate)
    #features, feat_list = generate_shannon_entropy(window_data, features, feat_list, sampling_rate)
    #features, feat_list = generate_hjorth_parameters(window_data, features, feat_list)
    
    return features, feat_list


def generate_samples(data, d_type):
    sampling_rate = 20 # Hz
    window_length = 2.5  # seconds
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
train = postprocess_data(train)
train.to_csv(root_dir + 'train.csv', index = False)

'''
test_raw = pd.read_csv(root_dir + "test_raw.csv")
test_data = preprocess_data(test_raw) 
test = generate_samples(test_data, 'Test')
test.to_csv(root_dir + 'test.csv', index = False)
'''

print("Total processing time: {:.2f} minutes".format((time.time()-start_time)/60))

                
#%% ---------------------------------------------------------------------------























