# -----------------------------------------------------------------
# Data preprocessing for human activity recognition

import time
import pandas as pd

start_time = time.time()

root_dir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train_raw = pd.read_csv(root_dir + "train_raw.csv")     
train_raw = train_raw.sort_values(['id', 'timestamp'])
train_raw.loc[:, 'idx'] = list(range(len(train_raw)))
train_raw = train_raw.set_index('idx')

train_raw.loc[:,'magnitude'] = (train_raw['x']**2 + train_raw['y']**2 + train_raw['z']**2)**0.5

sampling_rate = 20 # Hz
window_length = 2 # seconds
window_size = sampling_rate * window_length

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

                














