# --------------------------------------------------------------------------
# Data Challenge '17
# Team Classy Fyers final submission script
# --------------------------------------------------------------------------

def preprocess_data(data_raw):

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
    
    return features, feat_list


def generate_samples(data, d_type):
    import pandas as pd

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


def create_samples_files(root_dir, train_raw_filename, test_raw_filename):
    import pandas as pd
    
    print('Creating train samples file...')    
    train_raw = pd.read_csv(root_dir + train_raw_filename)   
    train_data = preprocess_data(train_raw) 
    train = generate_samples(train_data, 'Train')
    train = postprocess_data(train)
    train.to_csv(root_dir + 'train.csv', index = False)
    
    print('Creating test samples file...')    
    test_raw = pd.read_csv(root_dir + test_raw_filename)
    test_data = preprocess_data(test_raw) 
    test = generate_samples(test_data, 'Test')
    test.to_csv(root_dir + 'test.csv', index = False)
    
    return

def create_submission(root_dir, test_raw_filename):
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    
    print('Generating submission file...')
    
    train_data = pd.read_csv(root_dir + "train.csv")    
    X_train = train_data.loc[:, 'mean_x':].values
    y_train = train_data.loc[:, 'activity_id'].values
    
    test_data = pd.read_csv(root_dir + "test.csv")
    X_test = test_data.loc[:, 'mean_x':].values
                          
    params = dict()
    params['objective'] = 'multi:softmax'
    params['updater'] = 'grow_gpu_hist'
    params['num_class'] = 6
    params['max_depth'] = 4
    params['min_child_weight'] = 13
    params['subsample'] = 0.5
    params['colsample_bytree'] = 0.5
    params['learning_rate'] = 0.1
    num_round = 15
    
    dtrain = xgb.DMatrix(X_train, y_train)
    classifier = xgb.train(params, dtrain, num_round)

    dpred = xgb.DMatrix(X_test)
    y_pred = classifier.predict(dpred)

    # Assigns predictions to respective observation windows
    window_pred = test_data[['win_begin_idx', 'win_end_idx']].copy()
    window_pred['pred'] = y_pred[:]
    
    test_raw = pd.read_csv(root_dir + test_raw_filename)
    submission = test_raw.drop_duplicates().copy()                     
    submission['idx'] = list(range(len(submission)))
    submission = submission.set_index('idx')
    sub_pred = 5*np.ones(len(submission))
    
    for index, row in window_pred.iterrows():
        begin_idx = int(row['win_begin_idx'])
        end_idx = int(row['win_end_idx'] + 1)
        sub_pred[begin_idx:end_idx] = row['pred']
    
    submission.loc[:, 'pred'] = sub_pred[:]
    
    activities = {0 : "Jogging", 1 : "LyingDown", 2 : "Sitting",
                  3 : "Stairs" , 4 : "Standing",  5 : "Walking"}		 
    
    for activity_id in range(6):
        submission.loc[submission['pred'] == activity_id, 'activity'] = activities[activity_id]
        
    submission = submission.drop('pred', axis=1)
    submission.to_csv(root_dir + 'classy_fyers_final_submission.csv', index = False)
    
    return
    
    

# --------------------------------------------------------------------------- #
# Main module function
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    import argparse
    import time

    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir')
    parser.add_argument('train_raw_filename')
    parser.add_argument('test_raw_filename')
    args = parser.parse_args()
    
    print()
    print("Root directory: {}".format(args.root_dir))
    print("File containing raw training data: {}".format(args.train_raw_filename))    
    print("File containing raw testing data: {}".format(args.test_raw_filename))    
    print()
    
    create_samples_files(args.root_dir, args.train_raw_filename, args.test_raw_filename)
    create_submission(args.root_dir, args.test_raw_filename)
    
    print()
    print("Total processing time: {:.2f} minutes".format((time.time()-start_time)/60))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    