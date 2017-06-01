# --------------------------------------------------------------------------
# Model training for human activity recognition
# --------------------------------------------------------------------------

#%% ------------------------------------------------------------------------
# Load data

import time
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

root_dir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train_data = pd.read_csv(root_dir + "train.csv")
X = train_data.loc[:, 'mean_x':].values
y = train_data.loc[:, 'activity_id'].values
groups = train_data.loc[:, 'user_id'].values     
                       

#%% -------------------------------------------------------------------------
# Overfitting control

start_time = time.time()

gkf = GroupKFold(n_splits=4)

scores_test = []
scores_train = []
accuracy0 = []
accuracy1 = []
accuracy2 = []
accuracy3 = []
accuracy4 = []
accuracy5 = []
for train_index, test_index in gkf.split(X, y, groups):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    param = dict()
    param['objective'] = 'multi:softmax'
    param['num_class'] = 6
    param['updater'] = 'grow_gpu_hist'
    param['max_depth'] = 4
    param['min_child_weight'] = 13
    param['gamma'] = 0
    param['reg_lambda'] = 1
    param['reg_alpha'] = 0
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.5
    param['learning_rate'] = 0.1
    num_round = 15
    
    dtrain = xgb.DMatrix(X_train, y_train)
    classifier = xgb.train(param, dtrain, num_round)

    dpred = xgb.DMatrix(X_train)
    y_pred = classifier.predict(dpred)
    scores_train.append(accuracy_score(y_train, y_pred))

    dpred = xgb.DMatrix(X_test)
    y_pred = classifier.predict(dpred)
    scores_test.append(accuracy_score(y_test, y_pred))
    
    accuracy0.append(sum((y_pred == 0) & (y_pred == y_test))/sum(y_test == 0))
    accuracy1.append(sum((y_pred == 1) & (y_pred == y_test))/sum(y_test == 1))
    accuracy2.append(sum((y_pred == 2) & (y_pred == y_test))/sum(y_test == 2))
    accuracy3.append(sum((y_pred == 3) & (y_pred == y_test))/sum(y_test == 3))
    accuracy4.append(sum((y_pred == 4) & (y_pred == y_test))/sum(y_test == 4))
    accuracy5.append(sum((y_pred == 5) & (y_pred == y_test))/sum(y_test == 5))
    
scores_train = np.array(scores_train)
scores_test = np.array(scores_test)
accuracy0 = np.array(accuracy0)
accuracy1 = np.array(accuracy1)
accuracy2 = np.array(accuracy2)
accuracy3 = np.array(accuracy3)
accuracy4 = np.array(accuracy4)
accuracy5 = np.array(accuracy5)

print("Train set accuracy: {:.3f} (+/- {:.3f})".format(scores_train.mean(), scores_train.std()))
print("Test set accuracy:  {:.3f} (+/- {:.3f})".format(scores_test.mean(), scores_test.std()))
print(scores_test)
print()
print("Test set accuracy for {}: {:.3f}".format(0, accuracy0.mean()))
print("Test set accuracy for {}: {:.3f}".format(1, accuracy1.mean()))
print("Test set accuracy for {}: {:.3f}".format(2, accuracy2.mean()))
print("Test set accuracy for {}: {:.3f}".format(3, accuracy3.mean()))
print("Test set accuracy for {}: {:.3f}".format(4, accuracy4.mean()))
print("Test set accuracy for {}: {:.3f}".format(5, accuracy5.mean()))
print()
print("Total processing time: {:.2f} seconds".format((time.time()-start_time)))

# Benchmark
#Train set accuracy: 0.927 (+/- 0.006)
#Test set accuracy:  0.842 (+/- 0.007)
#[ 0.8528448   0.84125801  0.83276247  0.84004174]
#
#Test set accuracy for 0: 0.829
#Test set accuracy for 1: 0.000
#Test set accuracy for 2: 0.821
#Test set accuracy for 3: 0.000
#Test set accuracy for 4: 0.742
#Test set accuracy for 5: 0.929


#%% ------------------------------------------------------------------------
# Fold tests

gkf = list(GroupKFold(n_splits=4).split(X, y, groups))

train_index, test_index = gkf[1]

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

param = dict()
param['objective'] = 'multi:softmax'
param['num_class'] = 6
param['updater'] = 'grow_gpu_hist'
param['max_depth'] = 5
param['min_child_weight'] = 35
param['gamma'] = 0
param['reg_lambda'] = 1
param['reg_alpha'] = 0.0
param['subsample'] = 0.5
param['colsample_bytree'] = 0.5
param['learning_rate'] = 0.09
num_round = 25


dtrain = xgb.DMatrix(X_train, y_train)
dtest  = xgb.DMatrix(X_test, y_test)
evallist = [(dtest,'test'), (dtrain,'train')]
classifier = xgb.train(param, dtrain, num_round, evals = evallist)

dpred = xgb.DMatrix(X_test)
y_pred = classifier.predict(dpred)
confusion_matrix(y_test, y_pred)

dpred = xgb.DMatrix(X_train)
y_pred = classifier.predict(dpred)
confusion_matrix(y_train, y_pred)

Counter(y_test)
Counter(y_train)       


# "Jogging"  : 0, 
# "LyingDown": 1,
# "Sitting"  : 2,
# "Stairs"   : 3,
# "Standing" : 4,
# "Walking"  : 5



#%% ------------------------------------------------------------------------
# First stage model

import time
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

train_data = pd.read_csv(root_dir + "train.csv")

X = train_data.loc[:, 'mean_x':].values
y = train_data.loc[:, 'activity_id'].values
groups = train_data.loc[:, 'user_id'].values     


#%%-------------------------------------------------------------------------                       
start_time = time.time()

nm = NearMiss(random_state=31416, ratio = 'auto', n_jobs=-1)
sm1 = SMOTE(random_state=31416, ratio = 'auto', k_neighbors=5, n_jobs=-1)    
sm3 = SMOTE(random_state=31416, ratio = 'auto', k_neighbors=5, n_jobs=-1)    
ros = RandomOverSampler(random_state=31416)

gkf = GroupKFold(n_splits=4)

scores_test = []
scores_train = []
for train_index, test_index in gkf.split(X, y, groups):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    '''
    activity_filter = (y_train==1) | (y_train==5)
    X_res = X_train[activity_filter]
    y_res = y_train[activity_filter]
    X_train = X_train[~activity_filter]
    y_train = y_train[~activity_filter]
     
    X_res, y_res = sm1.fit_sample(X_res, y_res) 
    X_train = np.concatenate((X_train, X_res))
    y_train = np.concatenate((y_train, y_res))
    
    activity_filter = (y_train==2) | (y_train==5)
    X_res = X_train[activity_filter]
    y_res = y_train[activity_filter]
    X_train = X_train[~activity_filter]
    y_train = y_train[~activity_filter]
     
    X_res, y_res = sm3.fit_sample(X_res, y_res) 
    X_train = np.concatenate((X_train, X_res))
    y_train = np.concatenate((y_train, y_res))
    '''
    
    X_train, y_train = ros.fit_sample(X_train, y_train)
        
    
    param = dict()
    param['objective'] = 'multi:softmax'
    param['num_class'] = 6
    param['updater'] = 'grow_gpu_hist'
    param['max_depth'] = 5
    param['min_child_weight'] = 20
    param['gamma'] = 0
    param['reg_lambda'] = 1
    param['reg_alpha'] = 0.0
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.5
    param['learning_rate'] = 0.1
    num_round = 10
    
    dtrain = xgb.DMatrix(X_train, y_train)
    classifier = xgb.train(param, dtrain, num_round)

    dpred = xgb.DMatrix(X_train)
    y_pred = classifier.predict(dpred)
    scores_train.append(accuracy_score(y_train, y_pred))

    dpred = xgb.DMatrix(X_test)
    y_pred = classifier.predict(dpred)
    scores_test.append(accuracy_score(y_test, y_pred))
    
scores_train = np.array(scores_train)
scores_test = np.array(scores_test)

print("Train set accuracy: {:.3f} (+/- {:.3f})".format(scores_train.mean(), scores_train.std()))
print("Test set accuracy:  {:.3f} (+/- {:.3f})".format(scores_test.mean(), scores_test.std()))
print(scores_test)
print()

for i in range(6):
    accuracy = sum((y_pred == i) & (y_pred == y_test))/sum(y_test == i)
    print("Test set accuracy for {}: {:.2f}".format(i, accuracy))

print()    
print("Total processing time: {:.2f} seconds".format((time.time()-start_time)))

#Train set accuracy: 0.935 (+/- 0.004)
#Test set accuracy:  0.721 (+/- 0.043)
#[ 0.71775395  0.65687882  0.77810871  0.73166369]
#
#Test set accuracy for 0: 0.85
#Test set accuracy for 1: 0.59
#Test set accuracy for 2: 0.26
#Test set accuracy for 3: 0.22
#Test set accuracy for 4: 0.76
#Test set accuracy for 5: 0.84

    
#%% ------------------------------------------------------------------------
# Two stages model

start_time = time.time()

nm = NearMiss(random_state=31416, ratio = 'auto', n_jobs=-1)
sm1 = SMOTE(random_state=31416, ratio = 0.8, k_neighbors=5, n_jobs=-1)    
sm3 = SMOTE(random_state=31416, ratio = 'auto', k_neighbors=5, n_jobs=-1)    

gkf = GroupKFold(n_splits=4)

scores_test = []
scores_train = []
for train_index, test_index in gkf.split(X, y, groups):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # -----------------------------------
    # First stage classifier
    '''
    activity_filter = (y_train==1) | (y_train==5)
    X_res = X_train[activity_filter]
    y_res = y_train[activity_filter]
    X_train = X_train[~activity_filter]
    y_train = y_train[~activity_filter]
     
    X_res, y_res = sm1.fit_sample(X_res, y_res) 
    X_train = np.concatenate((X_train, X_res))
    y_train = np.concatenate((y_train, y_res))
    
    activity_filter = (y_train==2) | (y_train==5)
    X_res = X_train[activity_filter]
    y_res = y_train[activity_filter]
    X_train = X_train[~activity_filter]
    y_train = y_train[~activity_filter]
     
    X_res, y_res = sm3.fit_sample(X_res, y_res) 
    X_train = np.concatenate((X_train, X_res))
    y_train = np.concatenate((y_train, y_res))
    '''
    
    X_train, y_train = ros.fit_sample(X_train, y_train)
        
    
    param = dict()
    param['objective'] = 'multi:softmax'
    param['num_class'] = 6
    param['updater'] = 'grow_gpu_hist'
    param['max_depth'] = 5
    param['min_child_weight'] = 20
    param['gamma'] = 0
    param['reg_lambda'] = 1
    param['reg_alpha'] = 0.0
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.5
    param['learning_rate'] = 0.1
    num_round = 10
    
    dtrain = xgb.DMatrix(X_train, y_train)
    cls_stage1 = xgb.train(param, dtrain, num_round)
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Predictions from stage 1 classifier
    dpred = xgb.DMatrix(X_train)
    y_pred = cls_stage1.predict(dpred)
    
    X_train = np.column_stack((X_train, y_pred))
    
    # -----------------------------------
    # Main classifier
    
    param = dict()
    param['objective'] = 'multi:softmax'
    param['num_class'] = 6
    param['updater'] = 'grow_gpu_hist'
    param['max_depth'] = 4
    param['min_child_weight'] = 10
    param['gamma'] = 0
    param['reg_lambda'] = 1
    param['reg_alpha'] = 0.0
    param['subsample'] = 0.4
    param['colsample_bytree'] = 0.4
    param['learning_rate'] = 0.2
    num_round = 20
    
    dtrain = xgb.DMatrix(X_train, y_train)
    cls_main = xgb.train(param, dtrain, num_round)
    
    dpred = xgb.DMatrix(X_train)
    y_pred = cls_main.predict(dpred)
    scores_train.append(accuracy_score(y_train, y_pred))
    
    dpred = xgb.DMatrix(X_test)
    y_pred = cls_stage1.predict(dpred)
    X_test = np.column_stack((X_test, y_pred))
    dpred = xgb.DMatrix(X_test)
    y_pred = cls_main.predict(dpred)
    scores_test.append(accuracy_score(y_test, y_pred))
    
scores_train = np.array(scores_train)
scores_test = np.array(scores_test)

print("Train set accuracy: {:.3f} (+/- {:.3f})".format(scores_train.mean(), scores_train.std()))
print("Test set accuracy:  {:.3f} (+/- {:.3f})".format(scores_test.mean(), scores_test.std()))
print(scores_test)
print()

for i in range(6):
    accuracy = sum((y_pred == i) & (y_pred == y_test))/sum(y_test == i)
    print("Test set accuracy for {}: {:.3f}".format(i, accuracy))

print()    
print("Total processing time: {:.2f} seconds".format((time.time()-start_time)))
                       
                       


#%% ------------------------------------------------------------------------
# Submission generation

import time
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

start_time = time.time()

root_dir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train_data = pd.read_csv(root_dir + "train.csv")    
X_train = train_data.loc[:, 'mean_x':].values
y_train = train_data.loc[:, 'activity_id'].values

test_data = pd.read_csv(root_dir + "test.csv")
X_test = test_data.loc[:, 'mean_x':].values
                      
params = dict()
params['objective'] = 'multi:softmax'
params['n_estimators'] = 100
params['learning_rate'] = 0.1
params['max_depth'] = 3
params['min_child_weight'] = 10
params['gamma'] = 10
params['subsample'] = 0.5
params['colsample_bytree'] = 0.5
      
classifier = XGBClassifier(**params)
                      
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Assigns predictions to respective observation windows
window_pred = test_data[['win_begin_idx', 'win_end_idx']].copy()
window_pred['pred'] = y_pred[:]

test_raw = pd.read_csv(root_dir + "test_raw.csv")
submission = test_raw.drop_duplicates().copy()                     
submission['idx'] = list(range(len(submission)))
submission = submission.set_index('idx')
sub_pred = 5*np.ones(len(submission))

for index, row in window_pred.iterrows():
    begin_idx = row['win_begin_idx'].item()
    end_idx = row['win_end_idx'].item() + 1
    sub_pred[begin_idx:end_idx] = row['pred']

submission.loc[:, 'pred'] = sub_pred[:]

activities = {0 : "Jogging", 1 : "LyingDown", 2 : "Sitting",
              3 : "Stairs" , 4 : "Standing",  5 : "Walking"}		 

for activity_id in range(6):
    submission.loc[submission['pred'] == activity_id, 'activity'] = activities[activity_id]
    
submission = submission.drop('pred', axis=1)
submission.to_csv(root_dir + 'submission.csv', index = False)

print("Total processing time: {:.2f} minutes".format((time.time()-start_time)/60))




























