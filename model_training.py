# --------------------------------------------------------------------------
# Model training for human activity recognition
# --------------------------------------------------------------------------

#%% ------------------------------------------------------------------------
# Load data

import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

root_dir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train_data = pd.read_csv(root_dir + "train.csv")
X = train_data.loc[:, 'mean_x':].values
y = train_data.loc[:, 'activity_id'].values
groups = train_data.loc[:, 'user_id'].values     
                       
train_flip_x_data = pd.read_csv(root_dir + "train_flip_x.csv")    
X_flip_x = train_flip_x_data.loc[:, 'mean_x':].values
y_flip_x = train_flip_x_data.loc[:, 'activity_id'].values
                       
train_flip_y_data = pd.read_csv(root_dir + "train_flip_y.csv")    
X_flip_y = train_flip_y_data.loc[:, 'mean_x':].values
y_flip_y = train_flip_y_data.loc[:, 'activity_id'].values

train_flip_z_data = pd.read_csv(root_dir + "train_flip_z.csv")    
X_flip_z = train_flip_z_data.loc[:, 'mean_x':].values
y_flip_z = train_flip_z_data.loc[:, 'activity_id'].values
                         

#%% -------------------------------------------------------------------------
# Overfitting control

start_time = time.time()

gkf = GroupKFold(n_splits=4)

ros = RandomOverSampler(random_state=31416)
nm = NearMiss(random_state=31416, ratio = 0.6, n_jobs=-1)
sm = SMOTE(random_state=31416, ratio = 0.6, k_neighbors=2, n_jobs=-1)

scores_test = []
scores_train = []
for train_index, test_index in gkf.split(X, y, groups):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_flip_x_train = X_flip_x[train_index]
    y_flip_x_train = y_flip_x[train_index]
    X_flip_y_train = X_flip_y[train_index]
    y_flip_y_train = y_flip_y[train_index]
    X_flip_z_train = X_flip_z[train_index]
    y_flip_z_train = y_flip_z[train_index]
    
    X_train_augm = np.concatenate((X_train, X_flip_x_train, X_flip_y_train, X_flip_z_train))
    y_train_augm = np.concatenate((y_train, y_flip_x_train, y_flip_y_train, y_flip_z_train))
    
    '''
    activity_filter = (y_train_augm==2) | (y_train_augm==5)
    X_res = X_train_augm[activity_filter]
    y_res = y_train_augm[activity_filter]
    X_train_augm = X_train_augm[~activity_filter]
    y_train_augm = y_train_augm[~activity_filter]
 
    X_res, y_res = nm.fit_sample(X_res, y_res) 
    X_train_augm = np.concatenate((X_train_augm, X_res))
    y_train_augm = np.concatenate((y_train_augm, y_res))

    activity_filter = (y_train_augm==1) | (y_train_augm==4)
    X_res = X_train_augm[activity_filter]
    y_res = y_train_augm[activity_filter]
    X_train_augm = X_train_augm[~activity_filter]
    y_train_augm = y_train_augm[~activity_filter]
 
    X_res, y_res = sm.fit_sample(X_res, y_res) 
    X_train_augm = np.concatenate((X_train_augm, X_res))
    y_train_augm = np.concatenate((y_train_augm, y_res))
    '''

    #X_train_augm, y_train_augm = X_train, y_train
    
    param = dict()
    param['objective'] = 'multi:softmax'
    param['num_class'] = 6
    param['updater'] = 'grow_gpu_hist'
    param['max_depth'] = 4
    param['min_child_weight'] = 20
    param['gamma'] = 10
    param['reg_lambda'] = 1
    param['reg_alpha'] = 0.0
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.5
    param['learning_rate'] = 0.1
    num_round = 10

    dtrain = xgb.DMatrix(X_train_augm, y_train_augm)
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
print()
print("Total processing time: {:.2f} seconds".format((time.time()-start_time)))


#%% ------------------------------------------------------------------------
# Fold tests

gkf = list(GroupKFold(n_splits=4).split(X, y, groups))

train_index, test_index = gkf[0]

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

X_flip_x_train = X_flip_x[train_index]
y_flip_x_train = y_flip_x[train_index]
X_flip_y_train = X_flip_y[train_index]
y_flip_y_train = y_flip_y[train_index]
X_flip_z_train = X_flip_z[train_index]
y_flip_z_train = y_flip_z[train_index]

X_train_augm = np.concatenate((X_train, X_flip_x_train, X_flip_y_train, X_flip_z_train))
y_train_augm = np.concatenate((y_train, y_flip_x_train, y_flip_y_train, y_flip_z_train))

param = dict()
param['objective'] = 'multi:softmax'
param['num_class'] = 6
param['updater'] = 'grow_gpu_hist'
param['max_depth'] = 4
param['min_child_weight'] = 20
param['gamma'] = 15
param['reg_lambda'] = 1
param['reg_alpha'] = 0.0
param['subsample'] = 0.5
param['colsample_bytree'] = 0.5
param['learning_rate'] = 0.1
num_round = 10

dtrain = xgb.DMatrix(X_train_augm, y_train_augm)
dtest  = xgb.DMatrix(X_test, y_test)
deval = xgb.DMatrix(X_train, y_train)
evallist  = [(dtest,'test'), (deval,'train')]
classifier = xgb.train(param, dtrain, num_round, evals = evallist)

dpred = xgb.DMatrix(X_test)
y_pred = classifier.predict(dpred)


#%% ------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix

# "Jogging"  : 0, 
# "LyingDown": 1,
# "Sitting"  : 2,
# "Stairs"   : 3,
# "Standing" : 4,
# "Walking"  : 5

confusion_matrix(y_test, y_pred)

#Fold 0
#[[ 638,    0,    0,    0,  117],
# [   0,    1,  104,    4,    4],
# [   0,   12,  772,   12,   31],
# [   0,    0,  319,  108,    7],
# [   1,    0,    0,   11, 1566]])

#Fold 1    
#[[ 718,    0,    0,    0,    0,   21],
# [   0,    2,  188,    0,    0,    8],
# [   0,    3,  281,    0,   41,   59],
# [   0,    0,    0,    0,    0,  104],
# [   0,    0,   98,    1,  127,   50],
# [ 118,   11,  151,    1,   10, 1715]])    
    
    

#%%------------------------------------------------------------------------- #
# Parameter tuning

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

start_time = time.time()

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

gkf = list(GroupKFold(n_splits=4).split(X, y, groups))

#Best parameters set found:
#----------------------------
#Parameters: {'gamma': 10, 'max_depth': 3, 'min_child_weight': 10}
#Accuracy: 0.800 +/- 0.022
#Total processing time: 41.14 minutes (144 sets)

param_grid = dict()
#param_grid['n_estimators'] = [25, 50, 75, 100, 200] 
#param_grid['learning_rate'] = [0.05, 0.1, 0.2, 0.3]
#param_grid['max_depth'] = [3, 4]
param_grid['subsample'] = [0.4, 0.6, 0.8]
param_grid['colsample_bytree'] = [0.4, 0.6, 0.8]
#param_grid['min_child_weight'] = [5, 10, 15]
#param_grid['gamma'] = [5, 10, 15]
#param_grid['reg_alpha'] = [0, 0.001, 0.003, 0.01, 0.03]

classifier = GridSearchCV(classifier, param_grid, scoring = 'accuracy', cv = gkf, n_jobs = -1) 
classifier.fit(X, y)

print("Best parameters set found:")
print("----------------------------")
print("Parameters: " + str(classifier.best_params_))
print("Accuracy: {:0.3f}".format(classifier.best_score_))
print("Total processing time: {:.2f} minutes".format((time.time()-start_time)/60))
print()
print("Grid scores:")
print("-----------------------------------------------")
means = classifier.cv_results_['mean_test_score']
stds = classifier.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
    print("{:0.3f} (+/-{:0.3f}) for {}".format(mean, std, params))



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
sub_pred = np.zeros(len(submission))

for index, row in window_pred.iterrows():
    begin_idx = row['win_begin_idx']
    end_idx = row['win_end_idx'] + 1
    sub_pred[begin_idx:end_idx] = row['pred']

submission.loc[:, 'pred'] = sub_pred[:]

activities = {0 : "Jogging", 1 : "LyingDown", 2 : "Sitting",
              3 : "Stairs" , 4 : "Standing",  5 : "Walking"}		 

for activity_id in range(6):
    submission.loc[submission['pred'] == activity_id, 'activity'] = activities[activity_id]
    
submission = submission.drop('pred', axis=1)
submission.to_csv(root_dir + 'submission.csv', index = False)

print("Total processing time: {:.2f} minutes".format((time.time()-start_time)/60))


#%% ------------------------------------------------------------------------
# Over Sampling tests
from collections import Counter
from imblearn.over_sampling import RandomOverSampler 


classifier.fit(X_res, y_res)
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)


submission = pd.read_csv(root_dir + 'submission.csv')


Counter(submission['activity'])['Jogging']/len(submission)
Counter(submission['activity'])['LyingDown']/len(submission)
Counter(submission['activity'])['Sitting']/len(submission)
Counter(submission['activity'])['Stairs']/len(submission)
Counter(submission['activity'])['Standing']/len(submission)
Counter(submission['activity'])['Walking']/len(submission)

# "Jogging"  : 11% 
# "LyingDown": 5%
# "Sitting"  : 15%
# "Stairs"   : 1.3%
# "Standing" : 15%
# "Walking"  : 53%
































