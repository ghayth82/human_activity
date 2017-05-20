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

root_dir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train_data = pd.read_csv(root_dir + "train.csv")    

X = train_data.loc[:, 'mean_x':].values
y = train_data.loc[:, 'activity_id'].values
groups = train_data.loc[:, 'user_id'].values        

                       
#%% -------------------------------------------------------------------------
# Overfitting control

gkf = GroupKFold(n_splits=4)

scores_test = []
scores_train = []
for train_index, test_index in gkf.split(X, y, groups):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    param = dict()
    param['objective'] = 'multi:softmax'
    param['num_class'] = 6
    param['updater'] = 'grow_gpu_hist'
    param['max_depth'] = 3
    param['min_child_weight'] = 10
    param['gamma'] = 10
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.5
    param['learning_rate'] = 0.1
    num_round = 100

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
# Tests

gkf = GroupKFold(n_splits=5)
i=0
for train_index, test_index in gkf.split(X, y, groups):
    i = i+1
    if i == 2: break

Counter(groups[test_index])
Counter(y[test_index])

{0: 170, 1: 90,  2: 911, 3: 152, 4: 439, 5: 1203}
{0: 393, 1: 240, 2: 440, 3: 70,  4: 255, 5: 1568}
{0: 747, 1: 120, 2: 316,         4: 216, 5: 1566}
{0: 419, 1: 76,  2: 534, 3: 58,  4: 239, 5: 1635}
{0: 564, 1: 159, 2: 456, 3: 48,  4: 368, 5: 1368}

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

dtrain = xgb.DMatrix(X_train, y_train)
dtest  = xgb.DMatrix(X_test, y_test)
evallist  = [(dtest,'test'), (dtrain,'train')]
classifier = xgb.train(param, dtrain, 10, evals = evallist)
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

from xgboost import plot_importance
from matplotlib import pyplot

plot_importance(classifier)
pyplot.show()

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
































