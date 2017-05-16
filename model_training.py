# -----------------------------------------------------------------
# Model training for human activity recognition
# -----------------------------------------------------------------

#%% ------------------------------------------------------------------------
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from imblearn.over_sampling import RandomOverSampler 

start_time = time.time()

root_dir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train_data = pd.read_csv(root_dir + "train.csv")     

X = train_data.loc[:, 'mean_x':].values
y = train_data.loc[:, 'activity_id'].values
groups = train_data.loc[:, 'user_id'].values        
                       
#ros = RandomOverSampler(random_state=314159)

gkf = GroupKFold(n_splits=5)

#train_index, test_index = next(gkf.split(X, y, groups))

scores = []
for train_index, test_index in gkf.split(X, y, groups):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #X_res, y_res = ros.fit_sample(X_train, y_train)
    #classifier.fit(X_res, y_res)
    param = {'objective' : 'multi:softmax', 'num_class' : 6}    
    param['updater'] = 'grow_gpu_hist'

    dtrain = xgb.DMatrix(X_train, y_train)
    #dtrain = xgb.DMatrix(X_res, y_res)
    classifier = xgb.train(param, dtrain)                       

    #classifier.fit(X_train, y_train)
    dtest = xgb.DMatrix(X_test)
    y_pred = classifier.predict(dtest)
    scores.append(accuracy_score(y_test, y_pred))

scores = np.array(scores)

print("Accuracy: {:.3f} (+/- {:.3f})".format(scores.mean(), scores.std()))
print("Total processing time: {:.2f} minutes".format((time.time()-start_time)/60))

#%% ------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

#[[   0,    0,    0,    0,    0,    0],
# [   0,    0, 1885,    0, 1803,    1],
# [   0,   34, 3072,    8, 1599,  119],
# [   0,    0,    0,    1,    0,   35],
# [   0,    0,   11,    0,  375,    9],
# [   1,    0,    0,    1,    1,  308]]

#[ 0.44521213,  0.75847915,  0.75270095,  0.74770394,  0.83623204]

# "Jogging"  : 0, 
# "LyingDown": 1,
# "Sitting"  : 2,
# "Stairs"   : 3,
# "Standing" : 4,
# "Walking"  : 5


#%% ------------------------------------------------------------------------
# Submission generation

import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

start_time = time.time()

root_dir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train_data = pd.read_csv(root_dir + "train.csv")    
X_train = train_data.loc[:, 'mean_x':].values
y_train = train_data.loc[:, 'activity_id'].values

test_data = pd.read_csv(root_dir + "test.csv")    
X_test = test_data.loc[:, 'mean_x':].values

classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs = -1)
ros = RandomOverSampler(random_state=314159)

X_res, y_res = ros.fit_sample(X_train, y_train)
classifier.fit(X_res, y_res)
y_pred = classifier.predict(X_test)

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
































