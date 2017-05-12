# -----------------------------------------------------------------
# Model training for human activity recognition
# -----------------------------------------------------------------

#%% ------------------------------------------------------------------------
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

start_time = time.time()

root_dir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train_data = pd.read_csv(root_dir + "train.csv")     

X = train_data.loc[:, 'mean_x':].values
y = train_data.loc[:, 'activity_id'].values

classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs = -1)

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 314159)
scores = []
for train_index, test_index in skf.split(X, y):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

scores = np.array(scores)

print("Accuracy: {:.3f} (+/- {:.3f})".format(scores.mean(), scores.std()))
print("Total processing time: {:.2f} minutes".format((time.time()-start_time)/60))

#%% ------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
#      [[1101,    0,    4,    0,    4,   38],
#       [   0, 1020,   54,    0,    2,    3],
#       [   1,   46, 2174,    0,   31,   40],
#       [   6,    0,    1,  106,    0,   58],
#       [   0,    1,   83,    0,  721,   35],
#       [   5,    1,    6,    4,   22, 3688]]

# "Jogging"  : 0, 
# "LyingDown": 1,
# "Sitting"  : 2,
# "Stairs"   : 3,
# "Standing" : 4,
# "Walking"  : 5




