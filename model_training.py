# -----------------------------------------------------------------
# Model training for human activity recognition

import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold

start_time = time.time()

root_dir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train_data = pd.read_csv(root_dir + "train.csv")     

X = train_data.loc[:, 'median_x':].values
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


#confusion_matrix(y_test, y_pred)




