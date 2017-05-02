# -----------------------------------------------------------------
# Model training for human activity recognition

import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score


startTime = time.time()

rootDir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

trainData = pd.read_csv(rootDir + "train.csv")     

y = trainData.pop('activity_id')      
X = trainData           

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify = y, random_state=27182)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27182)

#X_train = StandardScaler().fit_transform(X_train)

classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced')
classifier.fit(X_train, y_train)

#X_test = StandardScaler().fit_transform(X_test)

y_pred = classifier.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

scores = cross_val_score(classifier, X, y, cv=5, scoring = 'accuracy', n_jobs=-1)
print("Accuracy: {:.3f} (+/- {:.3f})".format(scores.mean(), scores.std()))


print("Total processing time: {:.2f} minutes".format((time.time()-startTime)/60))


