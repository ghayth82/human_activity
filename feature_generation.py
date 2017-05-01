# -----------------------------------------------------------------
# Data preprocessing for human activity recognition

import time
import pandas as pd
#import numpy as np

startTime = time.time()

rootDir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train = pd.read_csv(rootDir + "train_raw.csv")     

samplingRate = 20 # Hz
windowLength = 2 # seconds
windowSize = samplingRate * windowLength


activities = [{"act_type": "Jogging"    , "act_id": 0},
              {"act_type": "LyingDown"  , "act_id": 1},
              {"act_type": "Sitting"    , "act_id": 2},
              {"act_type": "Stairs"     , "act_id": 3},
              {"act_type": "Standing"   , "act_id": 4},
              {"act_type": "Walking"    , "act_id": 5}]

userIDs = train.id.unique()

columns = ['median_x', 'mean_x', 'std_x', 
           'median_y', 'mean_y', 'std_y', 
           'median_z', 'mean_z', 'std_z', 
           'activity_id']

features = pd.DataFrame(columns = columns)

for userID in userIDs[30:31]:
    beginIndex = 0
    for activity in activities:
        userData = train[train.id == userID]
        activityData = userData[userData.activity == activity['act_type']]
        if (activityData.shape[0] >= windowSize):
            activityData.sort_values('timestamp')
            endIndex = windowSize
            while(endIndex < activityData.shape[0]):
                windowData = activityData[beginIndex:endIndex]
                featValues = {'median_x'    : windowData['x'].median(),
                              'mean_x'      : windowData['x'].mean(), 
                              'std_x'       : windowData['x'].std(), 
                              'median_y'    : windowData['y'].median(),
                              'mean_y'      : windowData['y'].mean(), 
                              'std_y'       : windowData['y'].std(), 
                              'median_z'    : windowData['z'].median(),
                              'mean_z'      : windowData['z'].mean(), 
                              'std_z'       : windowData['z'].std(), 
                              'activity_id' : activity['act_id']}

                features.append(featValues)
        
                beginIndex = endIndex
                endIndex = endIndex + windowSize
                
features.to_csv(rootDir + 'train.csv', index = False)

print("Total processing time: {:.2f} minutes".format((time.time()-startTime)/60))

                














