# -----------------------------------------------------------------
# Data preprocessing for human activity recognition

import time
import pandas as pd

startTime = time.time()

rootDir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train_raw = pd.read_csv(rootDir + "train_raw.csv")     

samplingRate = 20 # Hz
windowLength = 2 # seconds
windowSize = samplingRate * windowLength

activities = [{"act_type": "Jogging"    , "act_id": 0},
              {"act_type": "LyingDown"  , "act_id": 1},
              {"act_type": "Sitting"    , "act_id": 2},
              {"act_type": "Stairs"     , "act_id": 3},
              {"act_type": "Standing"   , "act_id": 4},
              {"act_type": "Walking"    , "act_id": 5}]

userIDs = train_raw.id.unique()

columns = ['median_x', 'mean_x', 'std_x', 
           'median_y', 'mean_y', 'std_y', 
           'median_z', 'mean_z', 'std_z', 
           'activity_id']

features = []
for userID in userIDs:
    userData = train_raw[train_raw.id == userID]
    for activity in activities:
        activityData = userData[userData.activity == activity['act_type']]
        activityData.sort_values('timestamp')

        beginIndex = 0
        endIndex = windowSize
        while(endIndex <= activityData.shape[0]):
            windowData = activityData[beginIndex:endIndex]
            
            features.append({'median_x'    : windowData['x'].median(),
                             'mean_x'      : windowData['x'].mean(), 
                             'std_x'       : windowData['x'].std(), 
                             'median_y'    : windowData['y'].median(),
                             'mean_y'      : windowData['y'].mean(), 
                             'std_y'       : windowData['y'].std(), 
                             'median_z'    : windowData['z'].median(),
                             'mean_z'      : windowData['z'].mean(), 
                             'std_z'       : windowData['z'].std(), 
                             'activity_id' : activity['act_id']})

            beginIndex = endIndex
            endIndex = endIndex + windowSize
                
train = pd.DataFrame(features, columns = columns)
train.to_csv(rootDir + 'train.csv', index = False)

print("Total processing time: {:.2f} minutes".format((time.time()-startTime)/60))

                














