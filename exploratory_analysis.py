# -----------------------------------------------------------------
# Exploratory analysis for human activity recognition

import pandas as pd

root_dir = "C:/Users/rarez/Documents/Data Science/human_activity/data/"

train_raw = pd.read_csv(root_dir + "train_raw.csv")     
train_raw = train_raw.drop_duplicates()
#train_raw = train_raw.sort_values(['id', 'activity', 'timestamp'])

test_raw = pd.read_csv(root_dir + "test_raw.csv")
test_raw = test_raw.drop_duplicates()
#test_raw = test_raw.sort_values(['timestamp'])

time_interval = train_raw['timestamp'].diff()
time_interval = test_raw['timestamp'].diff()

period_anomaly = time_interval[time_interval>350]

period_anomaly = time_interval[(time_interval>60) & (time_interval<350)]
period_anomaly.hist()

period_anomaly = time_interval[(time_interval>50) & (time_interval<=60)]
period_anomaly.hist()

period_anomaly = time_interval[(time_interval>40) & (time_interval<50)]
period_anomaly.hist()

period_anomaly = time_interval[(time_interval>5) & (time_interval<=40)]
period_anomaly.hist()

period_anomaly = time_interval[(time_interval>=0) & (time_interval<=5)]
period_anomaly.hist()

period_anomaly = time_interval[(time_interval>=0) & (time_interval<1)]
period_anomaly.hist()

period_anomaly[:100]

train_raw.loc[2750:2780]

'''
Sampling anomalies

train 1,851,580
t > 350ms : 1,704 (1,642) 0.09%
60ms < t < 350ms : 77,975 (70,125) 4.21%
50ms < t < 60ms : 69,523 (69,004) 3.75%
40ms < t < 50ms : 50,818 (50,820) 2.74%
5ms < t < 40ms : 2,522 (2,343) 0.14%
0ms < t < 5ms : 15,247 (~55K) 0.82%

test 767,756
t > 350ms : 264 (281) 0.04%
60ms < t < 350ms : 34,974 (34,950) 4.6%
50ms < t < 60ms : 34,671 (33,813) 4.52%
40ms < t < 50ms : 28,303 (29,118) 3.69%
5ms < t < 40ms : 676 (8,703) 0.09%
0ms < t < 5ms : 7,130 (8,249) 0.93%

'''    
    








