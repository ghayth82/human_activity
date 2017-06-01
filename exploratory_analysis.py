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

period_anomaly = time_interval[(time_interval>60) & (time_interval<360)]
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

period_anomaly = time_interval[(time_interval>=49.99) & (time_interval<50.01)]
period_anomaly.hist()


period_anomaly[:100]

train_raw.loc[1000:1200]

'''
Sampling anomalies

train 1,851,580
t > 350ms : 1,704 (1,642) 0.09%
60ms < t < 350ms : 77,975 4.21%
50ms < t < 60ms : 69,523 3.75%
40ms < t < 50ms : 50,818 2.74%
5ms < t < 40ms : 2,522 0.14%
0ms < t < 5ms : 15,247 0.82%

test 767,756
t > 350ms : 264 (281) 0.04%
60ms < t < 350ms : 34,974 (34,950) 4.6%
50ms < t < 60ms : 34,671 (33,813) 4.52%
40ms < t < 50ms : 28,303 (29,118) 3.69%
5ms < t < 40ms : 676 (8,703) 0.09%
0ms < t < 5ms : 7,130 (8,249) 0.93%

'''    
    
train_interp = train_raw.loc[11430:11480].copy()

train_interp = train_interp.set_index(['timestamp'], drop = False)

t_stamps = train_interp['timestamp'].values
t_stamps_new = list()                       

t_previous = t_stamps[0]

for j in range(len(t_stamps)):
    if(((t_stamps[j] - t_previous)>60) & ((t_stamps[j] - t_previous)<360)):
        t_previous = t_previous + 50
        while (t_stamps[j] - t_previous) > 40:
            t_stamps_new.append(t_previous)
            t_previous = t_previous + 50
       
    t_stamps_new.append(t_stamps[j])
    t_previous = t_stamps[j]
            
train_interp = train_interp.reindex(t_stamps_new)                       

train_interp.timestamp = train_interp.index.values
train_interp[['id', 'activity']] = train_interp[['id', 'activity']].ffill()   
train_interp[['x', 'y', 'z']] = train_interp[['x', 'y', 'z']].interpolate()                       
train_interp.id = train_interp.id.astype(int)
                       
submission = pd.read_csv(root_dir + "classy_fyers_final_submission.csv")                      
                       
sum(submission.ground_truth == submission.activity)/len(submission)

                       
submission['mag'] = (submission['x']**2 + submission['y']**2 + submission['z']**2)**0.5
          
submission = submission[submission.mag>5]          
         
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       


