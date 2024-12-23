import pm4py
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt

BPIC_2015_1 = pd.read_csv('.\dataset\BPIC15_1_sorted.csv')
BPIC_2015_2 = pd.read_csv('.\dataset\BPIC15_2_sorted.csv')
BPIC_2015_3 = pd.read_csv('.\dataset\BPIC15_3_sorted.csv')
BPIC_2015_4 = pd.read_csv('.\dataset\BPIC15_4_sorted.csv')
BPIC_2015_5 = pd.read_csv('.\dataset\BPIC15_5_sorted.csv')


Sepsis_Log = pd.read_csv('.\dataset\sepsis_cases_2_Trunc13.csv')
BPIC_2017 = pm4py.read_xes('.\dataset\BPI Challenge 2017.xes')
BPIC_2017 = pd.read_csv('.\dataset\BPI Challenge 2017.csv')
BPIC_2019 = pm4py.read_xes('.\dataset\BPI_Challenge_2019.xes')
traffic_fines = pm4py.read_xes('.\dataset\Road_Traffic_Fine_Management_Process.xes')

traffic_fines = traffic_fines[['case:concept:name', 'concept:name', 'time:timestamp', 'paymentAmount']]
traffic_fines = traffic_fines[['case:concept:name', 'concept:name', 'time:timestamp', 'paymentAmount']]
max_payments = traffic_fines[(traffic_fines['paymentAmount'] > 0) | (traffic_fines['paymentAmount'].isna())].groupby('case:concept:name')['paymentAmount'].max()
traffic_fines['paymentAmount'] = traffic_fines['case:concept:name'].map(max_payments).fillna(traffic_fines['paymentAmount'])
traffic_fines['paymentAmount'].fillna(0, inplace=True)
case_has_payment = traffic_fines.groupby('case:concept:name')['paymentAmount'].transform(lambda x: 1 if x.gt(0).any() else 0)
traffic_fines['outcome'] = case_has_payment


traffic_fines['case:concept:name'][traffic_fines['outcome'] == 1].nunique() / traffic_fines['case:concept:name'].nunique()



BPIC_2015_1['Complete Timestamp'] = pd.to_datetime(BPIC_2015_1['Complete Timestamp'])
BPIC_2015_2['Complete Timestamp'] = pd.to_datetime(BPIC_2015_2['Complete Timestamp'])
BPIC_2015_3['Complete Timestamp'] = pd.to_datetime(BPIC_2015_3['Complete Timestamp'])
BPIC_2015_4['Complete Timestamp'] = pd.to_datetime(BPIC_2015_4['Complete Timestamp'])
BPIC_2015_5['Complete Timestamp'] = pd.to_datetime(BPIC_2015_5['Complete Timestamp'])

BPIC_2017 = BPIC_2017[['case:concept:name', 'concept:name', 'time:timestamp', 'case:RequestedAmount']]

BPIC_2017[BPIC_2017['concept:name'].isin(['O_Refused', 'O_Cancelled', 'A_Cancelled'])]
BPIC_2017 = BPIC_2017[['case:concept:name', 'concept:name', 'time:timestamp', 'case:RequestedAmount']]
BPIC_2015_11 = BPIC_2015_1[['Case ID', 'activityNameEN', 'Complete Timestamp']]
BPIC_2015_11['activityNameEN'] = BPIC_2015_11['activityNameEN'].astype(str)
BPIC_2015_11['Case ID'] = BPIC_2015_11['Case ID'].astype(str)

BPIC_2019 = BPIC_2019[['case:concept:name', 'time:timestamp', 'concept:name']]


BPIC_2019['time:timestamp'] = pd.to_datetime(BPIC_2019['time:timestamp'])
BPIC_2019 = BPIC_2019[['case:concept:name', 'time:timestamp', 'concept:name']]
BPIC_2019['total_duration'] = (
    BPIC_2019.groupby('case:concept:name')['time:timestamp']
    .transform(lambda x: x.max() - x.min())
)

BPIC_2019 = BPIC_2019.sort_values(by=['case:concept:name', 'time:timestamp'])
BPIC_2019['next_timestamp'] = BPIC_2019.groupby('case:concept:name')['time:timestamp'].shift(-1)
BPIC_2019['task_duration'] = (BPIC_2019['next_timestamp'] - BPIC_2019['time:timestamp']).dt.total_seconds()
BPIC_2019['task_duration'] = BPIC_2019['task_duration'].fillna(0)
BPIC_2019 = BPIC_2019.drop(columns=['next_timestamp'])
BPIC_2019['total_duration'] = pd.to_numeric(BPIC_2019['total_duration'].dt.total_seconds())
df_case_durations = BPIC_2019.groupby('case:concept:name')['total_duration'].max().reset_index()
df_case_durations['total_duration'] = pd.to_numeric(df_case_durations['total_duration'].dt.total_seconds())
KPI = np.average(df_case_durations['total_duration'])

BPIC_2019['outcome'] = BPIC_2019['total_duration'].apply(lambda x: 1 if x > KPI else 0)

BPIC_2019['case:concept:name'][BPIC_2019['outcome'] == 1].nunique() / BPIC_2019['case:concept:name'].nunique()

BPIC_2015_1 = BPIC_2015_1[['Case ID', 'activityNameEN', 'Complete Timestamp']]
BPIC_2015_2 = BPIC_2015_2[['Case ID', 'activityNameEN', 'Complete Timestamp']]
BPIC_2015_3 = BPIC_2015_3[['Case ID', 'activityNameEN', 'Complete Timestamp']]
BPIC_2015_4 = BPIC_2015_4[['Case ID', 'activityNameEN', 'Complete Timestamp']]
BPIC_2015_5 = BPIC_2015_5[['Case ID', 'activityNameEN', 'Complete Timestamp']]

case1_has_request_complete = BPIC_2015_1.groupby('Case ID')['activityNameEN'].transform(lambda x: 0 if 'request complete' in x.values else 1)
case2_has_request_complete = BPIC_2015_2.groupby('Case ID')['activityNameEN'].transform(lambda x: 0 if 'request complete' in x.values else 1)
case3_has_request_complete = BPIC_2015_3.groupby('Case ID')['activityNameEN'].transform(lambda x: 0 if 'request complete' in x.values else 1)
case4_has_request_complete = BPIC_2015_4.groupby('Case ID')['activityNameEN'].transform(lambda x: 0 if 'request complete' in x.values else 1)
case5_has_request_complete = BPIC_2015_5.groupby('Case ID')['activityNameEN'].transform(lambda x: 0 if 'request complete' in x.values else 1)



BPIC_2015_1['outcome'] = case1_has_request_complete
BPIC_2015_2['outcome'] = case2_has_request_complete
BPIC_2015_3['outcome'] = case3_has_request_complete
BPIC_2015_4['outcome'] = case4_has_request_complete
BPIC_2015_5['outcome'] = case5_has_request_complete


BPIC_2015_1 = BPIC_2015_1.rename(columns={'activityNameEN': 'concept:name'})
BPIC_2015_1 = BPIC_2015_1.rename(columns={'Case ID': 'case:concept:name'})
BPIC_2015_1 = BPIC_2015_1.rename(columns={'time:timestamp\t': 'time:timestamp'})

BPIC_2015_2 = BPIC_2015_2.rename(columns={'activityNameEN': 'concept:name'})
BPIC_2015_2 = BPIC_2015_2.rename(columns={'Case ID': 'case:concept:name'})
BPIC_2015_2 = BPIC_2015_2.rename(columns={'time:timestamp\t': 'time:timestamp'})

BPIC_2015_3 = BPIC_2015_3.rename(columns={'activityNameEN': 'concept:name'})
BPIC_2015_3 = BPIC_2015_3.rename(columns={'Case ID': 'case:concept:name'})
BPIC_2015_3 = BPIC_2015_3.rename(columns={'time:timestamp\t': 'time:timestamp'})

BPIC_2015_4 = BPIC_2015_4.rename(columns={'activityNameEN': 'concept:name'})
BPIC_2015_4 = BPIC_2015_4.rename(columns={'Case ID': 'case:concept:name'})
BPIC_2015_4 = BPIC_2015_4.rename(columns={'time:timestamp\t': 'time:timestamp'})

BPIC_2015_5 = BPIC_2015_5.rename(columns={'activityNameEN': 'concept:name'})
BPIC_2015_5 = BPIC_2015_5.rename(columns={'Case ID': 'case:concept:name'})
BPIC_2015_5 = BPIC_2015_5.rename(columns={'time:timestamp\t': 'time:timestamp'})



Sepsis_Log = Sepsis_Log[['case:concept:name', 'concept:name', 'time:timestamp', 'label']]
BPIC_2017['outcome'] = BPIC_2017.groupby('case:concept:name')['concept:name'].transform(lambda x: 1 if x.isin(['O_Refused', 'O_Cancelled', 'A_Cancelled']).any() else 0)


from sklearn.model_selection import train_test_split
import os

def train_test_split_equal_outcome(df, outcome_column='outcome', test_size=0.2, random_state=None, output_dir='./'):
    cases_outcome_1 = df[df[outcome_column] == 1].groupby('case:concept:name').first().reset_index()
    cases_outcome_0 = df[df[outcome_column] == 0].groupby('case:concept:name').first().reset_index()
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')
    dataset_name = [name for name, value in globals().items() if value is df][0]
    cases_outcome_1_train, cases_outcome_1_test = train_test_split(cases_outcome_1, test_size=test_size, random_state=random_state)
    cases_outcome_0_train, cases_outcome_0_test = train_test_split(cases_outcome_0, test_size=test_size)
    
    cases_train = pd.concat([cases_outcome_1_train, cases_outcome_0_train])
    cases_test = pd.concat([cases_outcome_1_test, cases_outcome_0_test])

    train_cases = df[df['case:concept:name'].isin(cases_train['case:concept:name'])]
    test_cases = df[df['case:concept:name'].isin(cases_test['case:concept:name'])]

    train_cases = train_cases.sort_values(by=['case:concept:name', 'time:timestamp']).reset_index(drop=True)
    test_cases = test_cases.sort_values(by=['case:concept:name', 'time:timestamp']).reset_index(drop=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    
    train_filename = os.path.join(output_dir, f'{dataset_name}_train.csv')
    test_filename = os.path.join(output_dir, f'{dataset_name}_test.csv')

    train_cases.to_csv(train_filename, index=False)
    test_cases.to_csv(test_filename, index=False)
    
train_test_split_equal_outcome(BPIC_2017, outcome_column='outcome', test_size=0.2, random_state=42, output_dir='./evaluation_dataset')
train_test_split_equal_outcome(Sepsis_Log, outcome_column='label', test_size=0.2, random_state=42, output_dir='./evaluation_dataset')
train_test_split_equal_outcome(BPIC_2015_1, outcome_column='outcome', test_size=0.2, random_state=42, output_dir='./evaluation_dataset')
train_test_split_equal_outcome(BPIC_2015_2, outcome_column='outcome', test_size=0.2, random_state=42, output_dir='./evaluation_dataset')
train_test_split_equal_outcome(BPIC_2015_3, outcome_column='outcome', test_size=0.2, random_state=42, output_dir='./evaluation_dataset')
train_test_split_equal_outcome(BPIC_2015_4, outcome_column='outcome', test_size=0.2, random_state=42, output_dir='./evaluation_dataset')
train_test_split_equal_outcome(BPIC_2015_5, outcome_column='outcome', test_size=0.2, random_state=42, output_dir='./evaluation_dataset')
train_test_split_equal_outcome(BPIC_2019, outcome_column='outcome', test_size=0.2, random_state=42, output_dir='./evaluation_dataset')
train_test_split_equal_outcome(traffic_fines, outcome_column='outcome', test_size=0.2, random_state=42, output_dir='./evaluation_dataset')



    