'''
Process the raw JSON from the VES collector into a
dataframe and save to a csv. That csv can be loaded 
from the NRDataset.py file to make the torch dataset. 
'''
#TODO: Add a data-alignment function to combine ves measurments with Amarisoft UE data 

import sys
import time
import pandas as pd 
import json
from pathlib import Path
import numpy as np
from sklearn import preprocessing
import argparse
from functools import reduce
from datetime import datetime
import pytz

def get_ves_labels(ves_events):
    '''
    Iterates through the first ~1000 ves messages to get a complete list of labels.
    The measurments for a single timestamp come in a few chunks so dynamically aquiring labels 
    has to be done over a sufficient set of items in the JSON object pulled from the message router.
    '''
    # Timestamp is the index of each row
    feature_labels = ['timestamp']
    event_set = ves_events[:1000]
    for event in event_set:
        for objectInstance in event['event']['measurementsForVfScalingFields']['additionalObjects']:
            # print(objectInstance)
            # List of all top level metrics 
            for object_ in objectInstance['objectInstances']:
                feature_label = objectInstance['objectName'] + ' ' + object_['objectKeys'][0]['keyValue']
                feature_labels.append(feature_label)
    return feature_labels


def standardize_timecodes(timestamp: str):
    # Parse the month, day, year, time-of-day into a datetime object then UTC timestamp 
    time_list = timestamp.split()
    month = int(time_list[1])
    day = int(time_list[2])
    year = int(time_list[3])
    tod = time_list[4]
    hour = int(time_list[4].split(':')[0]) #+ 12 # This is an unfortunate assumption for measurments taken after noon... 
    minute = int(time_list[4].split(':')[1])
    second = int(time_list[4].split(':')[2])

    time = datetime(year, month, day, hour, minute, second, tzinfo=pytz.timezone('utc'))
    timestamp = time.timestamp()
    return timestamp


def ves2csv(path: Path, mal_id: int = 0):
    '''
    Process the JSON into a dataframe 

    json: VES events 
    path: name of the json file - will also be name of saved csv 
    mal_id: The benign/malicious label for this set of samples (0: benign, 1: malicious)
    '''
    # Load data from json
    data = json.loads(path.read_text(encoding="UTF-8"))
    # Static labels and data structures
    feature_labels = get_ves_labels(data)
    # print(feature_labels)
    data_dict = {key:[] for key in feature_labels}

    # Loop through collected events 
    for event in data:
        # Process the timecode into a UTC timestamp
        collector_timestamp = event['event']['commonEventHeader']['internalHeaderFields']['collectorTimeStamp']
        data_dict['timestamp'] += [standardize_timecodes(collector_timestamp)]
        for objectInstance in event['event']['measurementsForVfScalingFields']['additionalObjects']:
            # List of all top level metrics 
            for object_ in objectInstance['objectInstances']:
                feature_label = objectInstance['objectName'] + ' ' + object_['objectKeys'][0]['keyValue']
                data_dict[feature_label] += object_['objectInstance'].values()
                # feature_value = object_['objectInstance']
        for key, values in data_dict.items():
            if len(values) < len(data_dict['timestamp']):
                data_dict[key].append('NaN')

    # Process the data to consolidate samples from the same time into a single entry 
    df = pd.DataFrame.from_dict(data_dict)
    df = df.replace('NaN', None)
    df = df.infer_objects(copy=False).fillna(0)
    df = df.groupby(['timestamp']).sum()
    # Add benign/malicious labels
    df['label'] = [mal_id] * len(df.iloc[:, 0])
    # Write to csv
    # df.to_csv('data/' + path.stem + '.csv')
    df.head()
    return df


def combine_csv(frames: list, saveFile: str = 'data/NR_dataset.csv'):
    # Combines a list of dataframes into a single dataframe and saves to csv
    df_full = pd.concat(frames)
    df_full.to_csv(saveFile)
    return df_full


def split_slices(df, slice_labels, target_slice_sd):
    '''
    Function to split VES samples into a distinct sample for each slice

    ves2csv makes a dataset where each VES event is one sample. Each VES event
    contains data from all the slices on the core network. This function takes 
    the SNSSAI and corresponding UPF IP of each slice and splits each sample into
    a set of samples where each only contains data for one slice. Think of this 
    as a full core dataset -> slice specific dataset conversion. 

    slice_labels: List of tuples. Each tuple should contain the snssai of a slice and corresponding upf ip addr
    Ex: [(1-111111, 10.1.0.138:9090)]
    '''
    columns = df.columns 
    slice_dataframes = []
    for label in slice_labels: 
        target_columns = []
        for col in columns:
            if any(x in col for x in label): # | (not any(x in col for x in list(sum(slice_labels, ())) )):
                target_columns.append(col)
        # Select the target columns based on slice id
        df_subset = df.loc[:,target_columns]
        if target_slice_sd in label[0]: 
            # Malicious labels
            # df_subset['label'] = df['label']
            df_subset.loc[:,'label'] = df['label']
        else: 
            # df_subset['label'] = df['label'].values[:] * 0
            df_subset.loc[:,'label'] = df.loc[:,'label'] * 0
        # Strip the slice id from column names 
        col_names = {col: col.split()[0] if any(x in col for x in list(sum(slice_labels, ())) ) else col for col in target_columns}
        df_subset.rename(columns=col_names, inplace=True)
        # Add a label column to keep track of the slice id 
        df_subset['slice_id'] = label[0]
        slice_dataframes.append(df_subset)

    # Verically stack the component dataframes 
    df_slices = pd.concat(slice_dataframes, axis=0)
    return df_slices


def modify_labels(df, tup, tdown):
    '''
    Technically the attack is in process while the UEs swing off of the target slice, but I want
    to test the detector when load is being applied to the target slice separately. The detector
    is really struggling to identify an attack on individual slices currently and I think this 
    might be the reason. The attack signature doesn't show up on individual slices when they 
    aren't under load. 
    ''' 
    # Set appropriate start value - some of the data files have a few random samples at the start that were in the buffer before collection began 
    # We need to skip those samples 
    mal_sample_indices = df.index[df['label'] == 1].tolist()
    for i in mal_sample_indices:
        start_time = int(df['timestamp'][i])
        time_diff_error = int(df['timestamp'][i+20]) - (start_time + 100)
        if time_diff_error < 5: 
            break
    sim_duration = 1800
    time_elapsed = 0 
    # Iterate through timestamps to set the malicious times to correspond with the attack timing 
    while sim_duration > time_elapsed: 
        time_range = range(start_time, start_time + tdown)
        indicies = [index for index, timestamp in enumerate(df['timestamp']) if timestamp in time_range]
        # if index is in the time range set the label to 0 
        df.loc[indicies, 'label'] = 0
        time_elapsed = time_elapsed + tup + tdown 
        start_time = start_time + tup + tdown

    df.to_csv('slice-dataset-mitigated-ban-mod-labels.csv')
    return df



def make_ue_df(ue_sample_dir: Path, data_filter: list = None, mal_ue_list: list = [], malicious: bool = False):
    '''
    UE samples are collected across separate directories. The sub-directories
    are for different stats and the main directories are for different UE IMSIs. 

    ue_sample_dir: The main directory containing all the UE files
    data_filter: A list of strings that you can use to filter for only the date you're interested in 
    mal_ue_list: list of malicious UE IMSIs for labeling 
    malicious: Bool value used to set the UEs in mal_ue_list to have a label of 1 
    '''
    total_df = pd.DataFrame() 

    # Iterate through all UE folders 
    for pth in ue_sample_dir.iterdir():
        imsi = pth.stem.split('-')[1]
        # Iterate through each data file in a UE folder 
        for p in pth.iterdir():
            # Filter data if specified 
            if data_filter: 
                df_list = [pd.read_csv(p) for p in pth.iterdir() if any(x in str(p) for x in data_filter)]
            else: 
                df_list = [pd.read_csv(p) for p in pth.iterdir()]

            # Round epochs to nearest second for consistency between UE files 
            concat_matching_dfs = {}
            for df in df_list:
                df.epoch = df.epoch.round()
                if str(df.columns) in concat_matching_dfs:
                    concat_matching_dfs[str(df.columns)].append(df)
                else: 
                    concat_matching_dfs[str(df.columns)] = [df]

            df_list = [pd.concat(frames, axis=0, ignore_index=True) for frames in concat_matching_dfs.values()]
            # Awful lambda function to merge a list of dataframes 
            df_output = reduce(lambda  left,right: pd.merge(left,right,on=['epoch'],how='outer'), df_list).fillna(0)
            df_output['imsi'] = imsi

        # Stack data for each UE 
        total_df = pd.concat([total_df,df_output], axis=0, ignore_index=True)
    
    # Add label and rename epoch to timestamp 
    total_df['label'] = 0
    total_df = total_df.rename(columns={"epoch": "timestamp"}).fillna(0)
    # Set the malicious UEs to label 1 
    if malicious:
        for imsi in mal_ue_list: 
            total_df.loc[total_df['imsi'] == str(imsi), 'label'] = 1
    # Save the file to the data dir 
    benign_or_malicious = ue_sample_dir.stem.split('-')[1]
    return total_df


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Process raw ves messages and UE data samples into csv datasets.')
    parser.add_argument('ves_files_list', metavar='Benign File', type=Path, nargs=1,
                    help='A json file containing all the data files and keys indicating if the data is malicious and what slice is the target')
    parser.add_argument('benign_ue_dir', metavar='Benign UE dir.', type=Path, nargs=1,
                    help='Directory where benign UE data is.')
    parser.add_argument('malicious_ue_dir', metavar='Malicious UE dir.', type=Path, nargs=1,
                    help='Directory where malicious UE data is.')
    parser.add_argument('save_file_core', metavar='Save file for core data.', type=Path, nargs=1,
                    help='Output save file.')
    parser.add_argument('save_file_slice', metavar='Save file for slice data.', type=Path, nargs=1,
                    help='Output save file.')
    parser.add_argument('save_file_ue', metavar='Save file for ue data.', type=Path, nargs=1,
                    help='Output save file.')
    args = parser.parse_args()

    # Inputs 
    core_data = Path(sys.argv[1])
    bengin_ue_dir = Path(sys.argv[2])
    malicious_ue_dir = Path(sys.argv[3])
    save_file_core = Path(sys.argv[4])
    save_file_slice = Path(sys.argv[5])
    save_file_ue = Path(sys.argv[6])

    # Core Dataframes 
    core_data = json.loads(core_data.read_text(encoding="UTF-8"))
    ves_files = list(core_data.keys())
    mal_ids = [list(targets.values())[0] for targets in core_data.values()]
    target_slice_sds = [list(targets.values())[1] for targets in core_data.values()]
    core_frames = [ves2csv(Path(path), mal_id) for path, mal_id in zip(ves_files, mal_ids)]
    # Combined core data 
    core_df = combine_csv(core_frames, save_file_core)

    # Slice Dataframe 
    slice_labels = [('1-111111','10.1.0.138:9090'), ('1-222222','10.1.0.201:9090'), ('2-333333','10.1.0.228:9090'), ('2-444444','10.1.0.33:9090'), ('3-555555','10.1.0.76:9090'), ('3-666666','10.1.0.232:9090')]
    slice_frames = [split_slices(core_frame, slice_labels, target_slice_sd) for core_frame, target_slice_sd in zip(core_frames, target_slice_sds)]
    slice_df = combine_csv(slice_frames, save_file_slice)

    # UE Dataframe 
    benign_data_filter = ['09-24']
    benign_ue_df = make_ue_df(bengin_ue_dir, benign_data_filter)
    malicious_data_filter = ['09-24']
    mal_ue_imsis = [999700000000001, 999700000000003, 999700000000011, 999700000000012, 999700000000013, 999700000000015, 999700000000016, 999700000000017, 999700000000019, 999700000000024, 999700000000025, 
    999700000000027, 999700000000032, 999700000000033, 999700000000035, 999700000000038, 999700000000041, 999700000000043, 999700000000045, 999700000000047, 999700000000050, 999700000000054, 999700000000055, 
    999700000000069, 999700000000070, 999700000000071, 999700000000079, 999700000000081, 999700000000093, 999700000000111, 999700000000117, 999700000000118, 999700000000122, 999700000000123, 999700000000126, 
    999700000000128, 999700000000131, 999700000000144, 999700000000146, 999700000000148, 999700000000158, 999700000000159, 999700000000170, 999700000000179]
    malicious_ue_df = make_ue_df(malicious_ue_dir, malicious_data_filter, mal_ue_list = mal_ue_imsis, malicious = True)

    full_ue_df = combine_csv([benign_ue_df, malicious_ue_df], saveFile=save_file_ue)