'''
A derivative of consumer.py and ves2csv used for live data capture and 
model inference. 
'''

import time
import warnings
from pathlib import Path
import requests 
import json
import itertools
import pandas as pd
import pickle

from torch import nn 
from datetime import datetime
import pytz
from websocket import create_connection

from NRTimeSeriesML import *
from UE_feature_collection_loop import *
from attack_mitigation import *

# from UEDataClient import AI_client

def standardize_timecodes(timestamp: str):
    # Parse the month, day, year, time-of-day into a datetime object then UTC timestamp 
    time_list = timestamp.split()
    month = int(time_list[1])
    day = int(time_list[2])
    year = int(time_list[3])
    tod = time_list[4]
    hour = int(time_list[4].split(':')[0]) + 12 # This is an unfortunate assumption for measurments taken after noon... 
    minute = int(time_list[4].split(':')[1])
    second = int(time_list[4].split(':')[2])

    time = datetime(year, month, day, hour, minute, second, tzinfo=pytz.timezone('utc'))
    timestamp = time.timestamp()
    return timestamp


def process_ves_data(data, collected_slice_samples, feature_labels):
    # Data: New batch of ves mesasurments 
    # Collected_samples: dictionary of previously collected samples 
    # Feature_dict: Actual features that should be in a sample

    # Loop through collected events 
    for event in data:
        event = json.loads(event)
        # Process the timecode into a UTC timestamp
        collector_timestamp = event['event']['commonEventHeader']['internalHeaderFields']['collectorTimeStamp']
        standardized_timestamp = standardize_timecodes(collector_timestamp)
        # Add the timestamp if it hasn't been seen before 
        if standardized_timestamp in collected_slice_samples.keys(): 
            pass
        else: 
            collected_slice_samples[standardized_timestamp] = {key:[] for key in feature_labels}
        
        for objectInstance in event['event']['measurementsForVfScalingFields']['additionalObjects']:
            # List of all top level metrics 
            for object_ in objectInstance['objectInstances']:
                feature_label = objectInstance['objectName'] + ' ' + object_['objectKeys'][0]['keyValue']
                collected_slice_samples[standardized_timestamp][feature_label] += object_['objectInstance'].values()

    # Sort the samples by timestamp in the event that they arrive out of order 
    collected_slice_samples = dict(sorted(collected_slice_samples.items()))

    return collected_slice_samples


def split_samples_per_slice(collected_slice_samples, slice_samples):
    # Splits the raw samples for the entire core into 1 sample per slice 
    # slice_samples: List of dictionaries [{'1-111111': {timestamp: {features}}}, {'1-222222': {timestamp: {features}}}, ... ]
    slice_labels = [('1-111111','10.1.0.138:9090'), 
                    ('1-222222','10.1.0.201:9090'), 
                    ('2-333333','10.1.0.228:9090'), 
                    ('2-444444','10.1.0.33:9090'), 
                    ('3-555555','10.1.0.76:9090'), 
                    ('3-666666','10.1.0.232:9090')]
    
    for timestamp, features in collected_slice_samples.items():
        for slice_id in slice_labels:
            sub_dict = {feature:value for feature, value in features.items() if any(id_ in feature for id_ in slice_id)} #| (not any(x in feature for x in list(sum(slice_labels, ())) ))} 
            slice_samples[slice_id[0]][timestamp] = sub_dict

    return slice_samples


def stack_slice_data(slice_samples, frame_size): 
    start_timestamp = list(slice_samples['1-111111'].keys())[0]
    stacked_features = {}
    for slice_ in slice_samples.keys():
        stacked_features[slice_] = ([list(itertools.chain(*list(feature.values()))) for feature in list(slice_samples[slice_].values())[0:frame_size]])
    return start_timestamp, stacked_features


def fill_voids(feature_labels, collected_slice_samples, frame_size):
    # Sometimes the VES collector doesn't get data for a couple of features. 
    # In this event fill the feature with a 0 
    for i in range(frame_size):
        for key in feature_labels:
            if len(list(collected_slice_samples.values())[i][key]) == 0:
                list(collected_slice_samples.values())[i][key] = [0]
                

# TODO: Add the threshold params 
# Gathers all of the parameters needed to function 
def prep_artifacts(transform_path: Path, state_dict_path: Path, threshold_metrics_path: Path):
    # Load the fit data transformation 
    transform = pickle.load(open(transform_path, 'rb'))

    # Load the model state dict and instantiate the model based on the saved weight dimensions 
    state_dict = torch.load(state_dict_path, weights_only=True)
    dims = [item.size() for item in state_dict.values()]
    frame_size = dims[-1][0]
    filter_sizes = [dims[1][0], dims[3][0]]
    model = ConvAutoencoder(n_features=frame_size, filter_sizes=filter_sizes)
    model.load_state_dict(state_dict)

    # Instantiate the loss function 
    loss_fn = nn.MSELoss(reduction='mean')

    # Load the threshold detection metrics {mean:, std: }
    threshold_metrics = pickle.load(open(threshold_metrics_path, 'rb'))

    return transform, model, loss_fn, frame_size, threshold_metrics


def make_inference(transform, model, stacked_features, loss_fn):
    keys = [key for key in list(stacked_features.keys())]
    samples = [torch.tensor(transform.transform(sample)).to(torch.float32) for sample in list(stacked_features.values())]
    # print('Samples going into the model: ', samples)
    inferences = [model(sample) for sample in samples]
    losses = {key:loss_fn(inference, sample) for key, inference, sample in zip(keys, inferences, samples)}
    return losses


def make_pred(mean, num_stds, losses):
    # Returns True for malicious and False for benign
    # Keys are slice id or imsi
    predictions = {key: loss >= mean + num_stds for key, loss in losses.items()}
    return predictions


def get_candidate_ue(slice_predictions, ue_data_frame):
    # If no slices predicted exit  
    # if not any(slice_predictions.values()):
    #     return {}

    # Determine which slices are malicious
    malicious_slices = [slice_ for slice_, pred in slice_predictions.items() if pred]
    # Covert snssai into an idex to check against the UE features 
    slice_labels = ['1-111111','1-222222', 
                '2-333333','2-444444', 
                '3-555555','3-666666',]
    slice_indicies = [slice_labels.index(slice_) for slice_ in malicious_slices]
    # Call the client socket to get data from the UESIM box server socket 
    # ue_data = client.run_client(HOST, PORT, 1)
    # print(ue_data_frame)
    # timestamp = list(ue_data.keys())[0]
    # Check UEs to see if they are active on the target slice 
    candidates = {}
    for imsi, features in ue_data_frame.items():
        # Only check if the slice is active on the first index 
        if any((feature > 0 and idx in slice_indicies) for idx, feature in enumerate(features[0])):
            candidates[imsi] = features  
    return candidates


def flush_mr(collection_endpoint):
    # To make sure the UE and core data is synchronized the message router has to 
    # be emptied before the application starts to make sure they don't get out of sync
    for i in range(10): 
        requests.get(collection_endpoint).json()
        i += 1
    print('Message router emptied!')



def main(): 
    # All variable definitions 
    ####################################
    # Components of the message router endpoint
    message_router = 'http://10.0.0.28:30226/events/'
    ves_measurment_endpoint = 'unauthenticated.VES_MEASUREMENT_OUTPUT'
    consumer_info = '/my_consumer_group/1/'
    # Full endpoint 
    collection_endpoint = message_router + ves_measurment_endpoint + consumer_info

    flush_mr(collection_endpoint)

    # # UE Data collection client socket + HOST information 
    # HOST = '10.10.4.188'
    # PORT = 1234
    # ue_data_collection_client = AI_client()

    # Features that are gathered by the core - based off saved data not the live state of the ves-agent 
    feature_labels = pd.read_csv('sample_core_data/core-dataset.csv')
    feature_labels = list(feature_labels.columns)[1:]
    # print(feature_labels) # {timestamp_1: {}, timestamp_2: {}, ... }

    # Dictionaries that will expand to hold data
    collected_slice_samples = {}
    feature_dict = {key:[] for key in feature_labels}
    slice_labels = [('1-111111','10.1.0.138:9090'), 
                ('1-222222','10.1.0.201:9090'), 
                ('2-333333','10.1.0.228:9090'), 
                ('2-444444','10.1.0.33:9090'), 
                ('3-555555','10.1.0.76:9090'), 
                ('3-666666','10.1.0.232:9090')]
    slice_samples = {slice_id[0]:{} for slice_id in slice_labels}

    # UE variables 
    # Global var def (sort of) for enb ip
    enb_ip = '10.10.4.188'
    # Make an initial connection to establish the list of IMSIs 
    ws = None
    try:
        ws = create_connection('ws://%s:9002' % enb_ip)
        ws.recv()
        ws.send('{"message":"config_get"}')
        result =  ws.recv()
        j_res = json.loads(result)
        ran_id = get_ran_id(j_res)
        lte_cells = get_lte_cells(j_res)
        nr_cells = get_nr_cells(j_res)
        ws.send('{"message":"stats"}')
        result =  ws.recv()
        j_res = json.loads(result)

        ws.send('{"message":"ue_get" ,"stats":true}')
        result =  ws.recv()
        j_res = json.loads(result)
        imsi_list = imsi_id(j_res)
    except Exception as e:
        print(e)
        print('enb @ %s is not connected !' % enb_ip)
        if ws:
            ws.shutdown
    ws.shutdown

    # Empty dict for running UE samples 
    ue_sample_stack = {list(imsi.values())[0]:{} for imsi in imsi_list}
    # Empty dict for tracking UEs identified as malicious (don't attempt to ban an already removed UE)
    banned_ues = []


    # Artifacts for model inference and loss calculation 
    slice_transform_path = Path('artifacts/slice_transform.pkl')
    slice_state_dict_path = Path('artifacts/slice_state_dict')
    slice_threshold_metrics_path = Path('artifacts/slice_threshold_metrics.pkl')
    ue_transform_path = Path('artifacts/ue_transform.pkl')
    ue_state_dict_path = Path('artifacts/ue_state_dict')
    ue_threshold_metrics_path = Path('artifacts/ue_threshold_metrics.pkl')
    slice_transform, slice_model, loss_fn, slice_frame_size, slice_threshold_metrics = prep_artifacts(slice_transform_path, slice_state_dict_path, slice_threshold_metrics_path) 
    ue_transform, ue_model, loss_fn, ue_frame_size, ue_threshold_metrics = prep_artifacts(ue_transform_path, ue_state_dict_path, ue_threshold_metrics_path) 

    slice_mean_reconstruction_loss = slice_threshold_metrics['mean']
    slice_std_reconstruction_loss = slice_threshold_metrics['std']
    ue_mean_reconstruction_loss = ue_threshold_metrics['mean']
    ue_std_reconstruction_loss = ue_threshold_metrics['std']
    ####################################

    # Start timer - Not needed in micro-service
    start_time = time.time()
    current_time = time.time()
    scenario_duration = 1800
    measurment_window = 5

    # Data collection + inference loop 
    while (time.time() - start_time) < scenario_duration:
        if time.time() - current_time > measurment_window:
            print("Time: ", current_time)
            current_time = time.time()

            # Message router query  
            ves_msg = requests.get(collection_endpoint).json()
            # UE enb querry 
            ue_sample_stack = read(enb_ip, ue_sample_stack)

            # Add a failure condition to flush the sample stacks if a UE sample is missed 
            # Under healthy conditions this should only happen when Amarisoft is reconfigured to run a new scenario 
            if not list(ue_sample_stack.values())[0]:
                ue_sample_stack = {list(imsi.values())[0]:{} for imsi in imsi_list}
                collected_slice_samples = {}
                warnings.warn("WARNING: Missed UE sample! Flushing the sample buffer. Check the Amarisoft components if this message persists.")
                continue


            # Process the raw slice data into a dictionary of key value pairs 
            collected_slice_samples = process_ves_data(ves_msg, collected_slice_samples, feature_labels)
            # slice_samples = split_samples_per_slice(collected_slice_samples, slice_samples)
            # print(slice_samples)
            print('Num samples in slice buffer: ', len(collected_slice_samples))
            print('Num samples in ue buffer: ', len(list(ue_sample_stack.values())[0]))

            # Give a 10s buffer (2 samples) between the stack filling up and inference to get any data that came in out of order 
            if len(collected_slice_samples.keys()) > slice_frame_size + 2:
                # Process the slice data into frames
                fill_voids(feature_labels, collected_slice_samples, slice_frame_size)
                # Split the full data block into features per slice and stack them into a frame of frame size
                slice_samples = split_samples_per_slice(collected_slice_samples, slice_samples)
                start_timestamp, slice_stacked_features = stack_slice_data(slice_samples, slice_frame_size)
                with open('live_slice_samples.json', 'w') as fp:
                    json.dump(slice_stacked_features, fp)
                with open('live_ue_sample.json', 'w') as fp:
                    json.dump(ue_sample_stack, fp)

                # Process the UE data into frames 
                ue_start_timestamp, ue_stacked_features = stack_ue_data(ue_sample_stack, ue_frame_size)

                # Pass the samples to the AI model to predict loss and compare to the learned reconstruction loss threshold
                slice_losses = make_inference(slice_transform, slice_model, slice_stacked_features, loss_fn)
                print('Slice losses: ', slice_losses)
                slice_predictions = make_pred(slice_mean_reconstruction_loss, 3*slice_std_reconstruction_loss, slice_losses)
                print('Slice predictions: ', slice_predictions)

                # Identify candidate malicious UEs based on the slice predictions 
                # candidate_ues = get_candidate_ue(slice_predictions, ue_data_collection_client)
                candidate_ues = get_candidate_ue(slice_predictions, ue_stacked_features)
                print('Num UE candidates: ', len(candidate_ues))
                
                if len(candidate_ues) > 0:
                    # Pass UE samples to the UE model and compare to learned reconstruction loss threshold 
                    ue_losses = make_inference(ue_transform, ue_model, candidate_ues, loss_fn)
                    ue_predictions = make_pred(ue_mean_reconstruction_loss, 12*ue_std_reconstruction_loss, ue_losses)
                    print('Predicted UEs: ', ue_predictions)

                    # Ignore UEs that have already been removed from the network
                    for imsi in banned_ues:
                        ue_predictions.pop(imsi, None)
                    print('All banned UEs: ', banned_ues)

                    # # Ban any malicious UE from the network with the dbctl 
                    # malicious_imsi = remove_malicious_ues(ue_predictions)
                    # for imsi in malicious_imsi:
                    #     banned_ues.append(imsi)
                

                # Pop the first sample in each buffer to make room for new samples! 
                collected_slice_samples.pop(next(iter(collected_slice_samples)))
                for slice_ in slice_samples.keys():
                    slice_samples[slice_].pop(next(iter(slice_samples[slice_])))
                for imsi in ue_sample_stack.keys():
                    ue_sample_stack[imsi].pop(next(iter(ue_sample_stack[imsi])))

            # In case the UE data and core data become desynched (only by missing core samples) pop the extra UE samples
            if len(list(ue_sample_stack.values())[0]) > slice_frame_size + 2:
                for imsi in ue_sample_stack.keys():
                    ue_sample_stack[imsi].pop(next(iter(ue_sample_stack[imsi])))
            

    return 


if __name__ == '__main__': 
    main()