# Data
These csv files contain the processed data from the raw_data folder. The data is processes using `NRves2csv.py`. 

## Sumamry 
Slice-dataset is used for training the slice autoencoder and ue-dataset is used for training the UE autoencoder. 

5 30min scenarios - 1 benign and 4 malicious 
1 slice targed in each malicious scenario 
46 malicious UEs - Last 3 digits of IMSI [1, 3, 11, 12, 13, 15, 16, 17, 19, 24, 25, 27, 32, 33, 35, 38, 41, 43, 45, 47, 50, 54, 55, 69, 70, 71, 79, 81, 93, 111, 117, 118, 122, 123, 126, 128, 131, 144, 146, 148, 150, 158, 159, 170, 179, 180]

Slice-dataset:
~87% benign 
~13% malicious 

UE-dataset:
~80% benign 
~20% malicious 

There are different ratios of benign to malicious because there was only 1 target slice (1/6) in the slice data and 46 malicious UEs (1/4) in the UE data. 

## Core-Dataset
`core-dataset.csv` contains all of the VES events processed into separate columns by slice. This csv has all the same features as `slice-dataset.csv` in fewer rows but greater columns. 

## Slice-Dataset 
`slice-dataset.csv` is essentially a transformed version of core-dataset with a column added for the slice-id (sd) of each slice a data point corresponds to. In other words, each sample describes a slice rather than the state of the entire core. The slice-dataset is used to train the anomaly detection autoencoder. The slice-dataset is labeled by taking the target slice from a raw malicious json and labeling every sample with that slice id as malicious. This was done because the DSM attack has a distinct pattern across both the ramp-up and cool-down phase of attack. 

### Feature List 
Slice-Dataset:
- pdu_session_create_request_rate
- pdu_session_create_request_success_rate
- smf_sessions
- n4_establish
- upf_inbound_traffic
- upf_outbound_traffic
- registered_subscribers
- pcf_policy_association_requests
- pcf_policy_association_requests_success
- pcf_policy_association_number
- label
- slice_id (used for context and not in training)

## UE-dataset 
`ue-dataset.csv` is a combined version of all the raw UE data. 

### Feature List 
- slice1_count
- slice2_count
- slice3_count
- slice4_count
- slice5_count
- slice6_count
- dl_rx_count
- dl_retx_count
- dl_err_count
- ul_tx_count
- ul_retx_count
- ul
- dl
- imsi (used for context and not in training)
- label