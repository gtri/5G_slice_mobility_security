# Raw Data 
This folder contains the raw data captured from the Open5GS core and Amarisoft callbox UEs/eNB.

## Summary 
Data collected from 5 scenarios, 1 benign 4 malicious. Each scenario was 30 minutes long with a 15s polling interval. Each malicious scenario targeted 1 slice. 

## Core Data
The core data folder contains 5 json files of core metrics, a summary file `ves_files_list.json`, and a helper file `extend_json.py`.

### Metrics Files 
Each metric file is data collected from a 30 minute scenario that can be found in the Amarisoft artifacts folder. Data was collected at a 15s polling interval across each 30 minute scenario. `ves_files_list.json` has details on which files contained malicious data and which slice sd was targeted by an attack. `extend_json.py` is  a helper function that combines a list of json files into a single file in case you want to merge all the malicious data. 

#### Metrics List  
- Number of UE RAN connections 
- Number of AMF sessions per slice
- Number of PDU creation requests per slice
- Number of successful PDU session creations per slice 
- Number of N4 interface session establishment requests (between SMF and UPF) per slice
- Number of successful N4 establishments per slice 
- Number of failed N4 establishments per slice 
- CPU resources requested per VNF 
- UE registration initiaions with the AMF
- Successful UE registrations with the AMF
- UPF uplink packet count per slice 
- UPF downlink packet count per slice 
- Number of SMF sessions per slice 
- Number of UEs active on the Open5GS core network 
- Container CPU use (CPU seconds)
- AMF registered UE subscribers per slice 
- Number of PCF association requests per slice 
- Number of PCF sessions per slice 

## Amarisoft UE data (Collected for each UE)
The UE data folder contains 4 subdirectories for UE data collected from different scenarios. `ue-benign` and `ue-malicious` contain training data while `ue-pdu-mitigation`and `ue-ban-mitigation` were used for validation of different attack prevention strategies. The same IMSIs were used for malicious UEs in every malicious scenario: `[1, 3, 11, 12, 13, 15, 16, 17, 19, 24, 25, 27, 32, 33, 35, 38, 41, 43, 45, 47, 50, 54, 55, 69, 70, 71, 79, 81, 93, 111, 117, 118, 122, 123, 126, 128, 131, 144, 146, 148, 150, 158, 159, 170, 179, 180]`. The number corresponds to the last three digits of the UEs IMSI. 

There of the 4 categorized directories has a folder for each UE used in our scenarios labeled by the UE IMSI. (ue-999700000000001 - ue-999700000000180). Each of these folders contains raw data in dated csv files. There are three kinds of csv files  `slices`, `stats`, and `throughput`. `Slices` indicates which slices a UE was connected to at any given time, `stats` has uplink and downlink details, and `throughput` contains the total bitrate on the uplink and downlink. 

#### Metrics List
- Number of PDU sessions with each slice 
- Downlink rx 
- Downlink retransmissions 
- Downlink errors
- Uplink tx
- Uplink retransmissions
- Uplink datarate
- Downlink datarate 