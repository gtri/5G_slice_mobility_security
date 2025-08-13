# Amarisoft Artifacts 
The artifacts in this folder are based on our configured UEs and slices. 

## Our Configuration 
We configured 180 UEs and 6 slices.

| Slice | SST | SD | Traffic Type |
| internet  | 0x01  | 0x111111  | ICMP  |
| ims  | 0x01     | 0x222222  | VOIP  |
| slice3  | 0x02  | 0x333333  | HTTP  |
| slice4  | 0x02  | 0x444444  | HTTP  |
| slice5  | 0x03  | 0x555555  | UDP  |
| slice6  | 0x03  | 0x666666  | UDP  |

IMSIs 999700000000001 - 999700000000180. 

The UEs were partitioned into 3 even groups of 60 and assigned one of the three SST. Each UE had PDU session permission on both slices with their assigned SST. 

## Amarisoft UE data collection 
To collect data from the Amarisoft callbox we used collectd for data scraping. The data scraper is a slight modification of the Prometheus linked Amarisoft data collector here: https://github.com/medianetlab/amarisoft-prometheus-exporter-collectd

## Artifacts 

### UE traffic scenarios 
These scenarios can be loaded into the Amarisoft UE SIM BOX to replicate our experiments. 

The benign scenario is a control where all the UEs perform "benign" traffic patterns, 

The 6 malicious scenarios were created by modifying the ue_list parameter in the benign scenario using the manual_dsm.py file. The manual_dsm.py file automatically creates the event sequences that UEs perform for the DSM attack. 
Each scenario indicates which slice is under attacked in the file name. 

### manual_dsm.py
This script takes in a UE list json file and modifies the 'sim events' fields to perform a DSM attack loop with the list of specified ue ids.

Parameters: 
mal_ue_ids: The Amarisoft ID of each UE that will participate in the DSM attack 
tup: The duration that malicious UEs will surge to a target slice for 
tdown: The backoff time where each malicious UE retreats to an alternate slice 
target_slice: The slice targeted by the DSM attack 
scenario_duration: The total duration to create sim events for 

### edit_ids.py
This was a helper function that modified the Amarisoft assigned ue id to match the last digits of our UE IMSIs. The Amarisoft GUI doesn't allow for the creation of a single scenario where different UEs behave differently. Instead, several scenarios were created and compiled into a single scenario so that UEs with different slice permisions would behave differently. The edit_ids file made sure that ue ids aligned with IMSI so that they were consitent with the 5G core list of UE permissions. 

