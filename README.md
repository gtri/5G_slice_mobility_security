# 5G Slice Security  
## Project Scope and Purpose 
5G networks offer new features and improvements over previous generations of wireless network technology. Software definied networking and the use of virtual network functions are one such feature that enable the concept of "slicing" across 5G networks. Network slicing is an organizational concept used to segment portions of shared physical resources into separate virtual networks. A network slice is a functional end-to-end network that provides all of the network functions required to service user equipment (UE). The improvements in technology offered by 5G networks also introduce new attack vectors that can be exploited to harm the network. One such attack is the Distributed Slice Mobility (DSM) attack [1]. The DSM attack leverages network slicing by using a set of malicious UEs to surge onto a target slice and ease off of it in waves in a DoS style attack. This kind of attack can be difficult to detect because it leverages a legitimate 5G mechanic known as inter-slice mobility to move UEs between slices. 

This work proposes and implements an AI model to detect a DSM attack using metrics collected from an open-source software defined network. Additionally, this work was designed for compatibility with ONAP an open-source orchestration platform for 5G networks.  

This repository contains code for the following:
1. Open5GS configuration 
2. VESPA configuration 
3. Data processing pipelines for dataset creation and AI model training 
4. A 1D-CNN based Autoencoder for anomaly detection 


## Prerequisites
1. An Kubernetes based deployment of Open5GS with Prometheus enabled
    - Open5GS installation: https://github.com/Gradiant/5g-charts
    - Enable Prometheus for metrics: https://open5gs.org/open5gs/docs/tutorial/04-metrics-prometheus/
    - UPFs must be configured to have counters enabled for metric collection:  
        - We used 6 UPFs each hosted on a separate VM 
2. A deployment of ONAP 13.0.0 (Montreal) https://www.onap.org/ 
    - Must include the VES-collector, DMaaP and message-router
3. A eNB of choice
    - We used the Amarisoft Callbox 
4. A UE simulator/emulator of choice
    - We used the Amarisoft UE SIM BOX 
    - See also software  defined simulators such as UERANSIM: https://github.com/aligungr/UERANSIM 
5. VESPA for converting 5G core metrics collected in Prometheus to ONAP VES-events 
    - VESPA could be replaced by another data pipeline, but would require user defined VES events and data delivery to the ONAP VES-collector 
    - https://github.com/nokia/ONAP-VESPA 
6. A metric collection pipeline for UE data 
    - We used the Amarisoft callbox for our eNB and UE emulation which provides UE metrics through a remote API
    - Our metric collection was based on: https://github.com/medianetlab/amarisoft-prometheus-exporter-collectd  

Setting up all of these prerequisites is a non-trivial task! Getting all of the components to properly communicate requires a lot of effort and troubleshooting. 
Once all of the prerequisites are set up you will have a full 5G network testbed supervised by ONAP. 

## Data Collection 
Our core metrics were collected from an Open5GS core and our UE metrics were collected from an Amarisoft UE callbox. A different UE simulator such as UERANSIM could be used instead but will require a different data collection/preprocessing strategy. 

- Open5GS-Artifacts 
    - Contains Open5GS core configuration information 
    - Contains the VESPA configuration we used for making VES events from Prometheus data 
    - consumer.py - A data collection method for pulling ves evnets from the ONAP message-router 
- Amarisoft 
    - Contains the UE scenarios we used 
    - Helper functions for modifying Amarisoft generated scenarios 
    - The data collection modules used to scrape Amarisoft UE data 

## Data Preprocessing Pipeline + Model Training 
A step by step walkthrough is provided in the Instructions.ipynb 

- NRves2csv.py:
    - Transforms ONAP VES events captured from the ONAP message-router into csv files 
    - Compiles UE data scraped from the Amarisoft callbox into a single file 
- NRTimeSeriesML.py 
    - Functions for transforming the 5G core and UE data into time-series pytorch datasets 
    - A 1DCNN based autoencoder 
    - A PyTorch Lightning module 
        - Includes train + test loop 
        - A reconstruction loss class variable for tracking and comparing benign vs. malicious reconstruction loss 

# Contacts
Yatis Dodia yatis.dodia@gtri.gatech.edu

Jeff Pitcher jeff.pitcher@gtri.gatech.edu

Maxwell Yarter maxwell.yarter@gtri.gatech.edu

# References
[1] V. N. Sathi and C. S. R. Murthy, “Distributed Slice Mobility Attack: A Novel Targeted Attack
Against Network Slices of 5G Networks,” IEEE Networking Letters, vol. 3, no. 1, pp. 5–9,
Mar. 2021, doi: 10.1109/LNET.2020.3044642.
