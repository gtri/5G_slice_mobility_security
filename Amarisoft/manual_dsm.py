'''
Amarisoft schedules sim events based on a "ue-list" that details every event
for each UE defined in a scenario. Scheduling every event by hand is impractical, but
the Amarisoft GUI can make some simple scenarios with relative ease. The Amarisoft GUI
does not, however, have enough control to create the DSM attack. This file modifies the ue-list
JSON to schedule DSM attack events on a set of IMSIs specified by the user. 

mal_ue_list: List of IMSIs that participate in the attack
target_slice: The APN (slice) under attack
tup 
'''

import json 
import random
import argparse

def make_dsm_sim_events(mal_ue_list, target_slice, tup, tdown, scenario_duration, amarisoft_file: str = 'final-config/ue-list.json'): 
    # tup: Time of the attack (required to cause an increase in resource use)
    # tdown: Time of normal traffic (allowing the network to reduce resources)

    # Automatically outputs the new file 
    with open(amarisoft_file, 'r') as f:
        loaded_file = json.loads(f.read())
    ue_list = []
    
    for ue in mal_ue_list:
        current_time = 0
        sim_events = []
        # sim_events.append({
        # "start_time": current_time,
        # "event": "power_on"
        # })
        # Now iterate over the scenarios total duration to make traffic events
        while current_time < scenario_duration: 
            ease_off_events = random_gen_traffic_event(ue, current_time, tdown)
            current_time = current_time + tdown
            dsm_events = make_swing_load_events(ue, target_slice, current_time, tup)
            current_time = current_time + tup
            for item in ease_off_events:
                sim_events.append(item)
            for item in dsm_events:
                sim_events.append(item)

        for loaded_ue in loaded_file:
            if loaded_ue['ue_id'] == ue:
                loaded_ue['sim_events'] = sim_events

        ue_list.append(sim_events)
    
    with open('final-config/dsm-attack-list-' + target_slice + '.json', 'w') as f:
        json.dump(loaded_file, f)
        
    return ue_list

def make_swing_load_events(ue_id, target_slice, start_time, event_duration):
    sim_events = [] 
    event_duration = event_duration - 1
    sim_events.append({
        "start_time": start_time,
        "event": "power_on"
        })

    dsm_connect = {
    "apn": target_slice,
    "pdn_type": "ipv4",
    "start_time": start_time,
    "event": "pdn_connect"
    }
    sim_events.append(dsm_connect)

    dsm_traffic = {
    "start_time": start_time,
    "end_time": start_time + event_duration - 1,
    "dst_addr": "10.10.4.115",
    "payload_len": 1000,
    "bit_rate": 100000,
    "type": "udp",
    "apn": target_slice,
    "event": "cbr_recv"
    }
    sim_events.append(dsm_traffic)
    dsm_traffic = {
    "start_time": start_time,
    "end_time": start_time + event_duration - 1,
    "dst_addr": "10.10.4.115",
    "payload_len": 1000,
    "bit_rate": 100000,
    "type": "udp",
    "apn": target_slice,
    "event": "cbr_send"
    }
    sim_events.append(dsm_traffic)

    # dsm_disconnect = {
    # "event": "pdn_disconnect",
    # "pdn_type": "ipv4",
    # "apn": target_slice,
    # "start_time": start_time + event_duration
    # }
    # sim_events.append(dsm_disconnect)

    sim_events.append({
    "start_time": start_time + event_duration,
    "event": "power_off"
    })

    return sim_events


def random_gen_traffic_event(mal_ue_id, start_time, event_duration):
    # 50/50 chance to start a PDN and initiate traffic on the two valid slices 
    sim_events = []
    event_duration = event_duration - 1
    sim_events.append({
        "start_time": start_time,
        "event": "power_on"
        })

    prob1 = random.random()
    prob2 = random.random()
    # Event consists of PDN connect, traffic event, pdn_disconnect 
        # Slices "internet" and "ims"
    if (mal_ue_id in range(0,61)):
        if prob1 > 0.5:
            pdn_connect1 = {
            "apn": "internet",
            "pdn_type": "ipv4",
            "start_time": start_time,
            "event": "pdn_connect"
            }
            sim_events.append(pdn_connect1)

            traffic1 = {
            "start_time": start_time,
            "end_time": start_time + event_duration - 1,
            "dst_addr": "8.8.8.8",
            "payload_len": 1000,
            "delay": 1,
            "apn": "internet",
            "event": "ping"
            }
            sim_events.append(traffic1)

            # pdn_disconnect1 = {
            # "event": "pdn_disconnect",
            # "apn": "internet",
            # "start_time": start_time + event_duration,
            # "pdn_type": "ipv4"
            # }
            # sim_events.append(pdn_disconnect1)

        if prob2 > 0.5:
            pdn_connect2 = {
            "apn": "ims",
            "pdn_type": "ipv4",
            "start_time": start_time,
            "event": "pdn_connect"
            }
            sim_events.append(pdn_connect2)

            traffic2 = {
            "start_time": start_time,
            "end_time": start_time + event_duration - 1,
            "dst_addr": "8.8.8.8",
            "payload_len": 1000,
            "delay": 1,
            "apn": "ims",
            "event": "ping"
            }
            sim_events.append(traffic2)

            # pdn_disconnect2 = {
            # "event": "pdn_disconnect",
            # "apn": "ims",
            # "start_time": start_time + event_duration,
            # "pdn_type": "ipv4"
            # }
            # sim_events.append(pdn_disconnect2)
            

    # Slices 3 and 4
    elif (mal_ue_id in range(61,121)):
        if prob1 > 0.5:
            pdn_connect1 = {
            "apn": "slice3",
            "pdn_type": "ipv4",
            "start_time": start_time,
            "event": "pdn_connect"
            }
            sim_events.append(pdn_connect1)

            traffic1 = {
            "start_time": start_time,
            "end_time": start_time + event_duration - 1,
            "url": "http:10.10.4.115:8080/data?size=10000",
            "max_delay": 1,
            "max_cnx": 1000,
            "apn": "slice3",
            "event": "http"
            }
            sim_events.append(traffic1)

            # pdn_disconnect1 = {
            # "event": "pdn_disconnect",
            # "apn": "slice3",
            # "start_time": start_time + event_duration,
            # "pdn_type": "ipv4"
            # }
            # sim_events.append(pdn_disconnect1)

        if prob2 > 0.5:
            pdn_connect2 = {
            "apn": "slice4",
            "pdn_type": "ipv4",
            "start_time": start_time,
            "event": "pdn_connect"
            }
            sim_events.append(pdn_connect2)

            traffic2 = {
            "start_time": start_time,
            "end_time": start_time + event_duration - 1,
            "url": "http:10.10.4.115:8080/data?size=10000",
            "max_delay": 1,
            "max_cnx": 1000,
            "apn": "slice4",
            "event": "http"
            }
            sim_events.append(traffic2)

            # pdn_disconnect2 = {
            # "event": "pdn_disconnect",
            # "apn": "slice4",
            # "start_time": start_time + event_duration,
            # "pdn_type": "ipv4"
            # }
            # sim_events.append(pdn_disconnect2)

    elif (mal_ue_id in range(121,181)):
        if prob1 > 0.5:
            pdn_connect1 = {
            "apn": "slice5",
            "pdn_type": "ipv4",
            "start_time": start_time,
            "event": "pdn_connect"
            }
            sim_events.append(pdn_connect1)

            traffic1 = {
            "start_time": start_time,
            "end_time": start_time + event_duration - 1,
            "dst_addr": "10.10.4.115",
            "payload_len": 1000,
            "bit_rate": 10000,
            "type": "udp",
            "apn": "slice5",
            "event": "cbr_recv"
            }
            sim_events.append(traffic1)

            # pdn_disconnect1 = {
            # "event": "pdn_disconnect",
            # "apn": "slice5",
            # "start_time": start_time + event_duration,
            # "pdn_type": "ipv4"
            # }
            # sim_events.append(pdn_disconnect1)

        if prob2 > 0.5:
            pdn_connect2 = {
            "apn": "slice6",
            "pdn_type": "ipv4",
            "start_time": start_time,
            "event": "pdn_connect"
            }
            sim_events.append(pdn_connect2)

            traffic2 = {
            "start_time": start_time,
            "end_time": start_time + event_duration - 1,
            "dst_addr": "10.10.4.115",
            "payload_len": 1000,
            "bit_rate": 100000,
            "type": "udp",
            "apn": "slice6",
            "event": "cbr_recv"
            }
            sim_events.append(traffic2)

            # pdn_disconnect2 = {
            # "event": "pdn_disconnect",
            # "apn": "slice6",
            # "start_time": start_time + event_duration,
            # "pdn_type": "ipv4"
            # }
            # sim_events.append(pdn_disconnect2)

    else:
        raise ValueError("Invalid UE id. Outside the range of registered UEs")

    sim_events.append({
    "start_time": start_time + event_duration,
    "event": "power_off"
    })

    return sim_events


mal_ue_ids = [1, 3, 11, 12, 13, 15, 16, 17, 19, 24, 25, 27, 32, 33, 35, 38, 41, 43, 45, 47, 50, 54, 55, 69, 70, 71, 79, 81, 93, 111, 117, 118, 122, 123, 126, 128, 131, 144, 146, 148, 150, 158, 159, 170, 179, 180]
target_slice = 'ims'
tup = 30
tdown = 80
scenario_duration = 1800

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert an Amarisoft generated ue-list into a dsm attack. Assumes the same core configuration as ours!')
    parser.add_argument('t_up', metavar='Attack uptime', type=Path, nargs=1,
                    help='The time malicious ues are active on a target slice in seconds')
    parser.add_argument('t_down', metavar='Attack downtime', type=Path, nargs=1,
                    help='The time it takes for the core to spin down newly deployed resources in seconds')
    parser.add_argument('scenario_duration', metavar='Scenario duration', type=Path, nargs=1,
                    help='The length of the scenario, in seconds, used to determine how many attack iterations to generate')
    parser.add_argument('target_slice', metavar='Target slice', type=Path, nargs=1,
                    help='The slice you want the DSM attack to target')
    args = parser.parse_args()

    tup = int(sys.argv[1])
    tdown = int(sys.argv[2])
    scenario_duration = int(sys.argv[3])
    target_slice = str(sys.argv[4])

    # Our list of malicious UE ids 
    mal_ue_ids = [1, 3, 11, 12, 13, 15, 16, 17, 19, 24, 25, 27, 32, 33, 35, 38, 41, 43, 45, 47, 50, 54, 55, 69, 70, 71, 79, 81, 93, 111, 117, 118, 122, 123, 126, 128, 131, 144, 146, 148, 150, 158, 159, 170, 179, 180]

    # This writes out the new file automatically
    all_ue_events = make_dsm_sim_events(mal_ue_ids, target_slice, tup, tdown, scenario_duration)
