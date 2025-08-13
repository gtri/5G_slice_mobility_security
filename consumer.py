'''
Consumer loop for VES measurment collection.

Connects to ONAP message router and pulls VES events. 
The message router endpoint may need to be adjusted for other deployments. Our 
message router pod in ONAP was exposed via NodePort and used HTTP for querying. 

This consumer should be used concurrently with the scenario you want to collect 
data for. Technically, you could run the scenario then query the message-router 
repeatedly until you get all the data out, but it is unclear how much data 
the ONAP message-router stores at a time. 
'''
import time
from pathlib import Path
import requests 
import re 
import json
import argparse 

def write_json(new_data, filename='data.json'):
    # Check for empty data 
    if len(new_data) == 0:
        return 
    # Check for the savefile
    file = Path(filename)
    if file.is_file() == False:
        Path.touch(file)
    with open(filename,'r+') as file:
        # Load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data 
        file_data = file_data + [json.loads(item) for item in new_data]
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)


def ves_consumer_main(onap_ip: str = None, 
        message_router_port: str = None,
        scenario_duration: int = 60, 
        polling_interval: int = 15,
        save_file: str = 'ves-data.json'
        ): 

    # Start timer
    start_time = time.time()
    current_time = time.time()

    # Components of the message router endpoint - May need to be adjusted for your deployment 
    message_router = 'http://' + onap_ip + ':' + message_router_port + '/events/'
    ves_measurment_endpoint = 'unauthenticated.VES_MEASUREMENT_OUTPUT'
    consumer_info = '/my_consumer_group/1/'

    # Full endpoint 
    collection_endpoint = message_router + ves_measurment_endpoint + consumer_info

    while (time.time() - start_time) < scenario_duration:
        if time.time() - current_time > polling_interval:
            print("Time: ", current_time)
            current_time = time.time()

            # Message router query  
            ves_msg = requests.get(collection_endpoint)

            # Write query to saved data
            write_json(ves_msg.json(), save_file)

    return 


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Collect VES events from the ONAP message router.')
    parser.add_argument('onap_ip', metavar='ONAP IP Addr', type=str, nargs=1,
                    help='Assuming ONAP is deployed on a dedicated VM this will be the VM IP address.')
    parser.add_argument('message_router_port', metavar='ONAP message router Node Port', type=str, nargs=1,
                    help='If the ONAP message router uses HTTP and is externally exposed via NodePort use the NodePort IP.')
    parser.add_argument('scenario_duration', metavar='Scenario length', type=Path, nargs=1,
                    help='Length of the scenario you want to capture data during. Run this consumer concurrently with the scenario.')
    parser.add_argument('polling_interval', metavar='Sample rate', type=Path, nargs=1,
                    help='How frequently, in seconds, the consumer will query the message router for new VES events.')
    parser.add_argument('save_file', metavar='Save file', type=Path, nargs=1,
                    help='The JSON file to save VES events to.')
    args = parser.parse_args()

    # Inputs 
    onap_ip = str(sys.argv[1])
    message_router_port = str(sys.argv[2])
    scenario_duration = int(sys.argv[3])
    polling_interval = int(sys.argv[4])
    save_file = str(sys.argv[5])
    ves_consumer_main(onap_ip, message_router_port, scenario_duration, polling_interval, save_file)