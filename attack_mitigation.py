'''
Creates subprocesses to ban UEs that are identified as malicious. 
'''
import subprocess

# TODO: Make this communicate out to the core from the ONAP microservice 

def ban_ue(imsi):
    subprocess.Popen(['bash', 
                        'ue-management/remove-ue-permission.sh',
                        imsi])

def remove_malicious_ues(ue_predictions: dict):
    # Iterates over {imsi: T or F} to remove UEs predicted as malicious
    target_imsis = [imsi for imsi, pred in ue_predictions.items() if pred]
    for imsi in target_imsis:
        ban_ue(imsi)
    
                        