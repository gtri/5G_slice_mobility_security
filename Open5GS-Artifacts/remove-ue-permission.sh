#!/bin/bash

# Reacts to a DSM attack by stripping UE of its privledges on the network
# IMSI will come from the UEs that are predicted as part of the attack.

# Single operation 
IMSI=$1
POPULATE_POD=$2

kubectl exec $POPULATE_POD  -ti -- open5gs-dbctl remove $IMSI
