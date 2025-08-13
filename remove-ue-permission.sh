#!/bin/bash

# Reacts to a DSM attack by stripping UE of its privledges on the network
# IMSI will come from the UEs that are predicted as part of the attack.

# Single operation 
IMSI=$1

kubectl exec open5gs-populate-6f9f44ddf8-wlvxl  -ti -- open5gs-dbctl remove $IMSI

# # Loop over static array as initial test 
# declare -a mal_ue=(999700000000001 999700000000003 999700000000011 999700000000012 999700000000013 999700000000015 999700000000016 999700000000017 999700000000019 999700000000024 999700000000025 999700000000027 999700000000030 999700000000032 999700000000033 999700000000035 999700000000038 999700000000041 999700000000043 999700000000045 999700000000047 999700000000050 999700000000054 999700000000055 999700000000061 999700000000069 999700000000070 999700000000071 999700000000079 999700000000081 999700000000093 999700000000111 999700000000117 999700000000118 999700000000122 999700000000123 999700000000126 999700000000128 999700000000131 999700000000144 999700000000146 999700000000148 999700000000150 999700000000158 999700000000159 999700000000170 999700000000179 999700000000180)

# for i in "${mal_ue[@]}"
# do
#     kubectl exec open5gs-populate-6f9f44ddf8-wlvxl  -ti -- open5gs-dbctl remove $i
# done 

# Loop to fix the static loop
# for i in "${mal_ue[@]}"
# do
    # kubectl exec open5gs-populate-6f9f44ddf8-wlvxl  -ti -- open5gs-dbctl add_ue_with_slice $i 465B5CE8B199B49FAA5F0A2EE238A6BC E8ED289DEBA952E4283B54E88E6183CA 'internet' 1 111111 
    # kubectl exec open5gs-populate-6f9f44ddf8-wlvxl  -ti -- open5gs-dbctl update_slice $i 'ims' 1 222222 
    # kubectl exec open5gs-populate-6f9f44ddf8-wlvxl  -ti -- open5gs-dbctl update_slice $i 'slice3' 2 333333 
    # kubectl exec open5gs-populate-6f9f44ddf8-wlvxl  -ti -- open5gs-dbctl update_slice $i 'slice4' 2 444444 
    # kubectl exec open5gs-populate-6f9f44ddf8-wlvxl  -ti -- open5gs-dbctl update_slice $i 'slice5' 3 555555 
    # kubectl exec open5gs-populate-6f9f44ddf8-wlvxl  -ti -- open5gs-dbctl update_slice $i 'slice6' 3 666666 
# done 