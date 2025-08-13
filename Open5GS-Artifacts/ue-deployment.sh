#!/bin/bash

# Deploy N UE's incremented from a starting IMSI (probably wont work if the first number is 0)
# Arg 1 is the starting IMSI EX: 999700000000001 
# Arg 2 is the number to deploy starting from arg 1 

# Set the starting UE and number to deploy
START_IMSI=$1
n_ue=$2
sst=$3
POPULATE_POD=$4

CURRENT_IMSI=$START_IMSI
FINAL_IMSI=$((START_IMSI + n_ue))

if [ $sst == 1 ]; then
    dnn1='internet'
    dnn2='ims'
    sd1=111111
    sd2=222222
elif [ $sst == 2 ]; then
    dnn1='slice3'
    dnn2='slice4'
    sd1=333333
    sd2=444444
elif [ $sst == 3 ]; then 
    dnn1='slice5'
    dnn2='slice6'
    sd1='555555'
    sd2='666666'
else 
    echo 'Error: Valid sst are currently 1, 2, or 3'
    exit 1
fi
echo $START_IMSI $n_ue $sst $dnn1 $dnn2 $sd1 $sd2

# Loop over the number of UE's and add them to both slices 
until [ $CURRENT_IMSI -ge $FINAL_IMSI ]; do
    echo $CURRENT_IMSI
    kubectl exec $POPULATE_POD  -ti -- open5gs-dbctl add_ue_with_slice $CURRENT_IMSI 465B5CE8B199B49FAA5F0A2EE238A6BC E8ED289DEBA952E4283B54E88E6183CA $dnn1 $sst $sd1
    kubectl exec $POPULATE_POD  -ti -- open5gs-dbctl update_slice $CURRENT_IMSI $dnn2 $sst $sd2
    ((CURRENT_IMSI += 1))
done
