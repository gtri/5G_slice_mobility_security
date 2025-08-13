#!/bin/bash
# Resets the database based on the configuration we used. 

# This loops over a set of commands and is very slow... 
# If you are a more savy user of MongoDB there should be a way to do this faster

POPULATE_POD=$1

source ue-deployment.sh 999700000000001 60 1 $POPULATE_POD
source ue-deployment.sh 999700000000061 60 2 $POPULATE_POD
source ue-deployment.sh 999700000000121 60 3 $POPULATE_POD
source mal-ue.sh $POPULATE_POD