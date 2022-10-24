#/bin/bash

## This script is to be run on the federated site

source parameters.sh

mkdir -p tmp/worker
mkdir -p results/fed/workerlog/

if [[ -f "tmp/worker/$1" ]]; then
    echo "already running worker !! you forgot to stop the workers"
    echo "please manually stop workers and clear tmp folder in here"
    exit -1
fi

nohup \
    systemds WORKER $1 -stats 50 -config conf/$2.xml \
    > results/fed/workerlog/$HOSTNAME-$1.out 2>&1 &

echo Starting worker $HOSTNAME $1 $2

# Save process Id in file, to stop at a later time
echo $! > tmp/worker/$HOSTNAME-$1
