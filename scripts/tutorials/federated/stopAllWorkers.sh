#!/bin/bash

source parameters.sh

## This script is to be run locally.

## Close workers
for index in ${!address[*]}; do
    if [ "${address[$index]}" != "localhost" ]; then
        # remote workers stop.
        ssh ${address[$index]} " cd ${remoteDir}; ./scripts/stopWorker.sh" &
    else
        # stop all localhost workers.
        ./scripts/stopWorker.sh
    fi
done

./scripts/stopMonitoring.sh 

wait
