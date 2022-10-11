#/bin/bash

source parameters.sh

## This script is to be run locally.

echo "Starting Workers."
for index in ${!address[*]}; do
    if [ "${address[$index]}" == "localhost" ]; then
        ./scripts/startWorker.sh ${ports[$index]} $conf &
    else
        ssh ${address[$index]} " cd ${remoteDir}; ./scripts/startWorker.sh ${ports[$index]} $conf" &
    fi
done

##  Start the monitoring front and back end.

./scripts/startMonitoring.sh

for index in ${!address[*]}; do
    curl \
        --header "Content-Type: application/json" \
        --data "{\"name\":\"Worker - ${ports[$index]}\",\"address\":\"${address[$index]}:${ports[$index]}\"}" \
        http://localhost:8080/workers > /dev/null
done


echo "A Monitoring tool is started at http://localhost:4200"

wait
