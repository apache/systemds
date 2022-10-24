#/bin/bash

source parameters.sh

# Synchronize code and setup.
for index in ${!address[*]}; do
    # echo "${address[$index]}"
    if [  "${address[$index]}" != "localhost" ]; then
        # Make the required directiories
        ssh -A ${address[$index]} "mkdir -p ${remoteDir}; cd ${remoteDir}; mkdir -p results; mkdir -p data" &
        # Syncronize the configuration and scripts(start and stop worker).
        rsync -avhq -e ssh conf scripts parameters.sh ${address[$index]}:$remoteDir &
        ## Get Results
        rsync -avhq -e ssh ${address[$index]}:$remoteDir/results . &
    fi
done
wait
