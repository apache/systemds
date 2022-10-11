#/bin/bash
source parameters.sh

for index in ${!address[*]}; do
    echo ${ports[$index]}: ${address[$index]}:${ports[$index]}
    ssh -f -N -L ${ports[$index]}:${address[$index]}:${ports[$index]} ${address[$index]} &
done

wait
