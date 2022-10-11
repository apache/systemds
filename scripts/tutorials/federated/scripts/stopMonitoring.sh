#/bin/bash

if [[ -d "tmp/monitoring" ]]; then
    echo "Stopping Monitoring "
    if [[ -f "tmp/monitoring/monitoringProcessID" ]]; then
        pkill -P $(cat tmp/monitoring/monitoringProcessID)
        rm -f "tmp/monitoring/monitoringProcessID"
    fi
    if [[ -f "tmp/monitoring/UIProcessID" ]]; then
        echo "STOP NPM manually!! Process ID:"
        cat "tmp/monitoring/UIProcessID"
       
    fi
fi
