#/bin/bash

source parameters.sh

mkdir -p tmp/monitoring

nohup \
    systemds FEDMONITORING 8080 \
    > tmp/monitoring/log.out 2>&1 &

echo $! > tmp/monitoring/monitoringProcessID
echo "Starting monitoring"

here=$(pwd)

echo "$SYSTEMDS_ROOT"

cd "$SYSTEMDS_ROOT/scripts/monitoring"
nohup \
   npm start \
   > $here/tmp/monitoring/UILog.out 2>&1 &
cd $here
echo $! > "tmp/monitoring/UIProcessID"

echo "Starting UI"

sleep 10


