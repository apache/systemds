#!/bin/bash
if [ "$5" == "SPARK" ]; then CMD="./sparkDML "; DASH="-"; elif [ "$5" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi

BASE=$4/binomial

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

DFAM=2
if [ $3 -gt 2 ] 
then
   DFAM=3
fi

#for all intercept values
for i in 0 1 2
do
   #training
   tstart=$SECONDS
   ${CMD} -f ../algorithms/MultiLogReg.dml $DASH-explain $DASH-stats $DASH-nvargs icpt=$i reg=0.01 tol=0.0001 moi=3 mii=5 X=$1 Y=$2 B=${BASE}/b
   ttrain=$(($SECONDS - $tstart - 3))
   echo "MultiLogReg train ict="$i" on "$1": "$ttrain >> times.txt
   
   #predict
   tstart=$SECONDS   
   ${CMD} -f ../algorithms/GLM-predict.dml $DASH-explain $DASH-stats $DASH-nvargs dfam=$DFAM vpow=-1 link=2 lpow=-1 fmt=csv X=$1_test B=${BASE}/b Y=$2_test M=${BASE}/m O=${BASE}/out.csv
   tpredict=$(($SECONDS - $tstart - 3))
   echo "MultiLogReg predict ict="$i" on "$1": "$tpredict >> times.txt
done