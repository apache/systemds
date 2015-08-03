#!/bin/bash
if [ "$2" == "SPARK" ]; then CMD="./sparkDML "; DASH="-"; elif [ "$2" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi

BASE=$1/stratstats

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"


#XS data 10K rows
${CMD} -f ../datagen/genRandData4StratStats.dml $DASH-explain $DASH-stats $DASH-nvargs nr=10000 nf=100 D=${BASE}/A_10k/data Xcid=${BASE}/A_10k/Xcid Ycid=${BASE}/A_10k/Ycid A=${BASE}/A_10k/A fmt=csv

#S data 100K rows
#${CMD} -f ../datagen/genRandData4StratStats.dml $DASH-explain $DASH-stats $DASH-nvargs nr=100000 nf=100 D=${BASE}/A_100k/data Xcid=${BASE}/A_100k/Xcid Ycid=${BASE}/A_100k/Ycid A=${BASE}/A_100k/A fmt=csv

#M data 1M rows
#${CMD} -f ../datagen/genRandData4StratStats.dml $DASH-explain $DASH-stats $DASH-nvargs nr=1000000 nf=100 D=${BASE}/A_1M/data Xcid=${BASE}/A_1M/Xcid Ycid=${BASE}/A_1M/Ycid A=${BASE}/A_1M/A fmt=csv

#L data 10M rows
#${CMD} -f ../datagen/genRandData4StratStats.dml $DASH-explain $DASH-stats $DASH-nvargs nr=10000000 nf=100 D=${BASE}/A_10M/data Xcid=${BASE}/A_10M/Xcid Ycid=${BASE}/A_10M/Ycid A=${BASE}/A_10M/A fmt=csv
