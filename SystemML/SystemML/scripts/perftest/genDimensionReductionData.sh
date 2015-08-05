#!/bin/bash

if [ "$1" == "" -o "$2" == "" ]; then echo "Usage: $0 <hdfsDataDir> <MR | SPARK | ECHO>   e.g. $0 perftest SPARK" ; exit 1 ; fi
if [ "$2" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$2" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi


BASE=$1/dimensionreduction
FORMAT="csv" 

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"


#generate XS scenarios (80MB)
${CMD} -f ../datagen/genRandData4PCA.dml $DASH-nvargs 5000 2000 $BASE/pcaData5k_2k_dense $FORMAT

#generate S scenarios (800MB)
#${CMD} -f ../datagen/genRandData4PCA.dml $DASH-nvargs 50000 2000 $BASE/pcaData50k_2k_dense $FORMAT

#generate M scenarios (8GB)
#${CMD} -f ../datagen/genRandData4PCA.dml $DASH-nvargs 500000 2000 $BASE/pcaData500k_2k_dense $FORMAT

#generate L scenarios (80GB)
#${CMD} -f ../datagen/genRandData4PCA.dml $DASH-nvargs 5000000 2000 $BASE/pcaData5M_2k_dense $FORMAT

#generate XL scenarios (800GB)
#${CMD} -f ../datagen/genRandData4PCA.dml $DASH-nvargs 50000000 2000 $BASE/pcaData50M_2k_dense $FORMAT

