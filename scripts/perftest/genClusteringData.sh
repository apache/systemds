#!/bin/bash
if [ "$2" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$2" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi

BASE=$1/clustering

FORMAT="binary" 
DENSE_SP=0.9
SPARSE_SP=0.01

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

#generate XS scenarios (80MB)
${CMD} -f ../datagen/genRandData4Kmeans.dml $DASH-nvargs nr=10000 nf=1000 nc=50 dc=10.0 dr=1.0 fbf=100.0 cbf=100.0 X=$BASE/X10k_1k_dense C=$BASE/C10k_1k_dense Y=$BASE/y10k_1k_dense YbyC=$BASE/YbyC10k_1k_dense fmt=$FORMAT
${CMD} -f extractTestData.dml $DASH-args $BASE/X10k_1k_dense $BASE/y10k_1k_dense $BASE/X10k_1k_dense_test $BASE/y10k_1k_dense_test $FORMAT

#generate S scenarios (800MB)
#${CMD} -f ../datagen/genRandData4Kmeans.dml $DASH-nvargs nr=100000 nf=1000 nc=50 dc=10.0 dr=1.0 fbf=100.0 cbf=100.0 X=$BASE/X100k_1k_dense C=$BASE/C100k_1k_dense Y=$BASE/y100k_1k_dense YbyC=$BASE/YbyC100k_1k_dense fmt=$FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100k_1k_dense $BASE/y100k_1k_dense $BASE/X100k_1k_dense_test $BASE/y100k_1k_dense_test $FORMAT

#generate M scenarios (8GB)
#${CMD} -f ../datagen/genRandData4Kmeans.dml $DASH-nvargs nr=1000000 nf=1000 nc=50 dc=10.0 dr=1.0 fbf=100.0 cbf=100.0 X=$BASE/X1M_1k_dense C=$BASE/C1M_1k_dense Y=$BASE/y1M_1k_dense YbyC=$BASE/YbyC1M_1k_dense fmt=$FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X1M_1k_dense $BASE/y1M_1k_dense $BASE/X1M_1k_dense_test $BASE/y1M_1k_dense_test $FORMAT

#generate L scenarios (80GB)
#${CMD} -f ../datagen/genRandData4Kmeans.dml $DASH-nvargs nr=10000000 nf=1000 nc=50 dc=10.0 dr=1.0 fbf=100.0 cbf=100.0 X=$BASE/X10M_1k_dense C=$BASE/C10M_1k_dense Y=$BASE/y10M_1k_dense YbyC=$BASE/YbyC10M_1k_dense fmt=$FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X10M_1k_dense $BASE/y10M_1k_dense $BASE/X10M_1k_dense_test $BASE/y10M_1k_dense_test $FORMAT

#generate LARGE scenarios (800GB)
#${CMD} -f ../datagen/genRandData4Kmeans.dml $DASH-nvargs nr=100000000 nf=1000 nc=50 dc=10.0 dr=1.0 fbf=100.0 cbf=100.0 X=$BASE/X100M_1k_dense C=$BASE/C100M_1k_dense Y=$BASE/y100M_1k_dense YbyC=$BASE/YbyC100M_1k_dense fmt=$FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100M_1k_dense $BASE/y100M_1k_dense $BASE/X100M_1k_dense_test $BASE/y100M_1k_dense_test $FORMAT
 
