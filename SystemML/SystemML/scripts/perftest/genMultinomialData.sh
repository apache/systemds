#!/bin/bash

if [ "$1" == "" -o "$2" == "" ]; then echo "Usage: $0 <hdfsDataDir> <MR | SPARK | ECHO>   e.g. $0 perftest SPARK" ; exit 1 ; fi
if [ "$2" == "SPARK" ]; then CMD="./sparkDML "; DASH="-"; elif [ "$2" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi


BASE=$1/multinomial

FORMAT="csv" 
DENSE_SP=0.9
SPARSE_SP=0.01

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

#generate XS scenarios (80MB)
${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 10000 1000 $DENSE_SP 150 0 $BASE/X10k_1k_dense_k150_csv $BASE/y10k_1k_dense_k150_csv $FORMAT
${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 10000 1000 $SPARSE_SP 150 0 $BASE/X10k_1k_sparse_k150_csv $BASE/y10k_1k_sparse_k150_csv $FORMAT
${CMD} -f extractTestData.dml $DASH-args $BASE/X10k_1k_dense_k150_csv $BASE/y10k_1k_dense_k150_csv $BASE/X10k_1k_dense_k150_csv_test $BASE/y10k_1k_dense_k150_csv_test $FORMAT
${CMD} -f extractTestData.dml $DASH-args $BASE/X10k_1k_sparse_k150_csv $BASE/y10k_1k_sparse_k150_csv $BASE/X10k_1k_sparse_k150_csv_test $BASE/y10k_1k_sparse_k150_csv_test $FORMAT

##generate S scenarios (80MB)
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 100000 1000 $DENSE_SP 150 0 $BASE/X100k_1k_dense_k150_csv $BASE/y100k_1k_dense_k150_csv $FORMAT
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 100000 1000 $SPARSE_SP 150 0 $BASE/X100k_1k_sparse_k150_csv $BASE/y100k_1k_sparse_k150_csv $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100k_1k_dense_k150_csv $BASE/y100k_1k_dense_k150_csv $BASE/X100k_1k_dense_k150_csv_test $BASE/y100k_1k_dense_k150_csv_test $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100k_1k_sparse_k150_csv $BASE/y100k_1k_sparse_k150_csv $BASE/X100k_1k_sparse_k150_csv_test $BASE/y100k_1k_sparse_k150_csv_test $FORMAT
#
##generate M scenarios (8GB)
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 1000000 1000 $DENSE_SP 150 0 $BASE/X1M_1k_dense_k150_csv $BASE/y1M_1k_dense_k150_csv $FORMAT
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 1000000 1000 $SPARSE_SP 150 0 $BASE/X1M_1k_sparse_k150_csv $BASE/y1M_1k_sparse_k150_csv $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X1M_1k_dense_k150_csv $BASE/y1M_1k_dense_k150_csv $BASE/X1M_1k_dense_k150_csv_test $BASE/y1M_1k_dense_k150_csv_test $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X1M_1k_sparse_k150_csv $BASE/y1M_1k_sparse_k150_csv $BASE/X1M_1k_sparse_k150_csv_test $BASE/y1M_1k_sparse_k150_csv_test $FORMAT
#
##generate L scenarios (80GB)
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 10000000 1000 $DENSE_SP 150 0 $BASE/X10M_1k_dense_k150_csv $BASE/y10M_1k_dense_k150_csv $FORMAT
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 10000000 1000 $SPARSE_SP 150 0 $BASE/X10M_1k_sparse_k150_csv $BASE/y10M_1k_sparse_k150_csv $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X10M_1k_dense_k150_csv $BASE/y10M_1k_dense_k150_csv $BASE/X10M_1k_dense_k150_csv_test $BASE/y10M_1k_dense_k150_csv_test $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X10M_1k_sparse_k150_csv $BASE/y10M_1k_sparse_k150_csv $BASE/X10M_1k_sparse_k150_csv_test $BASE/y10M_1k_sparse_k150_csv_test $FORMAT
#
##generate LARGE scenarios (800GB)
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 100000000 1000 $DENSE_SP 150 0 $BASE/X100M_1k_dense_k150_csv $BASE/y100M_1k_dense_k150_csv $FORMAT
#${CMD} -f ../datagen/genRandData4Multinomial.dml $DASH-args 100000000 1000 $SPARSE_SP 150 0 $BASE/X100M_1k_sparse_k150_csv $BASE/y100M_1k_sparse_k150_csv $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100M_1k_dense_k150_csv $BASE/y100M_1k_dense_k150_csv $BASE/X100M_1k_dense_k150_csv_test $BASE/y100M_1k_dense_k150_csv_test $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100M_1k_sparse_k150_csv $BASE/y100M_1k_sparse_k150_csv $BASE/X100M_1k_sparse_k150_csv_test $BASE/y100M_1k_sparse_k150_csv_test $FORMAT
