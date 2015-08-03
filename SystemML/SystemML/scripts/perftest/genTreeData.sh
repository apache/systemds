#!/bin/bash

if [ "$1" == "" -o "$2" == "" ]; then echo "Usage: $0 <hdfsDataDir> <MR | SPARK | ECHO>   e.g. $0 perftest SPARK" ; exit 1 ; fi
if [ "$2" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$2" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi


BASE=$1/trees

FORMAT="csv" 
DENSE_SP=0.9
SPARSE_SP=0.01

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

echo "NOT DONE YET. WAITING FOR DML SCRIPT FROM FARAZ" ; exit 1


#generate XS scenarios (80MB)
${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-args 10000 1000 5 5 $BASE/w10k_1k_dense_csv $BASE/X10k_1k_dense_csv $BASE/y10k_1k_dense_csv 1 0 $DENSE_SP $FORMAT
${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-args 10000 1000 5 5 $BASE/w10k_1k_sparse_csv $BASE/X10k_1k_sparse_csv $BASE/y10k_1k_sparse_csv 1 0 $SPARSE_SP $FORMAT
${CMD} -f extractTestData.dml $DASH-args $BASE/X10k_1k_dense_csv $BASE/y10k_1k_dense_csv $BASE/X10k_1k_dense_csv_test $BASE/y10k_1k_dense_csv_test $FORMAT
${CMD} -f extractTestData.dml $DASH-args $BASE/X10k_1k_sparse_csv $BASE/y10k_1k_sparse_csv $BASE/X10k_1k_sparse_csv_test $BASE/y10k_1k_sparse_csv_test $FORMAT

##generate S scenarios (800MB)
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-args 100000 1000 5 5 $BASE/w100k_1k_dense_csv $BASE/X100k_1k_dense_csv $BASE/y100k_1k_dense_csv 1 0 $DENSE_SP $FORMAT
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-args 100000 1000 5 5 $BASE/w100k_1k_sparse_csv $BASE/X100k_1k_sparse_csv $BASE/y100k_1k_sparse_csv 1 0 $SPARSE_SP $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100k_1k_dense_csv $BASE/y100k_1k_dense_csv $BASE/X100k_1k_dense_csv_test $BASE/y100k_1k_dense_csv_test $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100k_1k_sparse_csv $BASE/y100k_1k_sparse_csv $BASE/X100k_1k_sparse_csv_test $BASE/y100k_1k_sparse_csv_test $FORMAT
#
##generate M scenarios (8GB)
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-args 1000000 1000 5 5 $BASE/w1M_1k_dense_csv $BASE/X1M_1k_dense_csv $BASE/y1M_1k_dense_csv 1 0 $DENSE_SP $FORMAT
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-args 1000000 1000 5 5 $BASE/w1M_1k_sparse_csv $BASE/X1M_1k_sparse_csv $BASE/y1M_1k_sparse_csv 1 0 $SPARSE_SP $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X1M_1k_dense_csv $BASE/y1M_1k_dense_csv $BASE/X1M_1k_dense_csv_test $BASE/y1M_1k_dense_csv_test $FORMAT
#${CMD}-f extractTestData.dml $DASH-args $BASE/X1M_1k_sparse_csv $BASE/y1M_1k_sparse_csv $BASE/X1M_1k_sparse_csv_test $BASE/y1M_1k_sparse_csv_test $FORMAT
#
##generate L scenarios (80GB)
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-args 10000000 1000 5 5 $BASE/w10M_1k_dense_csv $BASE/X10M_1k_dense_csv $BASE/y10M_1k_dense_csv 1 0 $DENSE_SP $FORMAT
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-args 10000000 1000 5 5 $BASE/w10M_1k_sparse_csv $BASE/X10M_1k_sparse_csv $BASE/y10M_1k_sparse_csv 1 0 $SPARSE_SP $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X10M_1k_dense_csv $BASE/y10M_1k_dense_csv $BASE/X10M_1k_dense_csv_test $BASE/y10M_1k_dense_csv_test $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X10M_1k_sparse_csv $BASE/y10M_1k_sparse_csv $BASE/X10M_1k_sparse_csv_test $BASE/y10M_1k_sparse_csv_test $FORMAT
#
##generate XL scenarios (800GB)
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-args 100000000 1000 5 5 $BASE/w100M_1k_dense_csv $BASE/X100M_1k_dense_csv $BASE/y100M_1k_dense_csv 1 0 $DENSE_SP $FORMAT
#${CMD} -f ../datagen/genRandData4LogisticRegression.dml $DASH-args 100000000 1000 5 5 $BASE/w100M_1k_sparse_csv $BASE/X100M_1k_sparse_csv $BASE/y100M_1k_sparse_csv 1 0 $SPARSE_SP $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100M_1k_dense_csv $BASE/y100M_1k_dense_csv $BASE/X100M_1k_dense_csv_test $BASE/y100M_1k_dense_csv_test $FORMAT
#${CMD} -f extractTestData.dml $DASH-args $BASE/X100M_1k_sparse_csv $BASE/y100M_1k_sparse_csv $BASE/X100M_1k_sparse_csv_test $BASE/y100M_1k_sparse_csv_test $FORMAT

