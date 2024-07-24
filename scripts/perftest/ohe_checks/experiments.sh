#!/bin/bash

mkdir BaselineLogs
mkdir OHELogs
run_base() {
    $SYSTEMDS_ROOT/bin/systemds $SYSTEMDS_ROOT/target/SystemDS.jar experiment.dml \
    --seed 42 --debug -nvargs rows=$1 cols=$2 dummy="$3" distinct=$4 > BaselineLogs/${5}_${1}_rows_${2}_cols_${3}_encoded_base.txt 2>&1
}

run_ohe() {
    $SYSTEMDS_ROOT/bin/systemds $SYSTEMDS_ROOT/target/SystemDS.jar experiment.dml \
    --seed 42 --debug --config ohe.xml -nvargs rows=$1 cols=$2 dummy="$3" distinct=$4> OHELogs/${5}_${1}_rows_${2}_cols_${3}_encoded_ohe.txt 2>&1
}

# Run same experiments but checking One-Hot-Encoded columns first
run_ohe 1000 1 "[1]" 10 1
run_ohe 1000 5 "[2]" 10 2
run_ohe 1000 5 "[1,2]" 10 3
run_ohe 1000 5 "[1,2,3]" 10 4
run_ohe 1000 5 "[1,2,3,4,5]" 10 5
run_ohe 1000 10 "[1,3,5]" 10 6
run_ohe 1000 10 "[1,2,5,6]" 10 7
run_ohe 100000 1 "[1]" 100 8
run_ohe 100000 5 "[1,2]" 100 9
run_ohe 100000 5 "[1,2,3]" 100 10
run_ohe 100000 100 "[1,3,50,60,70,80]" 100 11
run_ohe 100000 100 "[1,2,24,25,50,51]" 100 12

# Run baseline experiments
run_base 1000 1 "[1]" 10 1
run_base 1000 5 "[2]" 10 2
run_base 1000 5 "[1,2]" 10 3
run_base 1000 5 "[1,2,3]" 10 4
run_base 1000 5 "[1,2,3,4,5]" 10 5
run_base 1000 10 "[1,3,5]" 10 6
run_base 1000 10 "[1,2,5,6]" 10 7
run_base 100000 1 "[1]" 100 8
run_base 100000 5 "[1,2]" 100 9
run_base 100000 5 "[1,2,3]" 100 10
run_base 100000 100 "[1,3,50,60,70,80]" 100 11
run_base 100000 100 "[1,2,24,25,50,51]" 100 12