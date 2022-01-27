#!/usr/bin/env bash

path_1="$1/benchmark"
path_2="$path_1/$3"
mkdir -p $1
mkdir -p "$path_1"
mkdir -p "$path_2"

log_file="$path_2/$2.csv"
if test ! -f "$log_file"; then
  touch $log_file
  echo "dataset,data_nrows,data_ncols,col_index_percent,sample_nrows,generate_time,read_time" > $log_file
fi
