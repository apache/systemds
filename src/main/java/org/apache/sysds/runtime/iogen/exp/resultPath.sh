#!/usr/bin/env bash

log_file="$1.csv"
if test ! -f "$log_file"; then
  touch $log_file
  echo "dataset,data_nrows,data_ncols,col_selected_count,sample_nrows,generate_time,read_time" > $log_file
fi
