#!/bin/bash
#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------
if [ "$(basename $PWD)" != "perftest" ];
then
  echo "Please execute scripts from directory 'perftest'"
  exit 1;
fi

CMD=${1:-"systemds"}
DATADIR=${2:-"temp"}/io
MAXMEM=${3:-1}
REPEATS=${4:-1}

DATA=()
if [ $MAXMEM -ge 1 ]; then DATA+=("500_250_dense"); fi
if [ $MAXMEM -ge 10 ]; then DATA+=("5k_250_dense"); fi
if [ $MAXMEM -ge 80 ]; then DATA+=("10k_1k_dense"); fi
if [ $MAXMEM -ge 800 ]; then DATA+=("100k_1k_dense"); fi
if [ $MAXMEM -ge 8000 ]; then DATA+=("1M_1k_dense"); fi
if [ $MAXMEM -ge 80000 ]; then DATA+=("10M_1k_dense"); fi
if [ $MAXMEM -ge 800000 ]; then DATA+=("100M_1k_dense"); fi

echo "RUN IO Benchmarks: " $(date) >> results/times.txt;

execute_python_script () {
  script=$1
  input=$2
  repeats=$3
  DTYPE=$4
  printf "%-16s " "${script}; " >> results/times.txt;
  if [ -z "$DTYPE" ]; then
    TIME_IO=$(python ./python/io/${script} ${input} ${repeats});
  else
    TIME_IO=$(python ./python/io/${script} ${input} ${repeats} --dtype ${DTYPE});
  fi
  printf "%s\n" "$TIME_IO" >> results/times.txt
}

for d in ${DATA[@]}
do
  echo "-- Running IO benchmarks on "$d >> results/times.txt;
  DATAFILE="$DATADIR/X$d"
  F="runIO.sh" 
  for vtype in "double" "int" "string" "boolean"
  do
    . ./$F $CMD $DATAFILE $REPEATS $vtype
    cp "${DATAFILE}.mtd" "${DATAFILE}.mtd.backup" 
    sed -i "s/\"value_type\":.*$/\"value_type\": \"${vtype}\",/" "${DATAFILE}.mtd"
    printf "%-10s " "${vtype}: " >> results/times.txt;
    execute_python_script "load_native.py" $DATAFILE $REPEATS
    rm "${DATAFILE}.mtd"
    mv "${DATAFILE}.mtd.backup" "${DATAFILE}.mtd"
  done
  for vtype in "double" "float" "long" "int64" "int32" "uint8" "string" "bool"
  do
    printf "%-10s " "${vtype}: " >> results/times.txt;
    execute_python_script "load_numpy.py" $DATAFILE $REPEATS $vtype
    printf "%-10s " "${vtype}: " >> results/times.txt;
    execute_python_script "load_pandas.py" $DATAFILE $REPEATS $vtype
  done
done

echo -e "\n\n" >> results/times.txt
