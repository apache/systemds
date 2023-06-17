#!/usr/bin/env bash
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


CMD=$1
DATA=$2
REPEAT=${3:-1}
VTYPE=${4:-"double"}
DTYPE=${5:-"matrix"}

cp "${DATA}.mtd" "${DATA}.mtd.backup"
sed -i "s/\"data_type\":.*$/\"data_type\": \"${DTYPE}\",/" "${DATA}.mtd"
sed -i "s/\"value_type\":.*$/\"value_type\": \"${VTYPE}\",/" "${DATA}.mtd"
tstart=$(date +%s.%N)
printf "%-10s " "$VTYPE: " >> results/times.txt;
printf "%-16s " "read.dml; " >> results/times.txt;
for n in $(seq $REPEAT)
do
  ${CMD} -f ./scripts/read.dml \
    --config conf/SystemDS-config.xml \
    --stats \
    --nvargs INPUT="$DATA"
done

duration=$(echo "$(date +%s.%N) - $tstart" | bc)
printf "%s\n" "$duration" >> results/times.txt
rm "${DATA}.mtd"
mv "${DATA}.mtd.backup" "${DATA}.mtd"

