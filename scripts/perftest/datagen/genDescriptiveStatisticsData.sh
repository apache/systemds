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

CMD=$1
BASE=$2/bivar
MAXMEM=$3

FORMAT="binary"

c=1000
nc=100
mdomain=1100
set=20
labelset=10

#XS data 10K rows
if [ $MAXMEM -ge 80 ]; then
  ${CMD} -f datagen/genRandData4DescriptiveStats.dml --explain --stats --nvargs R=10000 C=$c NC=$nc MAXDOMAIN=$mdomain DATA=${BASE}/A_10k/data TYPES=${BASE}/A_10k/types SETSIZE=$set LABELSETSIZE=$labelset TYPES1=${BASE}/A_10k/set1.types TYPES2=${BASE}/A_10k/set2.types INDEX1=${BASE}/A_10k/set1.indices INDEX2=${BASE}/A_10k/set2.indices FMT=$FORMAT &
fi

#S data 100K rows
if [ $MAXMEM -ge 800 ]; then
  ${CMD} -f datagen/genRandData4DescriptiveStats.dml --explain --stats --nvargs R=100000 C=$c NC=$nc MAXDOMAIN=$mdomain DATA=${BASE}/A_100k/data TYPES=${BASE}/A_100k/types SETSIZE=$set LABELSETSIZE=$labelset TYPES1=${BASE}/A_100k/set1.types TYPES2=${BASE}/A_100k/set2.types INDEX1=${BASE}/A_100k/set1.indices INDEX2=${BASE}/A_100k/set2.indices FMT=$FORMAT &
fi

#M data 1M rows
if [ $MAXMEM -ge 8000 ]; then
  ${CMD} -f datagen/genRandData4DescriptiveStats.dml --explain --stats --nvargs R=1000000 C=$c NC=$nc MAXDOMAIN=$mdomain DATA=${BASE}/A_1M/data TYPES=${BASE}/A_1M/types SETSIZE=$set LABELSETSIZE=$labelset TYPES1=${BASE}/A_1M/set1.types TYPES2=${BASE}/A_1M/set2.types INDEX1=${BASE}/A_1M/set1.indices INDEX2=${BASE}/A_1M/set2.indices FMT=$FORMAT &
fi

#L data 10M rows
if [ $MAXMEM -ge 80000 ]; then
  ${CMD} -f datagen/genRandData4DescriptiveStats.dml --explain --stats --nvargs R=10000000 C=$c NC=$nc MAXDOMAIN=$mdomain DATA=${BASE}/A_10M/data TYPES=${BASE}/A_10M/types SETSIZE=$set LABELSETSIZE=$labelset TYPES1=${BASE}/A_10M/set1.types TYPES2=${BASE}/A_10M/set2.types INDEX1=${BASE}/A_10M/set1.indices INDEX2=${BASE}/A_10M/set2.indices FMT=$FORMAT
fi

wait