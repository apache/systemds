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

if [ "$(basename $PWD)" != "perftest" ]; then
  echo "Please execute scripts from directory 'perftest'"
  exit 1
fi

if ! command -v perf &>/dev/null; then
  echo "Perf stat not installed for matrix operation benchmarks, see README"
  exit 0
fi

CMD=$1

rep=2
innerRep=300
is=("100 1000 5000")
js=("100 1000 5000")
ks=("100 1000 5000")
spar=("1.0 0.35 0.1 0.01")
confs=("conf/std.xml conf/mkl.xml conf/openblas.xml")

# is=("1000")
# js=("1000")
# ks=("1000")
# spar=("1.0 0.01")
# confs=("conf/mkl.xml")
# confs=("conf/openblas.xml")

# Logging output
mkdir -p logs
LogName='logs/MM.log'
rm -f $LogName     # full log file
rm -f $LogName.log # Reduced log file

echo "MATRIX MULTIPLICATION" >>results/times.txt

for i in $is; do
  for j in $js; do
    for k in $ks; do
      for con in $confs; do

        tstart=$(date +%s.%N)

        perf stat -d -d -d -r $rep \
          ${CMD} scripts/MM.dml \
          -config $con \
          -stats \
          -args $i $j $k 1.0 1.0 $innerRep \
          >>$LogName 2>&1
        ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
        echo "Matrix mult $i x $j %*% $j x $k $con:" $ttrain >>results/times.txt

      done
      for sl in $spar; do
        for sr in $spar; do
          tstart=$(date +%s.%N)

          perf stat -d -d -d -r $rep \
            ${CMD} scripts/MM.dml \
            -config conf/std.xml \
            -stats \
            -args $i $j $k $sl $sr $innerRep \
            >>$LogName 2>&1
          ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
          echo "Matrix mult $i x $j %*% $j x $k spL $sl spR $sr :" $ttrain >>results/times.txt

        done
      done
    done
  done
done

echo -e "\n\n" >>results/times.txt

cat $LogName | grep -E ' ba\+\* |Total elapsed time|-----------| instructions |  cycles | CPUs utilized ' >>$LogName.log
