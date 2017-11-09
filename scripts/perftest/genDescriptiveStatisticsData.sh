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

if [ "$2" == "SPARK" ]; then CMD="./sparkDML.sh "; DASH="-"; elif [ "$2" == "MR" ]; then CMD="hadoop jar SystemML.jar " ; else CMD="echo " ; fi

FORMAT="binary"
BASE=$1/bivar

export HADOOP_CLIENT_OPTS="-Xmx2048m -Xms2048m -Xmn256m"

c=1000
nc=100
mdomain=1100
set=20
labelset=10

#XS data 10K rows
${CMD} -f ../datagen/genRandData4DescriptiveStats.dml $DASH-explain $DASH-stats $DASH-nvargs nr=10000 nf=$c nc=$nc maxdomain=$mdomain X=${BASE}/A_10k/data types=${BASE}/A_10k/types setsize=$set labelsetsize=$labelset types1=${BASE}/A_10k/set1.types types2=${BASE}/A_10k/set2.types index1=${BASE}/A_10k/set1.indices index2=${BASE}/A_10k/set2.indices fmt=$FORMAT

#S data 100K rows
#${CMD} -f ../datagen/genRandData4DescriptiveStats.dml $DASH-explain $DASH-stats $DASH-nvargs nr=100000 nf=$c nc=$nc maxdomain=$mdomain X=${BASE}/A_100k/data types=${BASE}/A_100k/types setsize=$set labelsetsize=$labelset types1=${BASE}/A_100k/set1.types types2=${BASE}/A_100k/set2.types index1=${BASE}/A_100k/set1.indices index2=${BASE}/A_100k/set2.indices fmt=$FORMAT

#M data 1M rows
#${CMD} -f ../datagen/genRandData4DescriptiveStats.dml $DASH-explain $DASH-stats $DASH-nvargs nr=1000000 nf=$c nc=$nc maxdomain=$mdomain X=${BASE}/A_1M/data types=${BASE}/A_1M/types setsize=$set labelsetsize=$labelset types1=${BASE}/A_1M/set1.types types2=${BASE}/A_1M/set2.types index1=${BASE}/A_1M/set1.indices index2=${BASE}/A_1M/set2.indices fmt=$FORMAT

#L data 10M rows
#${CMD} -f ../datagen/genRandData4DescriptiveStats.dml $DASH-explain $DASH-stats $DASH-nvargs nr=10000000 nf=$c nc=$nc maxdomain=$mdomain X=${BASE}/A_10M/data types=${BASE}/A_10M/types setsize=$set labelsetsize=$labelset types1=${BASE}/A_10M/set1.types types2=${BASE}/A_10M/set2.types index1=${BASE}/A_10M/set1.indices index2=${BASE}/A_10M/set2.indices fmt=$FORMAT
