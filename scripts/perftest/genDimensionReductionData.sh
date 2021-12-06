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

CMD=$1
BASE=$2/dimensionreduction

FORMAT="binary"
EXTRADOT=""
if [ "${CMD}" == "systemds" ]; then EXTRADOT="." ; fi

#generate XS scenarios (80MB)
${CMD} -f ${EXTRADOT}./datagen/genRandData4PCA.dml --nvargs R=5000 C=2000 OUT=$BASE/pcaData5k_2k_dense FMT=$FORMAT

#generate S scenarios (800MB)
#${CMD} -f ${EXTRADOT}./datagen/genRandData4PCA.dml --nvargs R=50000 C=2000 OUT=$BASE/pcaData50k_2k_dense FMT=$FORMAT

#generate M scenarios (8GB)
#${CMD} -f ${EXTRADOT}./datagen/genRandData4PCA.dml --nvargs R=500000 C=2000 OUT=$BASE/pcaData500k_2k_dense FMT=$FORMAT

#generate L scenarios (80GB)
#${CMD} -f ${EXTRADOT}./datagen/genRandData4PCA.dml --nvargs R=5000000 C=2000 OUT=$BASE/pcaData5M_2k_dense FMT=$FORMAT

#generate XL scenarios (800GB)
#${CMD} -f ${EXTRADOT}./datagen/genRandData4PCA.dml --nvargs R=50000000 C=2000 OUT=$BASE/pcaData50M_2k_dense FMT=$FORMAT

