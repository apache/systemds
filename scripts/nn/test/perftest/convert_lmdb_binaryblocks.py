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


import lmdb, caffe
import numpy as np
import sys
lmdb_file = sys.argv[1]
output_file_x = sys.argv[2]
output_file_y = sys.argv[3]
BUFFER_SIZE = int(sys.argv[4])
import lmdb, caffe
lmdb_cursor = lmdb.open(lmdb_file, readonly=True).begin().cursor()
datum = caffe.proto.caffe_pb2.Datum()
all_features = None
all_labels = None

from pyspark import SparkContext
sc = SparkContext()
# To turn off unnecessary systemml warnings
sc.setLogLevel("ERROR")
from systemml import MLContext, dml
ml = MLContext(sc)
isFirst = True

def writeDMLStr(varName, fileName):
	return 'write(' + varName + ', "' + fileName + '", format="binary"); '

def appendDMLStr(varName, fileName):
	return 'tmp' + varName + ' = read("' + fileName + '"); ' + varName + ' = rbind(tmp' + varName + ', ' + varName + ');'

dim2_label = 1
def write(delta_features, delta_labels):
	global isFirst, sc, output_file_x, output_file_y, ml
	scriptStr = writeDMLStr('X', output_file_x) + writeDMLStr('y', output_file_y)
	if isFirst:
		isFirst = False
		dim2_label = delta_labels.shape[1]
	else:
		# Append avoids creation of very large in-memory numpy array
		scriptStr = appendDMLStr('X', output_file_x) + appendDMLStr('y', output_file_y) + scriptStr
	print('Converting next batch of input LMDB training points to binary blocks')
	ml.execute(dml(scriptStr).input(X=delta_features, y=delta_labels))


i = 0
for _, value in lmdb_cursor:
    datum.ParseFromString(value)
    data = caffe.io.datum_to_array(datum).flatten()
    label = np.asarray(datum.label)
    all_features = data if all_features is None else np.vstack((all_features, data))
    all_labels = label if all_labels is None else np.vstack((all_labels, label))
    i = i + 1
    if i == BUFFER_SIZE:
    	write(all_features, all_labels)
    	all_features = None
    	all_labels = None
    	i = 0


if all_features is not None:
	write(all_features, all_labels)

if dim2_label == 1:
	# convert to 1-based labels
	scriptStr = 'y = read("' + output_file_y + '"); if(min(y) == 0) {  y = y+1; write(y, "' + output_file_y + '", format="binary"); }'
	ml.execute(dml(scriptStr))
