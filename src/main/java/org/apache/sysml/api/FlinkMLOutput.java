/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.api;

import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;

import java.util.HashMap;

public class FlinkMLOutput extends MLOutput {

    HashMap<String, DataSet<Tuple2<MatrixIndexes, MatrixBlock>>> _outputs;

    public FlinkMLOutput(HashMap<String, DataSet<Tuple2<MatrixIndexes, MatrixBlock>>> outputs,
                         HashMap<String, MatrixCharacteristics> outMetadata) {
        super(outMetadata);
        this._outputs = outputs;
    }

    public DataSet<Tuple2<MatrixIndexes, MatrixBlock>> getBinaryBlockedDataSet(
            String varName) throws DMLRuntimeException {
        if (_outputs.containsKey(varName)) {
            return _outputs.get(varName);
        }
        throw new DMLRuntimeException("Variable " + varName + " not found in the output symbol table.");
    }

    // TODO this should be refactored (Superclass MLOutput with Spark and Flink specific subclasses...)
    @Override
    public JavaPairRDD<MatrixIndexes, MatrixBlock> getBinaryBlockedRDD(String varName) throws DMLRuntimeException {
        throw new DMLRuntimeException("FlinkOutput can't return Spark RDDs!");
    }
}
