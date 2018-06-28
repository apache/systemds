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

package org.apache.sysml.runtime.controlprogram.paramserv.spark;

import org.apache.spark.api.java.function.Function2;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

import scala.Tuple2;

/**
 * Reducer allowing to append the matrices of features and labels
 */
public class DataPartitionerSparkReducer implements Function2<Tuple2<MatrixBlock, MatrixBlock>, Tuple2<MatrixBlock, MatrixBlock>, Tuple2<MatrixBlock, MatrixBlock>> {

	@Override
	public Tuple2<MatrixBlock, MatrixBlock> call(Tuple2<MatrixBlock, MatrixBlock> input1, Tuple2<MatrixBlock, MatrixBlock> input2) throws Exception {
		MatrixBlock features = ParamservUtils.rbindMatrix(input1._1, input2._1);
		MatrixBlock labels = ParamservUtils.rbindMatrix(input1._2, input2._2);
		return new Tuple2<>(features, labels);
	}

}
