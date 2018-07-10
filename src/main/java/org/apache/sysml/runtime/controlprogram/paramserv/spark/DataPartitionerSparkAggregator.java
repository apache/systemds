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

import java.io.Serializable;
import java.util.LinkedList;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

import scala.Tuple2;

public class DataPartitionerSparkAggregator implements PairFunction<Tuple2<Integer,LinkedList<Tuple2<Long,Tuple2<MatrixBlock,MatrixBlock>>>>, Integer, Tuple2<MatrixBlock, MatrixBlock>>, Serializable {

	private static final long serialVersionUID = -1245300852709085117L;

	public DataPartitionerSparkAggregator() {

	}

	/**
	 * Row-wise combine the matrix
	 * @param input workerID => [(rowBlockID, (features, labels))]
	 * @return workerID => [(features, labels)]
	 * @throws Exception Some exception
	 */
	@Override
	public Tuple2<Integer, Tuple2<MatrixBlock, MatrixBlock>> call(Tuple2<Integer, LinkedList<Tuple2<Long, Tuple2<MatrixBlock, MatrixBlock>>>> input) throws Exception {
		MatrixBlock fmb = null;
		MatrixBlock lmb = null;

		for (Tuple2<Long, Tuple2<MatrixBlock, MatrixBlock>> t : input._2) {
			MatrixBlock tmpFMB = t._2._1;
			MatrixBlock tmpLMB = t._2._2;
			if (fmb == null && lmb == null) {
				fmb = tmpFMB;
				lmb = tmpLMB;
				continue;
			}
			// Row-wise aggregation
			fmb = ParamservUtils.rbindMatrix(fmb, tmpFMB);
			lmb = ParamservUtils.rbindMatrix(lmb, tmpLMB);
		}
		return new Tuple2<>(input._1, new Tuple2<>(fmb, lmb));
	}
}
