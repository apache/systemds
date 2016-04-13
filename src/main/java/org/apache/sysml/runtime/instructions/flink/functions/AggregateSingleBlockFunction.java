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

package org.apache.sysml.runtime.instructions.flink.functions;

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;

public class AggregateSingleBlockFunction implements ReduceFunction<MatrixBlock> {

	private static final long serialVersionUID = -3672377410407066396L;

	private final AggregateOperator _op;
	private MatrixBlock _corr;

	public AggregateSingleBlockFunction(AggregateOperator op) {
		_op = op;
		_corr = null;
	}

	@Override
	public MatrixBlock reduce(MatrixBlock value1, MatrixBlock value2) throws Exception {
		//copy one first input
		MatrixBlock out = new MatrixBlock(value1);

		//create correction block (on demand)
		if (_corr == null) {
			_corr = new MatrixBlock(value1.getNumRows(), value1.getNumColumns(), false);
		}

		//aggregate second input
		if (_op.correctionExists) {
			OperationsOnMatrixValues.incrementalAggregation(
					out, _corr, value2, _op, true);
		} else {
			OperationsOnMatrixValues.incrementalAggregation(
					out, null, value2, _op, true);
		}

		return out;
	}
}
