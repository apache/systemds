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

package org.apache.sysds.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;

import scala.Tuple2;

public class MatrixMatrixBinaryOpFunction implements Function<Tuple2<MatrixBlock, MatrixBlock>, MatrixBlock> {
	// private static final Log LOG = LogFactory.getLog(MatrixMatrixBinaryOpFunction.class.getName());
	private static final long serialVersionUID = -2683276102742977900L;

	private BinaryOperator _bop;

	public MatrixMatrixBinaryOpFunction(BinaryOperator op) {
		_bop = op;
	}

	@Override
	public MatrixBlock call(Tuple2<MatrixBlock, MatrixBlock> arg0) throws Exception {
		MatrixBlock left = arg0._1();
		MatrixBlock right = arg0._2();
		if(right instanceof CompressedMatrixBlock)
			return ((CompressedMatrixBlock) right).binaryOperationsLeft(_bop, left, new MatrixBlock());
		else
			return left.binaryOperations(_bop, right, new MatrixBlock());

	}
}
