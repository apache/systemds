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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.sysml.lops.Checkpoint;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.SparseBlock;

public class CopyBlockPairFunction
		implements MapFunction<Tuple2<MatrixIndexes, MatrixBlock>, Tuple2<MatrixIndexes, MatrixBlock>> {

	private boolean _deepCopy = true;

	public CopyBlockPairFunction() {
		this(true);
	}

	public CopyBlockPairFunction(boolean deepCopy) {
		_deepCopy = deepCopy;
	}

	@Override
	public Tuple2<MatrixIndexes, MatrixBlock> map(Tuple2<MatrixIndexes, MatrixBlock> arg0)
			throws Exception {
		if (_deepCopy) {
			MatrixIndexes ix = new MatrixIndexes(arg0.f0);
			MatrixBlock block = null;
			//always create deep copies in more memory-efficient CSR representation
			//if block is already in sparse format
			if (Checkpoint.CHECKPOINT_SPARSE_CSR && arg0.f1.isInSparseFormat())
				block = new MatrixBlock(arg0.f1, SparseBlock.Type.CSR, true);
			else
				block = new MatrixBlock(arg0.f1);
			return new Tuple2<MatrixIndexes, MatrixBlock>(ix, block);
		} else {
			return arg0;
		}
	}
}
