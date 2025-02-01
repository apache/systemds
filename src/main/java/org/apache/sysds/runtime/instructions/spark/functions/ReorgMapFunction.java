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

import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.functionobjects.DiagIndex;
import org.apache.sysds.runtime.functionobjects.IndexFunction;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;

import scala.Tuple2;

public class ReorgMapFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
{
	private static final long serialVersionUID = 31065772250744103L;
	
	private ReorgOperator _reorgOp = null;
	private IndexFunction _indexFnObject = null;
	
	public ReorgMapFunction(String opcode) {
		if(opcode.equalsIgnoreCase(Opcodes.TRANSPOSE.toString())) {
			_indexFnObject = SwapIndex.getSwapIndexFnObject();
		}
		else if(opcode.equalsIgnoreCase(Opcodes.DIAG.toString())) { //diagM2V
			_indexFnObject = DiagIndex.getDiagIndexFnObject(false);
		}
		else {
			throw new DMLRuntimeException("Incorrect opcode for RDDReorgMapFunction:" + opcode);
		}
		_reorgOp = new ReorgOperator(_indexFnObject);
	}
	
	@Override
	public Tuple2<MatrixIndexes, MatrixBlock> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
		throws Exception 
	{
		MatrixIndexes ixIn = arg0._1();
		MatrixBlock blkIn = arg0._2();
		//swap the matrix indexes
		MatrixIndexes ixOut = new MatrixIndexes(ixIn);
		_indexFnObject.execute(ixIn, ixOut);
		//swap the matrix block data
		MatrixBlock blkOut = blkIn.reorgOperations(_reorgOp, new MatrixBlock(), -1, -1, -1);
		//output new tuple
		return new Tuple2<>(ixOut,blkOut);
	}
}

