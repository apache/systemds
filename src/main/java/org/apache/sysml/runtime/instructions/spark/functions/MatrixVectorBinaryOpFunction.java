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

package org.apache.sysml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import org.apache.sysml.lops.BinaryM.VectorType;
import org.apache.sysml.runtime.instructions.spark.data.PartitionedBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;

public class MatrixVectorBinaryOpFunction implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock>
{
	
	private static final long serialVersionUID = -7695883019452417300L;
	
	private BinaryOperator _op = null;
	private Broadcast<PartitionedBlock<MatrixBlock>> _pmV = null;
	private VectorType _vtype = null;
	
	public MatrixVectorBinaryOpFunction( BinaryOperator op, Broadcast<PartitionedBlock<MatrixBlock>> binput, VectorType vtype ) 
	{
		_op = op;
		_pmV = binput;
		_vtype = vtype;
	}

	@Override
	public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
		throws Exception 
	{
		MatrixIndexes ix = arg0._1();
		MatrixBlock in1 = arg0._2();
		
		//get the rhs block 
		int rix= (int)((_vtype==VectorType.COL_VECTOR) ? ix.getRowIndex() : 1);
		int cix= (int)((_vtype==VectorType.COL_VECTOR) ? 1 : ix.getColumnIndex());
		MatrixBlock in2 = _pmV.value().getBlock(rix, cix);
			
		//execute the binary operation
		MatrixBlock ret = (MatrixBlock) (in1.binaryOperations (_op, in2, new MatrixBlock()));
		return new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(ix), ret);
	}
}
