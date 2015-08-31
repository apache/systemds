/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.util.ArrayList;
import java.util.Iterator;

import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.lops.BinaryM.VectorType;
import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;

public class MatrixVectorBinaryOpPartitionFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes,MatrixBlock>>, MatrixIndexes,MatrixBlock>
{
	private static final long serialVersionUID = 9096091404578628534L;
	
	private BinaryOperator _op = null;
	private Broadcast<PartitionedMatrixBlock> _pmV = null;
	private VectorType _vtype = null;
	
	public MatrixVectorBinaryOpPartitionFunction( BinaryOperator op, Broadcast<PartitionedMatrixBlock> binput, VectorType vtype ) 
	{
		_op = op;
		_pmV = binput;
		_vtype = vtype;
	}

	@Override
	public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg0) 
		throws Exception 
	{
		//get the broadcast input
		PartitionedMatrixBlock pmV = _pmV.value();
		
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retList = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();
		
		while( arg0.hasNext() ) 
		{
			//unpack partition key-value pairs
			Tuple2<MatrixIndexes, MatrixBlock> tmp = arg0.next();
			MatrixIndexes ix = tmp._1();
			MatrixBlock in1 = tmp._2();
			
			//get the rhs block 
			int rix= (int)((_vtype==VectorType.COL_VECTOR) ? ix.getRowIndex() : 1);
			int cix= (int)((_vtype==VectorType.COL_VECTOR) ? 1 : ix.getColumnIndex());
			MatrixBlock in2 = pmV.getMatrixBlock(rix, cix);
				
			//execute the binary operation
			MatrixBlock ret = (MatrixBlock) (in1.binaryOperations (_op, in2, new MatrixBlock()));
			retList.add( new Tuple2<MatrixIndexes, MatrixBlock>(ix, ret) );
		}
		
		return retList;
	}
}
