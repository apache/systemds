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

import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;

public class OuterVectorBinaryOpFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock>
{
	
	private static final long serialVersionUID = 1730704346934726826L;
	
	private BinaryOperator _op;
	private Broadcast<PartitionedMatrixBlock> _pmV;
	
	public OuterVectorBinaryOpFunction( BinaryOperator op, Broadcast<PartitionedMatrixBlock> binput ) 
	{
		_op = op;
		_pmV = binput;
	}

	@Override
	public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
		throws Exception 
	{
		MatrixIndexes ix = arg0._1();
		MatrixBlock in1 = arg0._2();
		
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
		int pNumColsBlocks = _pmV.value().getNumColumnBlocks();
		for(int i = 1; i <= pNumColsBlocks; i++ ) {
			MatrixBlock in2 = _pmV.value().getMatrixBlock(1, i);
			MatrixBlock resultBlk = (MatrixBlock) (in1.binaryOperations (_op, in2, new MatrixBlock()));
			retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(ix.getRowIndex(), i), resultBlk));
		}
		
		return retVal;
	}
}
