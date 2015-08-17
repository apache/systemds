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

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;

public class AggregateDropCorrectionFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
{
	
	private static final long serialVersionUID = -5573656897943638857L;
	
	private AggregateOperator _op = null;
	
	public AggregateDropCorrectionFunction(AggregateOperator op)
	{
		_op = op;
	}

	@Override
	public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
		throws Exception 
	{
		MatrixIndexes ixIn = arg0._1();
		MatrixBlock blkIn = arg0._2();

		//copy inputs
		MatrixIndexes ixOut = new MatrixIndexes(ixIn);
		MatrixBlock blkOut = new MatrixBlock(blkIn);
		
		//drop correction
		blkOut.dropLastRowsOrColums(_op.correctionLocation);
		
		//output new tuple
		return new Tuple2<MatrixIndexes, MatrixBlock>(ixOut, blkOut);
	}	
}

