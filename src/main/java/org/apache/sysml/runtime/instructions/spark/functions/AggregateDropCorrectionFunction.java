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

package org.apache.sysml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function;

import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;

public class AggregateDropCorrectionFunction implements Function<MatrixBlock, MatrixBlock> 
{
	
	private static final long serialVersionUID = -5573656897943638857L;
	
	private AggregateOperator _op = null;
	
	public AggregateDropCorrectionFunction(AggregateOperator op)
	{
		_op = op;
	}

	@Override
	public MatrixBlock call(MatrixBlock arg0) 
		throws Exception 
	{
		//create output block copy
		MatrixBlock blkOut = new MatrixBlock(arg0);
		
		//drop correction
		blkOut.dropLastRowsOrColums(_op.correctionLocation);
		
		return blkOut;
	}	
}

