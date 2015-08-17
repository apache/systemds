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

import org.apache.spark.api.java.function.Function;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

/**
 * 
 */
public class FilterDiagBlocksFunction implements Function<Tuple2<MatrixIndexes,MatrixBlock>, Boolean> 
{
	
	private static final long serialVersionUID = 685221977882289849L;
	
	@Override
	public Boolean call(Tuple2<MatrixIndexes, MatrixBlock> arg0)
		throws Exception 
	{
		//returns true for matrix blocks on matrix diagonal
		MatrixIndexes ix = arg0._1();
		return (ix.getRowIndex() == ix.getColumnIndex());
	}
	

}
