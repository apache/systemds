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

import java.util.ArrayList;

import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;

import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;


public class ReplicateVectorFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock> 
{
	
	private static final long serialVersionUID = -1505557561471236851L;
	
	private boolean _byRow; 
	private long _numReplicas;
	
	public ReplicateVectorFunction(boolean byRow, long numReplicas) 
	{
		_byRow = byRow;
		_numReplicas = numReplicas;
	}
	
	@Override
	public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
		throws Exception 
	{
		MatrixIndexes ix = arg0._1();
		MatrixBlock mb = arg0._2();
		
		//sanity check inputs
		if(_byRow && (ix.getRowIndex() != 1 || mb.getNumRows()>1) ) {
			throw new Exception("Expected a row vector in ReplicateVector");
		}
		if(!_byRow && (ix.getColumnIndex() != 1 || mb.getNumColumns()>1) ) {
			throw new Exception("Expected a column vector in ReplicateVector");
		}
		
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();
		for(int i = 1; i <= _numReplicas; i++) {
			if( _byRow )
				retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(i, ix.getColumnIndex()), mb));
			else
				retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(ix.getRowIndex(), i), mb));
		}
		
		return retVal;
	}
}
