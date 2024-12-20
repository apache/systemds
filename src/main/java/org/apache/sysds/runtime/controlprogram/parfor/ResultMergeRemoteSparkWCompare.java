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

package org.apache.sysds.runtime.controlprogram.parfor;

import java.util.Iterator;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.util.DataConverter;

import scala.Tuple2;

public class ResultMergeRemoteSparkWCompare extends ResultMergeMatrix implements PairFunction<Tuple2<MatrixIndexes,Tuple2<Iterable<MatrixBlock>,MatrixBlock>>, MatrixIndexes, MatrixBlock>
{
	private static final long serialVersionUID = -5970805069405942836L;
	
	public ResultMergeRemoteSparkWCompare(boolean accum) {
		_isAccum = accum;
	}
	
	@Override
	public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Tuple2<Iterable<MatrixBlock>, MatrixBlock>> arg)
		throws Exception 
	{
		MatrixIndexes ixin = arg._1();
		Iterator<MatrixBlock> din = arg._2()._1().iterator();
		MatrixBlock cin = arg._2()._2();
		
		//create compare array
		DenseBlock compare = DataConverter.convertToDenseBlock(cin, false);
		
		//merge all blocks into compare block
		MatrixBlock out = new MatrixBlock(cin);
		while( din.hasNext() ) {
			if( _isAccum )
				mergeWithoutComp(out, din.next(), compare, false);
			else
				mergeWithComp(out, din.next(), compare);
		}
		
		//create output tuple
		return new Tuple2<>(new MatrixIndexes(ixin), out);
	}

	@Override
	public MatrixObject executeSerialMerge() {
		throw new DMLRuntimeException("Unsupported operation.");
	}

	@Override
	public MatrixObject executeParallelMerge(int par) {
		throw new DMLRuntimeException("Unsupported operation.");
	}
}
