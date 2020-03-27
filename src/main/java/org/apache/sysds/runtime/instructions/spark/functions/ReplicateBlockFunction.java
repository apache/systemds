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

import java.util.ArrayList;
import java.util.Iterator;

import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;

import scala.Tuple2;

public class ReplicateBlockFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
{
	private static final long serialVersionUID = -1184696764516975609L;
	
	private final long _len;
	private final long _blen;
	private final boolean _left;
	private final boolean _deep;
	
	public ReplicateBlockFunction(long len, long blen, boolean left) {
		//by default: shallow copy of blocks
		this(len, blen, left, false);
	}
	
	public ReplicateBlockFunction(long len, long blen, boolean left, boolean deep) {
		_len = len;
		_blen = blen;
		_left = left;
		_deep = deep;
	}
	
	@Override
	public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
		throws Exception 
	{
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<>();
		MatrixIndexes ixIn = arg0._1();
		MatrixBlock blkIn = arg0._2();
		
		long numBlocks = (long) Math.ceil((double)_len/_blen); 
		
		if( _left ) //LHS MATRIX
		{
			//replicate wrt # column blocks in RHS
			long i = ixIn.getRowIndex();
			for( long j=1; j<=numBlocks; j++ ) {
				MatrixIndexes tmpix = new MatrixIndexes(i, j);
				MatrixBlock tmpblk = _deep ? new MatrixBlock(blkIn) : blkIn;
				ret.add( new Tuple2<>(tmpix, tmpblk) );
			}
		} 
		else // RHS MATRIX
		{
			//replicate wrt # row blocks in LHS
			long j = ixIn.getColumnIndex();
			for( long i=1; i<=numBlocks; i++ ) {
				MatrixIndexes tmpix = new MatrixIndexes(i, j);
				MatrixBlock tmpblk = _deep ? new MatrixBlock(blkIn) : blkIn;
				ret.add( new Tuple2<>(tmpix, tmpblk) );
			}
		}
		
		//output list of new tuples
		return ret.iterator();
	}
}
