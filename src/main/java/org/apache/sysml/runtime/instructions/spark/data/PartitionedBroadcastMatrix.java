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

package org.apache.sysml.runtime.instructions.spark.data;

import java.io.Serializable;

import org.apache.spark.broadcast.Broadcast;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

/**
 * This class is a wrapper around an array of broadcasts of partitioned matrix blocks,
 * which is required due to 2GB limitations of Spark's broadcast handling. Without this
 * partitioning of Broadcast<PartitionedMatrixBlock> into Broadcast<PartitionedMatrixBlock>[],
 * we got java.lang.IllegalArgumentException: Size exceeds Integer.MAX_VALUE issue.
 * Despite various jiras, this issue still showed up in Spark 1.4/1.5. 
 * 
 */
public class PartitionedBroadcastMatrix implements Serializable
{
	private static final long serialVersionUID = 1225135967889810877L;
	private static final long BROADCAST_PARTSIZE = 200L*1024*1024; //200M cells ~ 1.6GB 
	
	private Broadcast<PartitionedMatrixBlock>[] _pbc = null;
	
	public PartitionedBroadcastMatrix(Broadcast<PartitionedMatrixBlock>[] broadcasts)
	{
		_pbc = broadcasts;
	}
	
	public Broadcast<PartitionedMatrixBlock>[] getBroadcasts() {
		return _pbc;
	}
	
	/**
	 * 
	 * @return
	 */
	public int getNumRowBlocks() {
		return _pbc[0].value().getNumRowBlocks();
	}
	
	public int getNumColumnBlocks() {
		return _pbc[0].value().getNumColumnBlocks();
	}
	
	/**
	 * 
	 * @param rowIndex
	 * @param colIndex
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public MatrixBlock getMatrixBlock(int rowIndex, int colIndex) 
		throws DMLRuntimeException 
	{
		if( _pbc.length > 1 ) { 
			//compute partition index
			PartitionedMatrixBlock tmp = _pbc[0].value();
			int numPerPart = computeBlocksPerPartition(tmp.getNumRows(), tmp.getNumCols(), 
					tmp.getNumRowsPerBlock(), tmp.getNumColumnsPerBlock());
			int ix = (rowIndex-1)*tmp.getNumColumnBlocks()+(colIndex-1);
			int pix = ix / numPerPart;
			
			//get matrix block from partition
			return _pbc[pix].value().getMatrixBlock(rowIndex, colIndex);	
		}
		else { //single partition
			return _pbc[0].value().getMatrixBlock(rowIndex, colIndex);
		}
		
	}
	
	public MatrixBlock sliceOperations(long rl, long ru, long cl, long cu, MatrixBlock matrixBlock) 
		throws DMLRuntimeException 
	{
		MatrixBlock ret = null;
		
		for( Broadcast<PartitionedMatrixBlock> bc : _pbc ) {
			PartitionedMatrixBlock pm = bc.value();
			MatrixBlock tmp = pm.sliceOperations(rl, ru, cl, cu, new MatrixBlock());
			if( ret != null )
				ret.merge(tmp, false);
			else
				ret = tmp;
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @return
	 */
	public static int computeBlocksPerPartition(long rlen, long clen, long brlen, long bclen) {
		return (int) Math.floor( BROADCAST_PARTSIZE /  
				Math.min(rlen, brlen) / Math.min(clen, bclen));
	}
}
