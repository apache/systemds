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

import org.apache.spark.broadcast.Broadcast;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;

/**
 * This class is a wrapper around an array of broadcasts of partitioned matrix blocks,
 * which is required due to 2GB limitations of Spark's broadcast handling. Without this
 * partitioning of Broadcast<PartitionedMatrixBlock> into Broadcast<PartitionedMatrixBlock>[],
 * we got java.lang.IllegalArgumentException: Size exceeds Integer.MAX_VALUE issue.
 * Despite various jiras, this issue still showed up in Spark 1.4/1.5. 
 * 
 */
public class PartitionedBroadcastFrame extends PartitionedBroadcast
{
	private static final long serialVersionUID = -1553711845891805319L;

	private Broadcast<PartitionedFrameBlock>[] _pbc = null;
	
	public PartitionedBroadcastFrame(Broadcast<PartitionedFrameBlock>[] broadcasts)
	{
		_pbc = broadcasts;
	}
	
	public Broadcast<PartitionedFrameBlock>[] getBroadcasts() {
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
	public FrameBlock getFrameBlock(int rowIndex, int colIndex) 
		throws DMLRuntimeException 
	{
		if( _pbc.length > 1 ) { 
			//compute partition index
			PartitionedFrameBlock tmp = _pbc[0].value();
			int numPerPart = computeBlocksPerPartition(tmp.getNumRows(), tmp.getNumCols(), 
					tmp.getNumRowsPerBlock(), tmp.getNumColumnsPerBlock());
			int ix = (rowIndex-1)*tmp.getNumColumnBlocks()+(colIndex-1);
			int pix = ix / numPerPart;
			
			//get frame block from partition
			return _pbc[pix].value().getFrameBlock(rowIndex, colIndex);	
		}
		else { //single partition
			return _pbc[0].value().getFrameBlock(rowIndex, colIndex);
		}
		
	}
	
	public FrameBlock sliceOperations(long rl, long ru, long cl, long cu, FrameBlock frameBlock) 
		throws DMLRuntimeException 
	{
		FrameBlock ret = null;
		
		for( Broadcast<PartitionedFrameBlock> bc : _pbc ) {
			PartitionedFrameBlock pm = bc.value();
			FrameBlock tmp = pm.sliceOperations(rl, ru, cl, cu, new FrameBlock());
			if( ret != null )
				ret.merge(tmp);
			else
				ret = tmp;
		}
		
		return ret;
	}
}
