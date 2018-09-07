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
import java.util.ArrayList;

import org.apache.spark.broadcast.Broadcast;
import org.apache.sysml.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysml.runtime.controlprogram.caching.CacheBlockFactory;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.util.IndexRange;

/**
 * This class is a wrapper around an array of broadcasts of partitioned matrix/frame blocks,
 * which is required due to 2GB limitations of Spark's broadcast handling. Without this
 * partitioning of {@code Broadcast<PartitionedBlock>} into {@code Broadcast<PartitionedBlock>[]},
 * we got java.lang.IllegalArgumentException: Size exceeds Integer.MAX_VALUE issue.
 * Despite various jiras, this issue still showed up in Spark 2.1. 
 * 
 */
public class PartitionedBroadcast<T extends CacheBlock> implements Serializable
{
	private static final long serialVersionUID = 7041959166079438401L;

	//note: that each block (max 240 * 1024) also requires some header space
	protected static final long BROADCAST_PARTSIZE = 240L*1024*1024; //250M cells > 1.875GB 
	
	private Broadcast<PartitionedBlock<T>>[] _pbc = null;
	private MatrixCharacteristics _mc;
	
	public PartitionedBroadcast() {
		//do nothing (required for Externalizable)
	}
	
	public PartitionedBroadcast(Broadcast<PartitionedBlock<T>>[] broadcasts, MatrixCharacteristics mc) {
		_pbc = broadcasts;
		_mc = mc;
	}
	
	public Broadcast<PartitionedBlock<T>>[] getBroadcasts() {
		return _pbc;
	}
	
	public long getNumRows() {
		return _mc.getRows();
	}
	
	public long getNumCols() {
		return _mc.getCols();
	}

	public int getNumRowBlocks() {
		return (int)_mc.getNumRowBlocks();
	}
	
	public int getNumColumnBlocks() {
		return (int)_mc.getNumColBlocks();
	}

	public static int computeBlocksPerPartition(long rlen, long clen, long brlen, long bclen) {
		return (int) Math.floor( BROADCAST_PARTSIZE /
			Math.min(rlen, brlen) / Math.min(clen, bclen));
	}

	public T getBlock(int rowIndex, int colIndex) {
		int pix = 0;
		if( _pbc.length > 1 ) { //compute partition index
			int numPerPart = computeBlocksPerPartition(_mc.getRows(),
				_mc.getCols(),_mc.getRowsPerBlock(), _mc.getColsPerBlock());
			int ix = (rowIndex-1)*getNumColumnBlocks()+(colIndex-1);
			pix = ix / numPerPart;
		}
		
		return _pbc[pix].value().getBlock(rowIndex, colIndex);
	}
	
	/**
	 * Utility for slice operations over partitioned matrices, where the index range can cover
	 * multiple blocks. The result is always a single result matrix block. All semantics are 
	 * equivalent to the core matrix block slice operations. 
	 * 
	 * @param rl row lower bound
	 * @param ru row upper bound
	 * @param cl column lower bound
	 * @param cu column upper bound
	 * @param block block object
	 * @return block object
	 */
	@SuppressWarnings("unchecked")
	public T slice(long rl, long ru, long cl, long cu, T block) {
		int lrl = (int) rl;
		int lru = (int) ru;
		int lcl = (int) cl;
		int lcu = (int) cu;
		
		ArrayList<Pair<?, ?>> allBlks = (ArrayList<Pair<?, ?>>) CacheBlockFactory.getPairList(block);
		int start_iix = (lrl-1)/_mc.getRowsPerBlock()+1;
		int end_iix = (lru-1)/_mc.getRowsPerBlock()+1;
		int start_jix = (lcl-1)/_mc.getColsPerBlock()+1;
		int end_jix = (lcu-1)/_mc.getColsPerBlock()+1;
		
		for( int iix = start_iix; iix <= end_iix; iix++ )
			for(int jix = start_jix; jix <= end_jix; jix++) {
				IndexRange ixrange = new IndexRange(rl, ru, cl, cu);
				allBlks.addAll(OperationsOnMatrixValues.performSlice(
					ixrange, _mc.getRowsPerBlock(), _mc.getColsPerBlock(), iix, jix, getBlock(iix, jix)));
			}
		
		T ret = (T) allBlks.get(0).getValue();
		for(int i=1; i<allBlks.size(); i++)
			ret.merge((T)allBlks.get(i).getValue(), false);
		return ret;
	}
	
	/**
	 * This method cleanups all underlying broadcasts of a partitioned broadcast,
	 * by forward the calls to SparkExecutionContext.cleanupBroadcastVariable.
	 */
	public void destroy() {
		for( Broadcast<PartitionedBlock<T>> bvar : _pbc )
			SparkExecutionContext.cleanupBroadcastVariable(bvar);
	}
}
