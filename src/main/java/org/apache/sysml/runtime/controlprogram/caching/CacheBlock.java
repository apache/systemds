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

package org.apache.sysml.runtime.controlprogram.caching;

import org.apache.hadoop.io.Writable;
import org.apache.sysml.runtime.DMLRuntimeException;


/**
 * Interface for all blocks handled by lazy write buffer. This abstraction 
 * allows us to keep the buffer pool independent of matrix and frame blocks. 
 * 
 */
public interface CacheBlock extends Writable 
{
	/**
	 * 
	 * @return
	 */
	public int getNumRows();
	
	/**
	 * 
	 * @return
	 */
	public int getNumColumns();
	
	/**
	 * Get the in-memory size in bytes of the cache block.
	 * @return
	 */
	public long getInMemorySize();
	
	/**
	 * Get the exact serialized size in bytes of the cache block.
	 * @return
	 */
	public long getExactSerializedSize();

	/**
	 * Indicates if the cache block is subject to shallow serialized,
	 * which is generally true if in-memory size and serialized size
	 * are almost identical allowing to avoid unnecessary deep serialize. 
	 * 
	 * @return
	 */
	public boolean isShallowSerialize();
	
	/**
	 * Free unnecessarily allocated empty block.
	 */
	public void compactEmptyBlock();
	
	/**
	 * Slice a sub block out of the current block and write into the given output block.
	 * This method returns the passed instance if not null.
	 * 
	 * @param rl
	 * @param ru
	 * @param cl
	 * @param cu
	 * @param block
	 * @return
	 * @throws DMLRuntimeException
	 */
	public CacheBlock sliceOperations(int rl, int ru, int cl, int cu, CacheBlock block) 
		throws DMLRuntimeException;
	
	/**
	 * Merge the given block into the current block. Both blocks needs to be of equal 
	 * dimensions and contain disjoint non-zero cells.
	 * 
	 * @param that
	 * @param appendOnly
	 * @throws DMLRuntimeException
	 */
	public void merge(CacheBlock that, boolean appendOnly) 
		throws DMLRuntimeException;
}
