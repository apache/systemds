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

package org.apache.sysds.runtime.controlprogram.caching;

import org.apache.hadoop.io.Writable;


/**
 * Interface for all blocks handled by lazy write buffer. This abstraction 
 * allows us to keep the buffer pool independent of matrix and frame blocks. 
 * 
 */
public interface CacheBlock extends Writable 
{

	public int getNumRows();

	public int getNumColumns();
	
	/**
	 * Get the in-memory size in bytes of the cache block.
	 * 
	 * @return in-memory size in bytes of cache block
	 */
	public long getInMemorySize();
	
	/**
	 * Get the exact serialized size in bytes of the cache block.
	 * 
	 * @return exact serialized size in bytes of cache block
	 */
	public long getExactSerializedSize();

	/**
	 * Indicates if the cache block is subject to shallow serialized,
	 * which is generally true if in-memory size and serialized size
	 * are almost identical allowing to avoid unnecessary deep serialize. 
	 * 
	 * @return true if shallow serialized
	 */
	public boolean isShallowSerialize();
	
	/**
	 * Indicates if the cache block is subject to shallow serialized,
	 * which is generally true if in-memory size and serialized size
	 * are almost identical allowing to avoid unnecessary deep serialize.
	 * 
	 * @param inclConvert if true report blocks as shallow serialize that are
	 * currently not amenable but can be brought into an amenable form
	 * via {@link #toShallowSerializeBlock() toShallowSerializeBlock}.
	 * 
	 * @return true if shallow serialized
	 */
	public boolean isShallowSerialize(boolean inclConvert);
	
	/**
	 * Converts a cache block that is not shallow serializable into
	 * a form that is shallow serializable. This methods has no affect
	 * if the given cache block is not amenable.
	 */
	public void toShallowSerializeBlock();
	
	/**
	 * Free unnecessarily allocated empty block.
	 */
	public void compactEmptyBlock();
	
	/**
	 * Slice a sub block out of the current block and write into the given output block.
	 * This method returns the passed instance if not null.
	 * 
	 * @param rl row lower
	 * @param ru row upper
	 * @param cl column lower
	 * @param cu column upper
	 * @param block cache block
	 * @return sub-block of cache block
	 */
	public CacheBlock slice(int rl, int ru, int cl, int cu, CacheBlock block);
	
	/**
	 * Merge the given block into the current block. Both blocks needs to be of equal 
	 * dimensions and contain disjoint non-zero cells.
	 * 
	 * @param that cache block
	 * @param appendOnly ?
	 */
	public void merge(CacheBlock that, boolean appendOnly);
}
