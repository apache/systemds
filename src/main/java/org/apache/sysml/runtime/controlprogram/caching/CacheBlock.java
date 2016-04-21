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


/**
 * Interface for all blocks handled by lazy write buffer. This abstraction 
 * allows us to keep the buffer pool independent of matrix and frame blocks. 
 * 
 */
public interface CacheBlock extends Writable 
{
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
}
