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

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

import org.apache.sysml.runtime.controlprogram.caching.CacheBlock;

/**
 * This class is base class for partitioned matrix/frame blocks, to be used
 * as broadcasts. Distributed tasks require block-partitioned broadcasts but a lazy partitioning per
 * task would create instance-local copies and hence replicate broadcast variables which are shared
 * by all tasks within an executor.  
 * 
 */
public abstract class PartitionedBlock implements Externalizable
{

	protected CacheBlock[] _partBlocks = null; 
	protected int _rlen = -1;
	protected int _clen = -1;
	protected int _brlen = -1;
	protected int _bclen = -1;
	protected int _offset = 0;
	
	public PartitionedBlock() {
		//do nothing (required for Externalizable)
	}
	
	
	public long getNumRows() {
		return _rlen;
	}
	
	public long getNumCols() {
		return _clen;
	}
	
	public long getNumRowsPerBlock() {
		return _brlen;
	}
	
	public long getNumColumnsPerBlock() {
		return _bclen;
	}
	
	/**
	 * 
	 * @return
	 */
	public int getNumRowBlocks() 
	{
		return (int)Math.ceil((double)_rlen/_brlen);
	}
	
	/**
	 * 
	 * @return
	 */
	public int getNumColumnBlocks() 
	{
		return (int)Math.ceil((double)_clen/_bclen);
	}
	
	/**
	 * Redirects the default java serialization via externalizable to our default 
	 * hadoop writable serialization for efficient broadcast deserialization. 
	 * 
	 * @param is
	 * @throws IOException
	 */
	public abstract void readExternal(ObjectInput is) throws IOException;
				
	/**
	 * Redirects the default java serialization via externalizable to our default 
	 * hadoop writable serialization for efficient broadcast serialization. 
	 * 
	 * @param is
	 * @throws IOException
	 */
	public abstract void writeExternal(ObjectOutput os)throws IOException; 
	
}
