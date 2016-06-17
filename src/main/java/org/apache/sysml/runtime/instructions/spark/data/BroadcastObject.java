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

import java.lang.ref.SoftReference;

import org.apache.spark.broadcast.Broadcast;

public class BroadcastObject extends LineageObject
{
	//soft reference storage for graceful cleanup in case of memory pressure
	protected SoftReference<PartitionedBroadcast> _bcHandle = null;
	
	public BroadcastObject( PartitionedBroadcast bvar, String varName )
	{
		_bcHandle = new SoftReference<PartitionedBroadcast>(bvar);
		_varName = varName;
	}
	
	/**
	 * 
	 * @return
	 */
	public PartitionedBroadcast getBroadcast()
	{
		return _bcHandle.get();
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isValid() 
	{
		//check for evicted soft reference
		PartitionedBroadcast pbm = _bcHandle.get();
		if( pbm == null )
			return false;
		
		//check for validity of individual broadcasts
		Broadcast<PartitionedBlock>[] tmp = pbm.getBroadcasts();
		for( Broadcast<PartitionedBlock> bc : tmp )
			if( !bc.isValid() )
				return false;		
		return true;
	}
}
