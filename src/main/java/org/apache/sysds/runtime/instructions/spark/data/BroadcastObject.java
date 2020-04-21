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

package org.apache.sysds.runtime.instructions.spark.data;

import java.lang.ref.SoftReference;

import org.apache.spark.broadcast.Broadcast;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;

public class BroadcastObject<T extends CacheBlock> extends LineageObject {
	//soft reference storage for graceful cleanup in case of memory pressure
	private SoftReference<PartitionedBroadcast<T>> _pbcRef; // partitioned broadcast object reference
	private SoftReference<Broadcast<T>> _npbcRef; // non partitioned broadcast object reference

	private long _pbcSize; // partitioned broadcast size
	private long _npbcSize; // non-partitioned broadcast size

	public BroadcastObject() {
		super();
	}

	public void setNonPartitionedBroadcast(Broadcast<T> bvar, long size) {
		_npbcRef = new SoftReference<>(bvar);
		_npbcSize = size;
	}

	public void setPartitionedBroadcast(PartitionedBroadcast<T> bvar, long size) {
		_pbcRef = new SoftReference<>(bvar);
		_pbcSize = size;
	}

	@SuppressWarnings("rawtypes")
	public PartitionedBroadcast getPartitionedBroadcast() {
		return _pbcRef.get();
	}

	public Broadcast<T> getNonPartitionedBroadcast() {
		return _npbcRef.get();
	}

	public long getPartitionedBroadcastSize() {
		return _pbcSize;
	}

	public long getNonPartitionedBroadcastSize() {
		return _npbcSize;
	}

	public boolean isPartitionedBroadcastValid() {
		return _pbcRef != null && checkPartitionedBroadcastValid();
	}

	public boolean isNonPartitionedBroadcastValid() {
		return _npbcRef != null && checkNonPartitionedBroadcastValid();
	}

	private boolean checkNonPartitionedBroadcastValid() {
		return _npbcRef.get() != null;
	}

	private boolean checkPartitionedBroadcastValid() {
		//check for evicted soft reference
		PartitionedBroadcast<T> pbm = _pbcRef.get();
		if (pbm == null)
			return false;

		//check for validity of individual broadcasts
		Broadcast<PartitionedBlock<T>>[] tmp = pbm.getBroadcasts();
		for (Broadcast<PartitionedBlock<T>> bc : tmp)
			if (!bc.isValid())
				return false;
		return true;
	}
}
