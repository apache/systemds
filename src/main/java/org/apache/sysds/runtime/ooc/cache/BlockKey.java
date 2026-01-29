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

package org.apache.sysds.runtime.ooc.cache;

import org.jetbrains.annotations.NotNull;

public class BlockKey implements Comparable<BlockKey> {
	private final long _streamId;
	private final long _sequenceNumber;

	public BlockKey(long streamId, long sequenceNumber) {
		this._streamId = streamId;
		this._sequenceNumber = sequenceNumber;
	}

	public long getStreamId() {
		return _streamId;
	}

	public long getSequenceNumber() {
		return _sequenceNumber;
	}

	@Override
	public int compareTo(@NotNull BlockKey blockKey) {
		int cmp = Long.compare(_streamId, blockKey._streamId);
		if (cmp != 0)
			return cmp;
		return Long.compare(_sequenceNumber, blockKey._sequenceNumber);
	}

	@Override
	public int hashCode() {
		return 31 * Long.hashCode(_streamId) + Long.hashCode(_sequenceNumber);
	}

	@Override
	public boolean equals(Object obj) {
		return obj instanceof BlockKey && ((BlockKey)obj)._streamId == _streamId && ((BlockKey)obj)._sequenceNumber == _sequenceNumber;
	}

	@Override
	public String toString() {
		return "BlockKey(" + _streamId + ", " + _sequenceNumber + ")";
	}

	public String toFileKey() {
		return _streamId + "_" + _sequenceNumber;
	}
}
