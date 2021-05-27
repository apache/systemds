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

package org.apache.sysds.runtime.io.hdf5.btree.record;

import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import java.nio.ByteBuffer;
import java.util.BitSet;

public class AttributeNameForIndexedAttributesRecord extends BTreeRecord {

	private final ByteBuffer heapId;
	private final BitSet flags;
	private final long creationOrder;
	private final long hash;

	public AttributeNameForIndexedAttributesRecord(ByteBuffer bb) {
		if (bb.remaining() != 17) {
			throw new HdfException(
					"Invalid length buffer for AttributeNameForIndexedAttributesRecord. remaining bytes = "
							+ bb.remaining());
		}

		heapId = Utils.createSubBuffer(bb, 8);
		flags = BitSet.valueOf(new byte[] { bb.get() });
		creationOrder = Utils.readBytesAsUnsignedLong(bb, 4);
		hash = Utils.readBytesAsUnsignedLong(bb, 4);
	}

	public ByteBuffer getHeapId() {
		return heapId;
	}

	public BitSet getFlags() {
		return flags;
	}

	public long getCreationOrder() {
		return creationOrder;
	}

	public long getHash() {
		return hash;
	}

}
