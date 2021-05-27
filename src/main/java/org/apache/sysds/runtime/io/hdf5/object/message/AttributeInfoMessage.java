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

package org.apache.sysds.runtime.io.hdf5.object.message;

import org.apache.sysds.runtime.io.hdf5.Superblock;
import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;

import java.nio.ByteBuffer;
import java.util.BitSet;

public class AttributeInfoMessage extends Message {

	private static final int MAXIMUM_CREATION_INDEX_PRESENT = 0;
	private static final int ATTRIBUTE_CREATION_ORDER_PRESENT = 1;

	private final int maximumCreationIndex;
	private final long fractalHeapAddress;
	private final long attributeNameBTreeAddress;
	private final long attributeCreationOrderBTreeAddress;

	/* package */ AttributeInfoMessage(ByteBuffer bb, Superblock sb, BitSet messageFlags) {
		super(messageFlags);

		final byte version = bb.get();
		if (version != 0) {
			throw new HdfException("Unrecognized version " + version);
		}

		BitSet flags = BitSet.valueOf(new byte[] { bb.get() });

		if (flags.get(MAXIMUM_CREATION_INDEX_PRESENT)) {
			maximumCreationIndex = Utils.readBytesAsUnsignedInt(bb, 2);
		} else {
			maximumCreationIndex = -1;
		}

		fractalHeapAddress = Utils.readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());

		attributeNameBTreeAddress = Utils.readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());

		if (flags.get(ATTRIBUTE_CREATION_ORDER_PRESENT)) {
			attributeCreationOrderBTreeAddress = Utils.readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());
		} else {
			attributeCreationOrderBTreeAddress = -1;
		}
	}

	public int getMaximumCreationIndex() {
		return maximumCreationIndex;
	}

	public long getFractalHeapAddress() {
		return fractalHeapAddress;
	}

	public long getAttributeNameBTreeAddress() {
		return attributeNameBTreeAddress;
	}

	public long getAttributeCreationOrderBTreeAddress() {
		return attributeCreationOrderBTreeAddress;
	}

}
