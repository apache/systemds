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

import java.nio.ByteBuffer;
import java.util.BitSet;

public class LinkInfoMessage extends Message {

	private static final int CREATION_ORDER_TRACKED = 0;
	private static final int CREATION_ORDER_INDEXED = 1;

	private final byte version;
	private final long maximumCreationIndex;
	private final long fractalHeapAddress;
	private final long bTreeNameIndexAddress;
	private final long bTreeCreationOrderIndexAddress;
	private final BitSet flags;

	 LinkInfoMessage(ByteBuffer bb, Superblock sb, BitSet messageFlags) {
		super(messageFlags);

		version = bb.get();
		flags = BitSet.valueOf(new byte[] { bb.get() });

		if (flags.get(CREATION_ORDER_TRACKED)) {
			maximumCreationIndex = Utils.readBytesAsUnsignedLong(bb, 8);
		} else {
			maximumCreationIndex = -1;
		}

		fractalHeapAddress = Utils.readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());

		bTreeNameIndexAddress = Utils.readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());

		if (flags.get(CREATION_ORDER_INDEXED)) {
			bTreeCreationOrderIndexAddress = Utils.readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());
		} else {
			bTreeCreationOrderIndexAddress = -1;
		}
	}

	public int getVersion() {
		return version;
	}

	public long getMaximumCreationIndex() {
		return maximumCreationIndex;
	}

	public long getFractalHeapAddress() {
		return fractalHeapAddress;
	}

	public long getBTreeNameIndexAddress() {
		return bTreeNameIndexAddress;
	}

	public long getBTreeCreationOrderIndexAddress() {
		return bTreeCreationOrderIndexAddress;
	}

	public boolean isLinkCreationOrderTracked() {
		return flags.get(CREATION_ORDER_TRACKED);
	}
}
