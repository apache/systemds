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

import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;

import java.nio.ByteBuffer;
import java.util.BitSet;

public class GroupInfoMessage extends Message {

	private static final int LINK_PHASE_CHANGE_PRESENT = 0;
	private static final int ESTIMATED_ENTRY_INFORMATION_PRESENT = 1;

	private final int maximumCompactLinks;
	private final int minimumDenseLinks;
	private final int estimatedNumberOfEntries;
	private final int estimatedLengthOfEntryName;

	/* package */ GroupInfoMessage(ByteBuffer bb, BitSet messageFlags) {
		super(messageFlags);

		final byte version = bb.get();
		if (version != 0) {
			throw new HdfException("Unrecognized version " + version);
		}

		BitSet flags = BitSet.valueOf(new byte[] { bb.get() });

		if (flags.get(LINK_PHASE_CHANGE_PRESENT)) {
			maximumCompactLinks = Utils.readBytesAsUnsignedInt(bb, 2);
			minimumDenseLinks = Utils.readBytesAsUnsignedInt(bb, 2);
		} else {
			maximumCompactLinks = -1;
			minimumDenseLinks = -1;
		}

		if (flags.get(ESTIMATED_ENTRY_INFORMATION_PRESENT)) {
			estimatedNumberOfEntries = Utils.readBytesAsUnsignedInt(bb, 2);
			estimatedLengthOfEntryName = Utils.readBytesAsUnsignedInt(bb, 2);
		} else {
			estimatedNumberOfEntries = -1;
			estimatedLengthOfEntryName = -1;
		}
	}

	public int getMaximumCompactLinks() {
		return maximumCompactLinks;
	}

	public int getMinimumDenseLinks() {
		return minimumDenseLinks;
	}

	public int getEstimatedNumberOfEntries() {
		return estimatedNumberOfEntries;
	}

	public int getEstimatedLengthOfEntryName() {
		return estimatedLengthOfEntryName;
	}

}
