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
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.BitSet;

public class ObjectModificationTimeMessage extends Message {

	private final long unixEpocSecond;

	ObjectModificationTimeMessage(ByteBuffer bb, BitSet flags) {
		super(flags);

		final byte version = bb.get();
		if (version != 1) {
			throw new HdfException("Unrecognized version " + version);
		}

		// Skip 3 unused bytes
		bb.position(bb.position() + 3);

		// Convert to unsigned long
		unixEpocSecond = Utils.readBytesAsUnsignedLong(bb, 4);
	}

	public LocalDateTime getModifiedTime() {
		return LocalDateTime.ofEpochSecond(unixEpocSecond, 0, ZoneOffset.UTC);
	}

	public long getUnixEpocSecond() {
		return unixEpocSecond;
	}

}
