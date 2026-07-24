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


package org.apache.sysds.runtime.io.hdf5.message;

import org.apache.sysds.runtime.io.hdf5.H5BufferBuilder;
import org.apache.sysds.runtime.io.hdf5.H5Constants;
import org.apache.sysds.runtime.io.hdf5.H5RootObject;
import org.apache.sysds.runtime.io.hdf5.Utils;

import java.nio.ByteBuffer;
import java.util.BitSet;

public class H5DataLayoutMessage extends H5Message {

	public static final byte LAYOUT_CLASS_COMPACT = 0;
	public static final byte LAYOUT_CLASS_CONTIGUOUS = 1;
	public static final byte LAYOUT_CLASS_CHUNKED = 2;
	public static final byte LAYOUT_CLASS_VIRTUAL = 3;

	private final long address;
	private final long size;
	private final byte layoutClass;
	private final byte layoutVersion;

	public H5DataLayoutMessage(H5RootObject rootObject, BitSet flags, ByteBuffer bb) {
		super(rootObject, flags);
		rootObject.setDataLayoutVersion(bb.get());
		layoutVersion = rootObject.getDataLayoutVersion();
		rootObject.setDataLayoutClass(bb.get());
		layoutClass = rootObject.getDataLayoutClass();
		this.address = Utils.readBytesAsUnsignedLong(bb, rootObject.getSuperblock().sizeOfOffsets);
		this.size = Utils.readBytesAsUnsignedLong(bb, rootObject.getSuperblock().sizeOfLengths);
		if(H5RootObject.HDF5_DEBUG) {
			System.out.println("[HDF5] Data layout (version=" + layoutVersion + ", class=" + layoutClass + ") address="
				+ address + " size=" + size);
		}
	}

	public H5DataLayoutMessage(H5RootObject rootObject, BitSet flags, long address, long size) {
		super(rootObject, flags);
		this.address = address;
		this.size = size;
		this.layoutVersion = rootObject.getDataLayoutVersion();
		this.layoutClass = rootObject.getDataLayoutClass();
	}

	@Override
	public void toBuffer(H5BufferBuilder bb) {
		super.toBuffer(bb, H5Constants.DATA_LAYOUT_MESSAGE);

		// Version
		bb.writeByte(rootObject.getDataLayoutVersion());

		// Layout Class
		bb.writeByte(rootObject.getDataLayoutClass());

		// Address
		bb.writeLong(address);

		// Size
		bb.writeLong(size);

		byte[] reserved = new byte[6];
		bb.writeBytes(reserved);
	}

	public long getAddress() {
		return address;
	}

	public long getSize() {
		return size;
	}
	
	public byte getLayoutClass() {
		return layoutClass;
	}

	public byte getLayoutVersion() {
		return layoutVersion;
	}
}
