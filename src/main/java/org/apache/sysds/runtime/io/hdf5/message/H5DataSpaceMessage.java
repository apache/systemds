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
import java.util.Arrays;
import java.util.BitSet;
import java.util.stream.IntStream;

public class H5DataSpaceMessage extends H5Message {

	private boolean maxSizesPresent;
	private long totalLength;

	public H5DataSpaceMessage(H5RootObject rootObject, BitSet flags) {
		super(rootObject, flags);
	}

	public H5DataSpaceMessage(H5RootObject rootObject, BitSet flags, ByteBuffer bb) {
		super(rootObject, flags);
		rootObject.setDataSpaceVersion(bb.get());
		rootObject.setRank(bb.get());
		byte[] flagBits = new byte[1];
		bb.get(flagBits);
		BitSet maxFlags = BitSet.valueOf(flagBits);
		maxSizesPresent = maxFlags.get(0);

		// Skip 5 reserved bytes
		bb.position(bb.position() + 5);

		// Dimensions sizes
		if(rootObject.getRank() != 0) {
			int[] dimensions = new int[rootObject.getRank()];
			for(int i = 0; i < rootObject.getRank(); i++) {
				dimensions[i] = Utils.readBytesAsUnsignedInt(bb, rootObject.getSuperblock().sizeOfLengths);
			}
			rootObject.setDimensions(dimensions);
		}
		else {
			rootObject.setDimensions(new int[0]);
		}

		// Max dimension sizes
		if(maxSizesPresent) {
			int[] maxSizes = new int[rootObject.getRank()];
			for(int i = 0; i < rootObject.getRank(); i++) {
				maxSizes[i] = Utils.readBytesAsUnsignedInt(bb, rootObject.getSuperblock().sizeOfLengths);
			}
			rootObject.setMaxSizes(maxSizes);
		}
		else {
			rootObject.setMaxSizes(new int[0]);
		}

		// Calculate the total length by multiplying all dimensions
		totalLength = IntStream.of(rootObject.getLogicalDimensions()).mapToLong(Long::valueOf)
			.reduce(1, Math::multiplyExact);
		if(H5RootObject.HDF5_DEBUG) {
			System.out.println("[HDF5] Dataspace rank=" + rootObject.getRank() + " dims="
				+ Arrays.toString(rootObject.getLogicalDimensions()) + " => rows=" + rootObject.getRow()
				+ ", cols(flat)="
				+ rootObject.getCol());
		}

	}

	@Override
	public void toBuffer(H5BufferBuilder bb) {
		super.toBuffer(bb, H5Constants.DATA_SPACE_MESSAGE);
		bb.writeByte(rootObject.getDataSpaceVersion());
		bb.writeByte(rootObject.getRank());

		byte flag = 0;
		if(rootObject.getMaxSizes() != null && rootObject.getMaxSizes().length > 0) {
			flag = 1;
		}
		bb.writeByte(flag);

		// Skip 5 reserved bytes
		byte[] reserved = new byte[5];
		bb.writeBytes(reserved);

		// Dimensions sizes
		if(rootObject.getRank() != 0) {
			for(int i = 0; i < rootObject.getRank(); i++) {
				bb.write(rootObject.getLogicalDimensions()[i], rootObject.getSuperblock().sizeOfLengths);
			}
		}
		// Max dimension sizes
		if(flag == 1) {
			for(int i = 0; i < rootObject.getRank(); i++) {
				bb.write(rootObject.getMaxSizes()[i], rootObject.getSuperblock().sizeOfLengths);
			}
		}
	}

	public boolean isMaxSizesPresent() {
		return maxSizesPresent;
	}

	public long getTotalLength() {
		return totalLength;
	}
}
