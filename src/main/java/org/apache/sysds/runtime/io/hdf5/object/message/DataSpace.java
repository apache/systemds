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

import org.apache.sysds.runtime.io.hdf5.BufferBuilder;
import org.apache.sysds.runtime.io.hdf5.Superblock;
import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.commons.lang3.ArrayUtils;

import java.nio.ByteBuffer;
import java.util.BitSet;
import java.util.stream.IntStream;

public class DataSpace {

	private byte version;
	private boolean maxSizesPresent;
	private int numberOfDimensions;
	private int[] dimensions;
	private int[] maxSizes;
	private byte type;
	private long totalLength;
	private int sizeOfLengths;

	public DataSpace(byte version, int numberOfDimensions, int[] dimensions, int[] maxSizes,
		int sizeOfLengths) {
		this.version = version;
		this.numberOfDimensions = numberOfDimensions;
		this.dimensions = dimensions;
		this.maxSizes = maxSizes;
		this.sizeOfLengths = sizeOfLengths;
	}

	private DataSpace(ByteBuffer bb, Superblock sb) {

		version = bb.get();
		int numberOfDimensions = bb.get();
		byte[] flagBits = new byte[1];
		bb.get(flagBits);
		BitSet flags = BitSet.valueOf(flagBits);
		maxSizesPresent = flags.get(0);

		if (version == 1) {
			// Skip 5 reserved bytes
			bb.position(bb.position() + 5);
			type = -1;
		} else if (version == 2) {
			type = bb.get();
		} else {
			throw new HdfException("Unrecognized version = " + version);
		}
		int aaa = sb.getSizeOfLengths();
		// Dimensions sizes
		if (numberOfDimensions != 0) {
			dimensions = new int[numberOfDimensions];
			for (int i = 0; i < numberOfDimensions; i++) {
				dimensions[i] = Utils.readBytesAsUnsignedInt(bb, sb.getSizeOfLengths());
			}
		} else {
			dimensions = new int[0];
		}

		// Max dimension sizes
		if (maxSizesPresent) {
			maxSizes = new int[numberOfDimensions];
			for (int i = 0; i < numberOfDimensions; i++) {
				maxSizes[i] = Utils.readBytesAsUnsignedInt(bb, sb.getSizeOfLengths());
			}
		} else {
			maxSizes = new int[0];
		}

		// If type == 2 then it's an empty dataset and totalLength should be 0
		if (type == 2) {
			totalLength = 0;
		} else {
			// Calculate the total length by multiplying all dimensions
			totalLength = IntStream.of(dimensions)
					.mapToLong(Long::valueOf) // Convert to long to avoid int overflow
					.reduce(1, Math::multiplyExact);
		}

		// Permutation indices - Note never implemented in HDF library!
	}

	public static DataSpace readDataSpace(ByteBuffer bb, Superblock sb) {
		return new DataSpace(bb, sb);
	}

	public BufferBuilder toBuffer() {
		BufferBuilder header = new BufferBuilder();
		return toBuffer(header);
	}
	public BufferBuilder toBuffer(BufferBuilder header) {
		header.writeByte(version);
		header.writeByte(numberOfDimensions);

		byte flag=0;
		if(maxSizes!=null && maxSizes.length>0){
			flag = 1;
		}
		header.writeByte(flag);

		if(version == 1) {
			// Skip 5 reserved bytes
			byte[] reserved=new byte[5];
			header.writeBytes(reserved);
			type = -1;
		}
		else if(version == 2) {
			header.writeByte(type);
		}
		else {
			throw new HdfException("Unrecognized version = " + version);
		}

		// Dimensions sizes
		if(numberOfDimensions!=0){
			for (int i = 0; i < numberOfDimensions; i++) {
				if(sizeOfLengths ==2){
					header.writeShort((short) dimensions[i]);
				} else if(sizeOfLengths ==4){
					header.writeInt( dimensions[i]);
				} else if(sizeOfLengths ==8){
					header.writeLong(dimensions[i]);
				}
			}
		}
		// Max dimension sizes
		if (flag ==1) {
			for (int i = 0; i < numberOfDimensions; i++) {
				if(sizeOfLengths ==2){
					header.writeShort((short) dimensions[i]);
				} else if(sizeOfLengths ==4){
					header.writeInt( dimensions[i]);
				} else if(sizeOfLengths ==8){
					header.writeLong(dimensions[i]);
				}
			}
		}
		return header;
	}

	public byte getVersion() {
		return version;
	}

	public void setVersion(byte version) {
		this.version = version;
	}

	public boolean isMaxSizesPresent() {
		return maxSizesPresent;
	}

	public void setMaxSizesPresent(boolean maxSizesPresent) {
		this.maxSizesPresent = maxSizesPresent;
	}

	public int[] getDimensions() {
		return dimensions;
	}

	public void setDimensions(int[] dimensions) {
		this.dimensions = dimensions;
	}

	public int[] getMaxSizes() {
		return maxSizes;
	}

	public void setMaxSizes(int[] maxSizes) {
		this.maxSizes = maxSizes;
	}

	public byte getType() {
		return type;
	}

	public void setType(byte type) {
		this.type = type;
	}

	public long getTotalLength() {
		return totalLength;
	}

	public void setTotalLength(long totalLength) {
		this.totalLength = totalLength;
	}

	public int getNumberOfDimensions() {
		return numberOfDimensions;
	}

	public void setNumberOfDimensions(int numberOfDimensions) {
		this.numberOfDimensions = numberOfDimensions;
	}

	public void setSizeOfLengths(int sizeOfLengths) {
		this.sizeOfLengths = sizeOfLengths;
	}
}
