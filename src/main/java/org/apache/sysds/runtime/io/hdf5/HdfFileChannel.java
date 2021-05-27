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


package org.apache.sysds.runtime.io.hdf5;

import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileChannel.MapMode;
import org.apache.sysds.runtime.io.hdf5.Superblock.SuperblockV0V1;
import org.apache.sysds.runtime.io.hdf5.Superblock.SuperblockV2V3;

import static java.nio.ByteOrder.LITTLE_ENDIAN;

public class HdfFileChannel {

	private final FileChannel fc;
	private final Superblock sb;

	public HdfFileChannel(FileChannel fileChannel, Superblock superblock) {
		this.fc = fileChannel;
		this.sb = superblock;
	}

	public ByteBuffer readBufferFromAddress(long address, int length) {
		ByteBuffer bb = ByteBuffer.allocate(length);
		try {
			fc.read(bb, address + sb.getBaseAddressByte());
		}
		catch(IOException e) {
			throw new HdfException("Failed to read from file at address '" + address + "' (raw address '" + address + sb
				.getBaseAddressByte() + "'", e);
		}
		bb.order(LITTLE_ENDIAN);
		bb.rewind();
		return bb;
	}

	public ByteBuffer map(long address, long length) {
		return mapNoOffset(address + sb.getBaseAddressByte(), length);
	}

	public ByteBuffer mapNoOffset(long address, long length) {
		try {
			return fc.map(MapMode.READ_ONLY, address, length);
		}
		catch(IOException e) {
			throw new HdfException("Failed to map buffer at address '" + address + "' of length '" + length + "'", e);
		}
	}

	public long getUserBlockSize() {
		return sb.getBaseAddressByte();
	}

	public Superblock getSuperblock() {
		return sb;
	}

	public FileChannel getFileChannel() {
		return fc;
	}

	public int getSizeOfOffsets() {
		return sb.getSizeOfOffsets();
	}

	public int getSizeOfLengths() {
		return sb.getSizeOfLengths();
	}

	public final void close() {
		try {
			fc.close();
		}
		catch(IOException e) {
			throw new HdfException("Failed closing HDF5 file", e);
		}
	}

	public long size() {
		try {
			return fc.size();
		}
		catch(IOException e) {
			throw new HdfException("Failed to get size of HDF5 file", e);
		}
	}

	public long getRootGroupAddress(){
		if(this.sb instanceof SuperblockV0V1 ){
			return ((SuperblockV0V1)this.sb).getRootGroupSymbolTableAddress();
		}
		else if(this.sb instanceof SuperblockV2V3 ){
			return ((SuperblockV2V3)this.sb).getRootGroupObjectHeaderAddress();
		}
		else
			throw new HdfException("Failed to to find address");
	};

}
