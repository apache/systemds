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

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

import static java.nio.ByteOrder.LITTLE_ENDIAN;

public class H5RootObject {

	protected BufferedInputStream bufferedInputStream;
	protected BufferedOutputStream bufferedOutputStream;
	protected H5Superblock superblock;
	protected int rank;
	protected long row;
	protected long col;
	protected int[] dimensions;
	protected long maxRow;
	protected long maxCol;
	protected int[] maxSizes;
	protected String datasetName;
	public H5BufferBuilder bufferBuilder;

	protected byte dataSpaceVersion = 1;
	protected byte objectHeaderVersion = 1;
	protected byte localHeapVersion = 0;
	protected byte fillValueVersion = 2;
	protected byte dataLayoutVersion = 3;
	protected byte objectModificationTimeVersion = 1;
	protected byte groupSymbolTableNodeVersion = 1;

	protected byte dataLayoutClass = 1;

	public ByteBuffer readBufferFromAddress(long address, int length) {
		ByteBuffer bb = ByteBuffer.allocate(length);
		try {
			byte[] b = new byte[length];
			bufferedInputStream.reset();
			bufferedInputStream.skip(address);
			bufferedInputStream.read(b);
			bb.put(b);
		}
		catch(IOException e) {
			throw new H5RuntimeException(e);
		}
		bb.order(LITTLE_ENDIAN);
		bb.rewind();
		return bb;
	}

	public ByteBuffer readBufferFromAddressNoOrder(long address, int length) {
		ByteBuffer bb = ByteBuffer.allocate(length);
		try {
			byte[] b = new byte[length];
			bufferedInputStream.reset();
			bufferedInputStream.skip(address);
			bufferedInputStream.read(b);
			bb.put(b);
		}
		catch(IOException e) {
			throw new H5RuntimeException(e);
		}
		bb.rewind();
		return bb;
	}

	public BufferedInputStream getBufferedInputStream() {
		return bufferedInputStream;
	}

	public void setBufferedInputStream(BufferedInputStream bufferedInputStream) {
		this.bufferedInputStream = bufferedInputStream;
	}

	public BufferedOutputStream getBufferedOutputStream() {
		return bufferedOutputStream;
	}

	public void setBufferedOutputStream(BufferedOutputStream bufferedOutputStream) {
		this.bufferedOutputStream = bufferedOutputStream;
	}

	public H5Superblock getSuperblock() {
		return superblock;
	}

	public void setSuperblock(H5Superblock superblock) {
		this.superblock = superblock;
	}

	public long getRow() {
		return row;
	}

	public void setRow(long row) {
		this.row = row;
		this.dimensions[0] = (int) row;
	}

	public long getCol() {
		return col;
	}

	public void setCol(long col) {
		this.col = col;
		this.dimensions[1] = (int) col;
	}

	public int getRank() {
		return rank;
	}

	public void setRank(int rank) {
		this.rank = rank;
		this.dimensions = new int[rank];
		this.maxSizes = new int[rank];
	}

	public long getMaxRow() {
		return maxRow;
	}

	public void setMaxRow(long maxRow) {
		this.maxRow = maxRow;
		this.maxSizes[0] = (int) maxRow;
	}

	public long getMaxCol() {
		return maxCol;
	}

	public void setMaxCol(long maxCol) {
		this.maxCol = maxCol;
		this.maxSizes[1] = (int) maxCol;
	}

	public String getDatasetName() {
		return datasetName;
	}

	public void setDatasetName(String datasetName) {
		this.datasetName = datasetName;
	}

	public int[] getDimensions() {
		return dimensions;
	}

	public int[] getMaxSizes() {
		return maxSizes;
	}

	public byte getDataSpaceVersion() {
		return dataSpaceVersion;
	}

	public void setDataSpaceVersion(byte dataSpaceVersion) {
		this.dataSpaceVersion = dataSpaceVersion;
	}

	public void setDimensions(int[] dimensions) {
		this.dimensions = dimensions;
		this.row = dimensions[0];
		this.col = dimensions[1];
	}

	public void setMaxSizes(int[] maxSizes) {
		this.maxSizes = maxSizes;
		this.maxRow = maxSizes[0];
		this.maxCol = maxSizes[1];
	}

	public byte getObjectHeaderVersion() {
		return objectHeaderVersion;
	}

	public void setObjectHeaderVersion(byte objectHeaderVersion) {
		this.objectHeaderVersion = objectHeaderVersion;
	}

	public byte getLocalHeapVersion() {
		return localHeapVersion;
	}

	public void setLocalHeapVersion(byte localHeapVersion) {
		this.localHeapVersion = localHeapVersion;
	}

	public byte getFillValueVersion() {
		return fillValueVersion;
	}

	public void setFillValueVersion(byte fillValueVersion) {
		this.fillValueVersion = fillValueVersion;
	}

	public byte getDataLayoutVersion() {
		return dataLayoutVersion;
	}

	public void setDataLayoutVersion(byte dataLayoutVersion) {
		this.dataLayoutVersion = dataLayoutVersion;
	}

	public byte getDataLayoutClass() {
		return dataLayoutClass;
	}

	public void setDataLayoutClass(byte dataLayoutClass) {
		this.dataLayoutClass = dataLayoutClass;
	}

	public byte getObjectModificationTimeVersion() {
		return objectModificationTimeVersion;
	}

	public void setObjectModificationTimeVersion(byte objectModificationTimeVersion) {
		this.objectModificationTimeVersion = objectModificationTimeVersion;
	}

	public byte getGroupSymbolTableNodeVersion() {
		return groupSymbolTableNodeVersion;
	}

	public void setGroupSymbolTableNodeVersion(byte groupSymbolTableNodeVersion) {
		this.groupSymbolTableNodeVersion = groupSymbolTableNodeVersion;
	}
}
