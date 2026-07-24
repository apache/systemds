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

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

import static java.nio.ByteOrder.LITTLE_ENDIAN;

public class H5RootObject {

	protected H5ByteReader byteReader;
	protected BufferedOutputStream bufferedOutputStream;
	protected H5Superblock superblock;
	protected int rank;
	protected long row;
	protected long col;
	protected int[] logicalDimensions;
	protected int[] rawDimensions;
	protected int[] axisPermutation;
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
	public static final boolean HDF5_DEBUG = Boolean.getBoolean("sysds.hdf5.debug");

	public ByteBuffer readBufferFromAddress(long address, int length) {
		try {
			ByteBuffer bb = byteReader.read(address, length);
			bb.order(LITTLE_ENDIAN);
			bb.rewind();
			return bb;
		}
		catch(IOException e) {
			throw new H5RuntimeException(e);
		}
	}

	public ByteBuffer readBufferFromAddressNoOrder(long address, int length) {
		try {
			ByteBuffer bb = byteReader.read(address, length);
			bb.rewind();
			return bb;
		}
		catch(IOException e) {
			throw new H5RuntimeException(e);
		}
	}

	public void setByteReader(H5ByteReader byteReader) {
		this.byteReader = byteReader;
	}

	public H5ByteReader getByteReader() {
		return byteReader;
	}

	public void close() {
		try {
			if(byteReader != null)
				byteReader.close();
		}
		catch(IOException e) {
			throw new H5RuntimeException(e);
		}
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
		if(this.logicalDimensions != null && this.logicalDimensions.length > 0)
			this.logicalDimensions[0] = (int) row;
	}

	public long getCol() {
		return col;
	}

	public void setCol(long col) {
		this.col = col;
		if(this.logicalDimensions != null && this.logicalDimensions.length > 1)
			this.logicalDimensions[1] = (int) col;
	}

	public int getRank() {
		return rank;
	}

	public void setRank(int rank) {
		this.rank = rank;
		this.logicalDimensions = new int[rank];
		this.maxSizes = new int[rank];
	}

	public long getMaxRow() {
		return maxRow;
	}

	public void setMaxRow(long maxRow) {
		this.maxRow = maxRow;
		if(this.maxSizes != null && this.maxSizes.length > 0)
			this.maxSizes[0] = (int) maxRow;
	}

	public long getMaxCol() {
		return maxCol;
	}

	public void setMaxCol(long maxCol) {
		this.maxCol = maxCol;
		if(this.maxSizes != null && this.maxSizes.length > 1)
			this.maxSizes[1] = (int) maxCol;
	}

	public String getDatasetName() {
		return datasetName;
	}

	public void setDatasetName(String datasetName) {
		this.datasetName = datasetName;
	}

	public int[] getDimensions() {
		return logicalDimensions;
	}

	public int[] getLogicalDimensions() {
		return logicalDimensions;
	}

	public int[] getMaxSizes() {
		return maxSizes;
	}

	public int[] getRawDimensions() {
		return rawDimensions;
	}

	public int[] getAxisPermutation() {
		return axisPermutation;
	}

	public byte getDataSpaceVersion() {
		return dataSpaceVersion;
	}

	public void setDataSpaceVersion(byte dataSpaceVersion) {
		this.dataSpaceVersion = dataSpaceVersion;
	}

	public void setDimensions(int[] dimensions) {
		this.rawDimensions = dimensions;
		if(dimensions == null || dimensions.length == 0) {
			this.logicalDimensions = dimensions;
			this.axisPermutation = new int[0];
			this.row = 0;
			this.col = 0;
			return;
		}
		int[] logical = Arrays.copyOf(dimensions, dimensions.length);
		int[] permutation = identityPermutation(dimensions.length);
		this.logicalDimensions = logical;
		this.axisPermutation = permutation;
		this.row = logicalDimensions[0];
		this.col = flattenColumns(logicalDimensions);
		if(HDF5_DEBUG) {
			System.out.println("[HDF5] setDimensions rank=" + dimensions.length + " rawDims="
				+ java.util.Arrays.toString(dimensions) + " logicalDims=" + java.util.Arrays.toString(logicalDimensions)
				+ " axisPerm=" + java.util.Arrays.toString(axisPermutation) + " => rows=" + row + " cols(flat)=" + col);
		}
		if(HDF5_DEBUG) {
			System.out.println("[HDF5] setDimensions debug raw=" + java.util.Arrays.toString(dimensions)
				+ " logical=" + java.util.Arrays.toString(logicalDimensions) + " perm="
				+ java.util.Arrays.toString(axisPermutation));
		}
	}

	public void setMaxSizes(int[] maxSizes) {
		this.maxSizes = maxSizes;
		if(maxSizes == null || maxSizes.length == 0) {
			this.maxRow = 0;
			this.maxCol = 0;
			return;
		}
		this.maxRow = maxSizes[0];
		this.maxCol = flattenColumns(maxSizes);
		if(HDF5_DEBUG) {
			System.out.println("[HDF5] setMaxSizes rank=" + maxSizes.length + " max=" + java.util.Arrays.toString(maxSizes)
				+ " => maxRows=" + maxRow + " maxCols(flat)=" + maxCol);
		}
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

	private long flattenColumns(int[] dims) {
		if(dims.length == 1) {
			return 1;
		}
		long product = 1;
		for(int i = 1; i < dims.length; i++) {
			product = Math.multiplyExact(product, dims[i]);
		}
		return product;
	}

	private static int[] identityPermutation(int rank) {
		int[] perm = new int[rank];
		for(int i = 0; i < rank; i++)
			perm[i] = i;
		return perm;
	}

}
