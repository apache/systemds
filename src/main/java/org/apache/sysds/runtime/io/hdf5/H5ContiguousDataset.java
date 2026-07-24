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

import org.apache.sysds.runtime.io.hdf5.message.H5DataLayoutMessage;
import org.apache.sysds.runtime.io.hdf5.message.H5DataSpaceMessage;
import org.apache.sysds.runtime.io.hdf5.message.H5DataTypeMessage;

import java.nio.ByteBuffer;
import java.util.Arrays;

import static java.nio.ByteOrder.LITTLE_ENDIAN;

public class H5ContiguousDataset {

	private final H5RootObject rootObject;
	private final H5DataLayoutMessage dataLayoutMessage;
	private final H5DataTypeMessage dataTypeMessage;
	@SuppressWarnings("unused")
	private final H5DataSpaceMessage dataSpaceMessage;
	private final boolean rankGt2;
	private final long elemSize;
	private final long dataSize;
	private ByteBuffer fullData;
	private boolean fullDataLoaded = false;
	private final int[] dims;
	private final int[] fileDims;
	private final long[] fileStrides;
	private final int[] axisPermutation;
	private final long rowByteStride;
	private final long rowByteSize;
	private long[] colOffsets;

	public H5ContiguousDataset(H5RootObject rootObject, H5ObjectHeader objectHeader) {

		this.rootObject = rootObject;
		this.dataLayoutMessage = objectHeader.getMessageOfType(H5DataLayoutMessage.class);
		if(this.dataLayoutMessage.getLayoutClass() != H5DataLayoutMessage.LAYOUT_CLASS_CONTIGUOUS) {
			throw new H5RuntimeException("Unsupported data layout class: "
				+ this.dataLayoutMessage.getLayoutClass() + " (only contiguous datasets are supported).");
		}
		this.dataTypeMessage = objectHeader.getMessageOfType(H5DataTypeMessage.class);
		this.dataSpaceMessage = objectHeader.getMessageOfType(H5DataSpaceMessage.class);

		this.dims = rootObject.getLogicalDimensions();
		this.fileDims = rootObject.getRawDimensions() != null ? rootObject.getRawDimensions() : this.dims;
		this.axisPermutation = normalizePermutation(rootObject.getAxisPermutation(), this.dims);
		this.rankGt2 = this.dims != null && this.dims.length > 2;
		this.elemSize = this.dataTypeMessage.getDoubleDataType().getSize();
		this.dataSize = this.dataLayoutMessage.getSize();
		this.fileStrides = computeStridesRowMajor(this.fileDims);
		this.rowByteStride = (fileStrides.length == 0) ? 0 : fileStrides[axisPermutation[0]] * elemSize;
		if(H5RootObject.HDF5_DEBUG && rankGt2) {
			System.out.println("[HDF5] dataset=" + rootObject.getDatasetName() + " logicalDims="
				+ Arrays.toString(dims) + " fileDims=" + Arrays.toString(fileDims) + " axisPerm="
				+ Arrays.toString(axisPermutation) + " fileStrides=" + Arrays.toString(fileStrides));
		}

		this.rowByteSize = rootObject.getCol() * elemSize;
	}

	public ByteBuffer getDataBuffer(int row) {
		return getDataBuffer(row, 1);
	}

	public ByteBuffer getDataBuffer(int row, int rowCount) {
		try {
			long cols = rootObject.getCol();
			long rowBytes = cols * elemSize;
			if(rowBytes > Integer.MAX_VALUE) {
				throw new H5RuntimeException("Row byte size exceeds buffer capacity: " + rowBytes);
			}
			if(rowCount <= 0) {
				throw new H5RuntimeException("Row count must be positive, got " + rowCount);
			}
			long readLengthLong = rowBytes * rowCount;
			if(readLengthLong > Integer.MAX_VALUE) {
				throw new H5RuntimeException("Requested read exceeds buffer capacity: " + readLengthLong);
			}
			int readLength = (int) readLengthLong;

			if(rankGt2) {
				if(isRowContiguous()) {
					long rowPos = row * rowByteSize;
					long layoutAddress = dataLayoutMessage.getAddress();
					long dataAddress = layoutAddress + rowPos;
					ByteBuffer data = rootObject.readBufferFromAddressNoOrder(dataAddress, readLength);
					data.order(LITTLE_ENDIAN);
					if(H5RootObject.HDF5_DEBUG) {
						System.out.println("[HDF5] getDataBuffer (rank>2 contiguous) dataset=" + rootObject.getDatasetName()
							+ " row=" + row + " rowCount=" + rowCount + " readLength=" + readLength);
					}
					return data;
				}
				if(rowCount != 1) {
					throw new H5RuntimeException("Row block reads are not supported for non-contiguous rank>2 datasets.");
				}
				if(!fullDataLoaded) {
					fullData = rootObject.readBufferFromAddressNoOrder(dataLayoutMessage.getAddress(),
						(int) dataSize);
					fullData.order(LITTLE_ENDIAN);
					fullDataLoaded = true;
				}
				if(colOffsets == null) {
					colOffsets = new long[(int) cols];
					for(int c = 0; c < cols; c++) {
						colOffsets[c] = computeByteOffset(0, c);
					}
				}
				ByteBuffer rowBuf = ByteBuffer.allocate(readLength).order(LITTLE_ENDIAN);
				if(H5RootObject.HDF5_DEBUG && row == 0) {
					long debugCols = Math.min(cols, 5);
					for(long c = 0; c < debugCols; c++) {
						long byteOff = rowByteStride * row + colOffsets[(int) c];
						double v = fullData.getDouble((int) byteOff);
						System.out.println("[HDF5] map(row=" + row + ", col=" + c + ") -> byteOff=" + byteOff
							+ " val=" + v);
					}
				}
				for(int c = 0; c < cols; c++) {
					long byteOff = rowByteStride * row + colOffsets[c];
					double v = fullData.getDouble((int) byteOff);
					if(H5RootObject.HDF5_DEBUG && row == 3 && c == 3) {
						System.out.println("[HDF5] sample(row=" + row + ", col=" + c + ") byteOff=" + byteOff
							+ " val=" + v);
					}
					rowBuf.putDouble(v);
				}
				rowBuf.rewind();
				if(H5RootObject.HDF5_DEBUG) {
					System.out.println("[HDF5] getDataBuffer (rank>2) dataset=" + rootObject.getDatasetName() + " row=" + row
						+ " cols=" + cols + " elemSize=" + elemSize + " dataSize=" + dataSize);
				}
				return rowBuf;
			}
			else {
				long rowPos = row * rowBytes;
				long layoutAddress = dataLayoutMessage.getAddress();
				// layoutAddress is already an absolute file offset for the contiguous data block.
				long dataAddress = layoutAddress + rowPos;
				ByteBuffer data = rootObject.readBufferFromAddressNoOrder(dataAddress, readLength);
				data.order(LITTLE_ENDIAN);
				if(H5RootObject.HDF5_DEBUG) {
					System.out.println("[HDF5] getDataBuffer dataset=" + rootObject.getDatasetName() + " row=" + row
						+ " layoutAddr=" + layoutAddress + " rowPos=" + rowPos + " readLength=" + readLength
						+ " col=" + cols + " rowCount=" + rowCount);
				}
				return data;
			}
		}
		catch(Exception e) {
			throw new H5RuntimeException("Failed to map data buffer for dataset", e);
		}
	}

	public void readRowDoubles(int row, double[] dest, int destPos) {
		long cols = rootObject.getCol();
		if(cols > Integer.MAX_VALUE) {
			throw new H5RuntimeException("Column count exceeds buffer capacity: " + cols);
		}
		int ncol = (int) cols;
		if(rankGt2) {
			if(isRowContiguous()) {
				ByteBuffer data = getDataBuffer(row, 1);
				data.order(LITTLE_ENDIAN);
				data.asDoubleBuffer().get(dest, destPos, ncol);
				return;
			}
			if(!fullDataLoaded) {
				fullData = rootObject.readBufferFromAddressNoOrder(dataLayoutMessage.getAddress(), (int) dataSize);
				fullData.order(LITTLE_ENDIAN);
				fullDataLoaded = true;
			}
			if(colOffsets == null) {
				colOffsets = new long[ncol];
				for(int c = 0; c < ncol; c++) {
					colOffsets[c] = computeByteOffset(0, c);
				}
			}
			long rowBase = rowByteStride * row;
			for(int c = 0; c < ncol; c++) {
				dest[destPos + c] = fullData.getDouble((int) (rowBase + colOffsets[c]));
			}
			return;
		}
		ByteBuffer data = getDataBuffer(row);
		data.order(LITTLE_ENDIAN);
		data.asDoubleBuffer().get(dest, destPos, ncol);
	}

	private static long[] computeStridesRowMajor(int[] dims) {
		if(dims == null || dims.length == 0)
			return new long[0];
		long[] strides = new long[dims.length];
		strides[dims.length - 1] = 1;
		for(int i = dims.length - 2; i >= 0; i--) {
			strides[i] = strides[i + 1] * dims[i + 1];
		}
		return strides;
	}

	private long computeByteOffset(long row, long col) {
		long linear = row * fileStrides[axisPermutation[0]];
		long rem = col;
		for(int axis = dims.length - 1; axis >= 1; axis--) {
			int dim = dims[axis];
			long idx = (dim == 0) ? 0 : rem % dim;
			rem = (dim == 0) ? 0 : rem / dim;
			linear += idx * fileStrides[axisPermutation[axis]];
		}
		return linear * elemSize;
	}

	private static int[] normalizePermutation(int[] permutation, int[] dims) {
		int rank = (dims == null) ? 0 : dims.length;
		if(permutation == null || permutation.length != rank) {
			int[] identity = new int[rank];
			for(int i = 0; i < rank; i++)
				identity[i] = i;
			return identity;
		}
		return permutation;
	}

	public H5DataTypeMessage getDataType() {
		return dataTypeMessage;
	}

	public long getDataAddress() {
		return dataLayoutMessage.getAddress();
	}

	public long getDataSize() {
		return dataSize;
	}

	public long getElementSize() {
		return elemSize;
	}

	public boolean isRankGt2() {
		return rankGt2;
	}

	public long getRowByteSize() {
		return rowByteSize;
	}

	public boolean isRowContiguous() {
		return rowByteStride == rowByteSize;
	}
}
