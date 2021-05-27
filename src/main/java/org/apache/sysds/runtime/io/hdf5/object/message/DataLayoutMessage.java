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

import org.apache.sysds.runtime.io.hdf5.Constants;
import org.apache.sysds.runtime.io.hdf5.Superblock;
import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.exceptions.UnsupportedHdfException;
import org.apache.commons.lang3.ArrayUtils;

import java.nio.ByteBuffer;
import java.util.BitSet;

public abstract class DataLayoutMessage extends Message {

	public DataLayoutMessage(BitSet flags) {
		super(flags);
	}

	public abstract DataLayout getDataLayout();

	public static DataLayoutMessage createDataLayoutMessage(ByteBuffer bb, Superblock sb, BitSet flags) {
		final byte version = bb.get();

		switch (version) {
			case 1:
			case 2:
				return readV1V2Message(bb, sb, flags);
			case 3:
			case 4:
				return readV3V4Message(bb, sb, flags, version);
			default:
				throw new UnsupportedHdfException("Unsupported data layout message version detected. Detected version = " + version);
		}
	}

	private static DataLayoutMessage readV1V2Message(ByteBuffer bb, Superblock sb, BitSet flags) {
		byte dimensionality = bb.get(); // for chunked is +1 than actual dims

		final byte layoutClass = bb.get();

		bb.position(bb.position() + 5); // skip reserved bytes

		final long dataAddress;
		if (layoutClass != 0) { // not compact
			dataAddress = Utils.readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());
		} else {
			dataAddress = Constants.UNDEFINED_ADDRESS;
		}

		// If chunked value stored is +1 so correct it here
		if (layoutClass == 2) {
			dimensionality--;
		}

		int[] dimensions = new int[dimensionality];
		for (int i = 0; i < dimensions.length; i++) {
			dimensions[i] = Utils.readBytesAsUnsignedInt(bb, 4);
		}

		switch (layoutClass) {
			case 0: // Compact Storage
				final int compactDataSize = Utils.readBytesAsUnsignedInt(bb, 4);
				final ByteBuffer compactDataBuffer = Utils.createSubBuffer(bb, compactDataSize);
				return new CompactDataLayoutMessage(flags, compactDataBuffer);
			case 1: // Contiguous
				return new ContiguousDataLayoutMessage(flags, dataAddress, -1L);
			case 2: // Chunked
				final int dataElementSize = Utils.readBytesAsUnsignedInt(bb, 4);
				return new ChunkedDataLayoutMessage(flags, dataAddress, dataElementSize, dimensions);
			default:
				throw new UnsupportedHdfException("Unknown storage layout " + layoutClass);
		}
	}

	private static DataLayoutMessage readV3V4Message(ByteBuffer bb, Superblock sb, BitSet flags, byte version) {
		final byte layoutClass = bb.get();

		switch (layoutClass) {
			case 0: // Compact Storage
				return new CompactDataLayoutMessage(bb, flags);
			case 1: // Contiguous Storage
				return new ContiguousDataLayoutMessage(bb, sb, flags);
			case 2: // Chunked Storage
				if (version == 3) {
					return new ChunkedDataLayoutMessage(bb, sb, flags);
				} else { // v4
					return new ChunkedDataLayoutMessageV4(bb, sb, flags);
				}
			case 3: // Virtual storage
				throw new UnsupportedHdfException("Virtual storage is not supported");
			default:
				throw new UnsupportedHdfException("Unknown storage layout " + layoutClass);
		}
	}

	public static class CompactDataLayoutMessage extends DataLayoutMessage {

		private final ByteBuffer dataBuffer;

		private CompactDataLayoutMessage(BitSet flags, ByteBuffer dataBuffer) {
			super(flags);
			this.dataBuffer = dataBuffer;
		}

		private CompactDataLayoutMessage(ByteBuffer bb, BitSet flags) {
			super(flags);
			final int compactDataSize = Utils.readBytesAsUnsignedInt(bb, 2);
			this.dataBuffer = Utils.createSubBuffer(bb, compactDataSize);
		}

		@Override
		public DataLayout getDataLayout() {
			return DataLayout.COMPACT;
		}

		public ByteBuffer getDataBuffer() {
			return dataBuffer.slice();
		}
	}

	public static class ContiguousDataLayoutMessage extends DataLayoutMessage {

		private final long address;
		private final long size;

		private ContiguousDataLayoutMessage(BitSet flags, long address, long size) {
			super(flags);
			this.address = address;
			this.size = size;
		}

		private ContiguousDataLayoutMessage(ByteBuffer bb, Superblock sb, BitSet flags) {
			super(flags);
			address = Utils.readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());
			size = Utils.readBytesAsUnsignedLong(bb, sb.getSizeOfLengths());
		}

		@Override
		public DataLayout getDataLayout() {
			return DataLayout.CONTIGUOUS;
		}

		public long getAddress() {
			return address;
		}

		/**
		 * @return size in bytes if known or -1 otherwise
		 */
		public long getSize() {
			return size;
		}
	}

	public static class ChunkedDataLayoutMessage extends DataLayoutMessage {

		private final long bTreeAddress;
		private final int size;
		private final int[] chunkDimensions;

		public ChunkedDataLayoutMessage(BitSet flags, long bTreeAddress, int size, int[] chunkDimensions) {
			super(flags);
			this.bTreeAddress = bTreeAddress;
			this.size = size;
			this.chunkDimensions = ArrayUtils.clone(chunkDimensions);
		}

		private ChunkedDataLayoutMessage(ByteBuffer bb, Superblock sb, BitSet flags) {
			super(flags);
			final int chunkDimensionality = bb.get() - 1;
			bTreeAddress = Utils.readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());
			chunkDimensions = new int[chunkDimensionality];
			for (int i = 0; i < chunkDimensions.length; i++) {
				chunkDimensions[i] = Utils.readBytesAsUnsignedInt(bb, 4);
			}
			size = Utils.readBytesAsUnsignedInt(bb, 4);
		}

		@Override
		public DataLayout getDataLayout() {
			return DataLayout.CHUNKED;
		}

		public long getBTreeAddress() {
			return bTreeAddress;
		}

		public int getSize() {
			return size;
		}

		public int[] getChunkDimensions() {
			return ArrayUtils.clone(chunkDimensions);
		}
	}

	public static class ChunkedDataLayoutMessageV4 extends DataLayoutMessage {

		private static final int DONT_FILTER_PARTIAL_BOUND_CHUNKS = 0;
		private static final int SINGLE_INDEX_WITH_FILTER = 1;

		private final long address;
		private final byte indexingType;
		private final int[] chunkDimensions;

		private byte pageBits;
		private byte maxBits;
		private byte indexElements;
		private byte minPointers;
		private byte minElements;
		private int nodeSize;
		private byte splitPercent;
		private byte mergePercent;

		// Fields only for filtered single chunk
		private boolean isFilteredSingleChunk = false;
		private int sizeOfFilteredSingleChunk;
		private BitSet filterMaskFilteredSingleChunk;

		private ChunkedDataLayoutMessageV4(ByteBuffer bb, Superblock sb, BitSet flags) {
			super(flags);

			final BitSet chunkedFlags = BitSet.valueOf(new byte[]{bb.get()});
			final int chunkDimensionality = bb.get();
			final int dimSizeBytes = bb.get();

			chunkDimensions = new int[chunkDimensionality];
			for (int i = 0; i < chunkDimensions.length; i++) {
				chunkDimensions[i] = Utils.readBytesAsUnsignedInt(bb, dimSizeBytes);
			}

			indexingType = bb.get();

			switch (indexingType) {
				case 1: // Single Chunk
					if (chunkedFlags.get(SINGLE_INDEX_WITH_FILTER)) {
						isFilteredSingleChunk = true;
						sizeOfFilteredSingleChunk = Utils.readBytesAsUnsignedInt(bb, sb.getSizeOfLengths());
						filterMaskFilteredSingleChunk = BitSet.valueOf(new byte[]{bb.get(), bb.get(), bb.get(), bb.get()});
					}
					break;

				case 2: // Implicit
					break; // There is nothing for this case

				case 3: // Fixed Array
					pageBits = bb.get();
					break;

				case 4: // Extensible Array
					maxBits = bb.get();
					indexElements = bb.get();
					minPointers = bb.get();
					minElements = bb.get();
					pageBits = bb.get(); // This is wrong in the spec says 2 bytes its actually 1
					break;

				case 5: // B tree v2
					nodeSize = bb.getInt();
					splitPercent = bb.get();
					mergePercent = bb.get();
					break;

				default:
					throw new UnsupportedHdfException("Unrecognized chunk indexing type. type=" + indexingType);
			}

			address = Utils.readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());
		}

		@Override
		public DataLayout getDataLayout() {
			return DataLayout.CHUNKED;
		}

		public long getAddress() {
			return address;
		}

		public byte getPageBits() {
			return pageBits;
		}

		public byte getMaxBits() {
			return maxBits;
		}

		public byte getIndexElements() {
			return indexElements;
		}

		public byte getMinPointers() {
			return minPointers;
		}

		public byte getMinElements() {
			return minElements;
		}

		public int getNodeSize() {
			return nodeSize;
		}

		public byte getSplitPercent() {
			return splitPercent;
		}

		public byte getMergePercent() {
			return mergePercent;
		}

		public byte getIndexingType() {
			return indexingType;
		}

		public int[] getChunkDimensions() {
			return ArrayUtils.clone(chunkDimensions);
		}

		public int getSizeOfFilteredSingleChunk() {
			if (!isFilteredSingleChunk) {
				throw new HdfException("Requested size of filtered single chunk when its not set.");
			}
			return sizeOfFilteredSingleChunk;
		}

		public BitSet getFilterMaskFilteredSingleChunk() {
			if (!isFilteredSingleChunk) {
				throw new HdfException("Requested filter mask of filtered single chunk when its not set.");
			}
			return filterMaskFilteredSingleChunk;
		}

		public boolean isFilteredSingleChunk() {
			return isFilteredSingleChunk;
		}
	}

}
