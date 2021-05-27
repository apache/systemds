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
import org.apache.sysds.runtime.io.hdf5.checksum.ChecksumUtils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.exceptions.UnsupportedHdfException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.Map.Entry;

import static org.apache.sysds.runtime.io.hdf5.Constants.UNDEFINED_ADDRESS;
import static org.apache.sysds.runtime.io.hdf5.Utils.*;
import static java.nio.ByteOrder.LITTLE_ENDIAN;

/**
 * Fractal heap implementation. Used for storing data which can be looked up via
 * an ID.
 * <p>
 * <a href=
 * "https://support.hdfgroup.org/HDF5/doc/H5.format.html#FractalHeap">Format
 * Spec</a>
 * </p>
 *
 * @author James Mudd
 */
public class FractalHeap {
	private static final Logger logger = LoggerFactory.getLogger(FractalHeap.class);

	private static final byte[] FRACTAL_HEAP_SIGNATURE = "FRHP".getBytes(StandardCharsets.US_ASCII);
	private static final byte[] INDIRECT_BLOCK_SIGNATURE = "FHIB".getBytes(StandardCharsets.US_ASCII);
	private static final byte[] DIRECT_BLOCK_SIGNATURE = "FHDB".getBytes(StandardCharsets.US_ASCII);

	private static final BigInteger TWO = BigInteger.valueOf(2L);

	private final long address;
	private final HdfFileChannel hdfFc;
	private final Superblock sb;

	private final int maxDirectBlockSize;
	private final long maxSizeOfManagedObjects;
	private final int idLength;
	private final int ioFiltersLength;
	private final int currentRowsInRootIndirectBlock;
	private final int startingRowsInRootIndirectBlock;
	private final int startingBlockSize;
	private final int tableWidth;
	private final long numberOfTinyObjectsInHeap;
	private final long sizeOfTinyObjectsInHeap;
	private final long numberOfHugeObjectsInHeap;
	private final long sizeOfHugeObjectsInHeap;
	private final long numberOfManagedObjectsInHeap;
	private final long offsetOfDirectBlockAllocationIteratorInManagedSpace;
	private final long amountOfAllocatedManagedSpaceInHeap;
	private final long amountOfManagedSpaceInHeap;
	private final long addressOfManagedBlocksFreeSpaceManager;
	private final long freeSpaceInManagedBlocks;
	private final long bTreeAddressOfHugeObjects;
	private final long nextHugeObjectId;
	private final BitSet flags;

	private int blockIndex = 0;

	/**
	 * This map is that holds all the direct blocks keyed by their offset in the
	 * heap address space.
	 */
	private final NavigableMap<Long, DirectBlock> directBlocks = new TreeMap<>(); // Sorted map

	private final int bytesToStoreOffset;
	private final int bytesToStoreLength;

	public FractalHeap(HdfFileChannel hdfFc, long address) {
		this.hdfFc = hdfFc;
		this.sb = hdfFc.getSuperblock();
		this.address = address;

		final int headerSize = 4 + 1 + 2 + 2 + 1 + 4 + 12 * sb.getSizeOfLengths() + 3 * sb.getSizeOfOffsets() + 2
				+ 2 + 2 + 2 + 4;

		ByteBuffer bb = hdfFc.readBufferFromAddress(address, headerSize);

		byte[] formatSignatureBytes = new byte[4];
		bb.get(formatSignatureBytes, 0, formatSignatureBytes.length);

		// Verify signature
		if (!Arrays.equals(FRACTAL_HEAP_SIGNATURE, formatSignatureBytes)) {
			throw new HdfException("Fractal heap signature 'FRHP' not matched, at address " + address);
		}

		// Version Number
		final byte version = bb.get();
		if (version != 0) {
			throw new HdfException("Unsupported fractal heap version detected. Version: " + version);
		}

		idLength = readBytesAsUnsignedInt(bb, 2);
		ioFiltersLength = readBytesAsUnsignedInt(bb, 2);

		flags = BitSet.valueOf(new byte[] { bb.get() });

		maxSizeOfManagedObjects = readBytesAsUnsignedLong(bb, 4);

		nextHugeObjectId = readBytesAsUnsignedLong(bb, sb.getSizeOfLengths());

		bTreeAddressOfHugeObjects = readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());

		freeSpaceInManagedBlocks = readBytesAsUnsignedLong(bb, sb.getSizeOfLengths());

		addressOfManagedBlocksFreeSpaceManager = readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());

		amountOfManagedSpaceInHeap = readBytesAsUnsignedLong(bb, sb.getSizeOfLengths());
		amountOfAllocatedManagedSpaceInHeap = readBytesAsUnsignedLong(bb, sb.getSizeOfLengths());
		offsetOfDirectBlockAllocationIteratorInManagedSpace = readBytesAsUnsignedLong(bb, sb.getSizeOfLengths());
		numberOfManagedObjectsInHeap = readBytesAsUnsignedLong(bb, sb.getSizeOfLengths());

		sizeOfHugeObjectsInHeap = readBytesAsUnsignedLong(bb, sb.getSizeOfLengths());
		numberOfHugeObjectsInHeap = readBytesAsUnsignedLong(bb, sb.getSizeOfLengths());
		sizeOfTinyObjectsInHeap = readBytesAsUnsignedLong(bb, sb.getSizeOfLengths());
		numberOfTinyObjectsInHeap = readBytesAsUnsignedLong(bb, sb.getSizeOfLengths());

		tableWidth = readBytesAsUnsignedInt(bb, 2);

		startingBlockSize = readBytesAsUnsignedInt(bb, sb.getSizeOfLengths());
		maxDirectBlockSize = readBytesAsUnsignedInt(bb, sb.getSizeOfLengths());

		// Value stored in bits
		final int maxHeapSize = readBytesAsUnsignedInt(bb, 2);
		// Calculate byte sizes needed later
		bytesToStoreOffset = (int) Math.ceil(maxHeapSize / 8.0);
		bytesToStoreLength = bytesNeededToHoldNumber(Math.min(maxDirectBlockSize, maxSizeOfManagedObjects));

		startingRowsInRootIndirectBlock = readBytesAsUnsignedInt(bb, 2);

		final long addressOfRootBlock = readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());

		currentRowsInRootIndirectBlock = readBytesAsUnsignedInt(bb, 2);

		if (ioFiltersLength > 0) {
			throw new UnsupportedHdfException("IO filters are currently not supported");
		}

		// Read the root block
		if (addressOfRootBlock != UNDEFINED_ADDRESS) {
			if (currentRowsInRootIndirectBlock == 0) {
				// Read direct block
				DirectBlock db = new DirectBlock(addressOfRootBlock);
				directBlocks.put(db.blockOffset, db);
			} else {
				// Read indirect block
				IndirectBlock indirectBlock = new IndirectBlock(addressOfRootBlock);
				for (long directBlockAddress : indirectBlock.childBlockAddresses) {
					int blockSize = getSizeOfDirectBlock(blockIndex++);
					if (blockSize != -1) {
						DirectBlock db = new DirectBlock(directBlockAddress);
						directBlocks.put(db.getBlockOffset(), db);
					} else {
						new IndirectBlock(address);
					}
				}
			}
		}

		bb.rewind();
		ChecksumUtils.validateChecksum(bb);

		logger.debug("Read fractal heap at address {}, loaded {} direct blocks", address, directBlocks.size());
	}

	public ByteBuffer getId(ByteBuffer buffer) {
		if (buffer.remaining() != idLength) {
			throw new HdfException("ID length is incorrect accessing fractal heap at address " + address
					+ ". IDs should be " + idLength + " bytes but was " + buffer.capacity() + " bytes.");
		}

		BitSet idFlags = BitSet.valueOf(new byte[] { buffer.get() });

		final int version = bitsToInt(idFlags, 6, 2);
		if (version != 0) {
			throw new HdfException("Unsupported btree v2 ID version detected. Version: " + version);
		}

		final int type = bitsToInt(idFlags, 4, 2);

		switch (type) {
		case 0: // Managed Objects
			long offset = readBytesAsUnsignedLong(buffer, bytesToStoreOffset);
			int length = readBytesAsUnsignedInt(buffer, bytesToStoreLength);

			logger.debug("Getting ID at offset={} length={}", offset, length);

			// Figure out which direct block holds the offset
			Entry<Long, DirectBlock> entry = directBlocks.floorEntry(offset);

			ByteBuffer bb = entry.getValue().getData();
			bb.order(LITTLE_ENDIAN);
			bb.position(Math.toIntExact(offset - entry.getKey()));
			return createSubBuffer(bb, length);

//		case 1: // Huge objects
//            if (this.bTreeAddressOfHugeObjects <= 0) {
//			    throw new UnsupportedHdfException("Huge objects without BTreev2 are currently not supported");
//			}
//
//			BTreeV2<HugeFractalHeapObjectUnfilteredRecord> hugeObjectBTree =
//			    new BTreeV2<>(this.hdfFc, this.bTreeAddressOfHugeObjects);
//
//			if (hugeObjectBTree.getRecords().size() != 1) {
//			    throw new UnsupportedHdfException("Only Huge objects BTrees with 1 record are currently supported");
//			}
//
//			HugeFractalHeapObjectUnfilteredRecord ho = hugeObjectBTree.getRecords().get(0);
//
//			return this.hdfFc.readBufferFromAddress(ho.getAddress(), (int) ho.getLength());
		case 2: // Tiny objects
			throw new UnsupportedHdfException("Tiny objects are currently not supported");
		default:
			throw new HdfException("Unrecognized ID type, type=" + type);
		}
	}

	private class IndirectBlock {

		private final List<Long> childBlockAddresses;

		private IndirectBlock(long address) {
			final int headerSize = 4 + 1 + sb.getSizeOfOffsets() + bytesToStoreOffset
					+ currentRowsInRootIndirectBlock * tableWidth * getRowSize() + 4;

			ByteBuffer bb = hdfFc.readBufferFromAddress(address, headerSize);

			byte[] formatSignatureBytes = new byte[4];
			bb.get(formatSignatureBytes, 0, formatSignatureBytes.length);

			// Verify signature
			if (!Arrays.equals(INDIRECT_BLOCK_SIGNATURE, formatSignatureBytes)) {
				throw new HdfException(
						"Fractal heap indirect block signature 'FHIB' not matched, at address " + address);
			}

			// Version Number
			byte indirectBlockVersion = bb.get();
			if (indirectBlockVersion != 0) {
				throw new HdfException("Unsupported indirect block version detected. Version: " + indirectBlockVersion);
			}

			long heapAddress = readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());
			if (heapAddress != FractalHeap.this.address) {
				throw new HdfException("Indirect block read from invalid fractal heap");
			}

			final long blockOffset = readBytesAsUnsignedLong(bb, bytesToStoreOffset);

			childBlockAddresses = new ArrayList<>(currentRowsInRootIndirectBlock * tableWidth);
			for (int i = 0; i < currentRowsInRootIndirectBlock * tableWidth; i++) {
				// TODO only works for unfiltered
				long childAddress = readBytesAsUnsignedLong(bb, getRowSize());
				if (childAddress == UNDEFINED_ADDRESS) {
					break;
				} else {
					childBlockAddresses.add(childAddress);
				}
			}

			// Validate checksum
			bb.rewind();
			ChecksumUtils.validateChecksum(bb);
		}

		private boolean isIoFilters() {
			return ioFiltersLength > 0;
		}

		private int getRowSize() {
			int size = sb.getSizeOfOffsets();
			if (isIoFilters()) {
				size += sb.getSizeOfLengths();
				size += 4; // filter mask
			}
			return size;
		}

	}

	private class DirectBlock {

		private static final int CHECKSUM_PRESENT_BIT = 1;
		private final long address;
		private final ByteBuffer data;
		private final long blockOffset;

		private DirectBlock(long address) {
			this.address = address;

			final int headerSize = 4 + 1 + sb.getSizeOfOffsets() + bytesToStoreOffset + 4;

			ByteBuffer bb = hdfFc.readBufferFromAddress(address, headerSize);

			byte[] formatSignatureBytes = new byte[4];
			bb.get(formatSignatureBytes, 0, formatSignatureBytes.length);

			// Verify signature
			if (!Arrays.equals(DIRECT_BLOCK_SIGNATURE, formatSignatureBytes)) {
				throw new HdfException("Fractal heap direct block signature 'FHDB' not matched, at address " + address);
			}

			// Version Number
			byte directBlockVersion = bb.get();
			if (directBlockVersion != 0) {
				throw new HdfException("Unsupported direct block version detected. Version: " + directBlockVersion);
			}

			long heapAddress = readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());
			if (heapAddress != FractalHeap.this.address) {
				throw new HdfException("Indirect block read from invalid fractal heap");
			}

			blockOffset = readBytesAsUnsignedLong(bb, bytesToStoreOffset);

			data = hdfFc.map(address, getSizeOfDirectBlock(blockIndex));

			if (checksumPresent()) {
				int storedChecksum = bb.getInt();
				// TODO Validate checksum
			}
		}

		private boolean checksumPresent() {
			return flags.get(CHECKSUM_PRESENT_BIT);
		}

		public ByteBuffer getData() {
			return data.order(LITTLE_ENDIAN);
		}

		public long getBlockOffset() {
			return blockOffset;
		}

		@Override
		public String toString() {
			return "DirectBlock [address=" + address + ", blockOffset=" + blockOffset + ", data=" + data + "]";
		}

	}

	private int getSizeOfDirectBlock(int blockIndex) {
		int row = blockIndex / tableWidth; // int division
		if (row < 2) {
			return startingBlockSize;
		} else {
			int size = startingBlockSize * TWO.pow(row - 1).intValueExact();
			if (size < maxDirectBlockSize) {
				return size;
			} else {
				return -1; // Indicates the block is an indirect block
			}
		}
	}

	@Override
	public String toString() {
		return "FractalHeap [address=" + address + ", idLength=" + idLength + ", numberOfTinyObjectsInHeap="
				+ numberOfTinyObjectsInHeap + ", numberOfHugeObjectsInHeap=" + numberOfHugeObjectsInHeap
				+ ", numberOfManagedObjectsInHeap=" + numberOfManagedObjectsInHeap + "]";
	}

	public long getAddress() {
		return address;
	}

	public int getMaxDirectBlockSize() {
		return maxDirectBlockSize;
	}

	public long getMaxSizeOfManagedObjects() {
		return maxSizeOfManagedObjects;
	}

	public int getIdLength() {
		return idLength;
	}

	public int getIoFiltersLength() {
		return ioFiltersLength;
	}

	public int getStartingRowsInRootIndirectBlock() {
		return startingRowsInRootIndirectBlock;
	}

	public long getNumberOfTinyObjectsInHeap() {
		return numberOfTinyObjectsInHeap;
	}

	public long getSizeOfTinyObjectsInHeap() {
		return sizeOfTinyObjectsInHeap;
	}

	public long getNumberOfHugeObjectsInHeap() {
		return numberOfHugeObjectsInHeap;
	}

	public long getSizeOfHugeObjectsInHeap() {
		return sizeOfHugeObjectsInHeap;
	}

	public long getNumberOfManagedObjectsInHeap() {
		return numberOfManagedObjectsInHeap;
	}

	public long getOffsetOfDirectBlockAllocationIteratorInManagedSpace() {
		return offsetOfDirectBlockAllocationIteratorInManagedSpace;
	}

	public long getAmountOfAllocatedManagedSpaceInHeap() {
		return amountOfAllocatedManagedSpaceInHeap;
	}

	public long getAmountOfManagedSpaceInHeap() {
		return amountOfManagedSpaceInHeap;
	}

	public long getAddressOfManagedBlocksFreeSpaceManager() {
		return addressOfManagedBlocksFreeSpaceManager;
	}

	public long getFreeSpaceInManagedBlocks() {
		return freeSpaceInManagedBlocks;
	}

	public long getBTreeAddressOfHugeObjects() {
		return bTreeAddressOfHugeObjects;
	}

	public long getNextHugeObjectId() {
		return nextHugeObjectId;
	}

}
