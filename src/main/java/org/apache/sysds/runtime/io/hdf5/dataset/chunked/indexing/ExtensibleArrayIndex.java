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


package org.apache.sysds.runtime.io.hdf5.dataset.chunked.indexing;

import org.apache.sysds.runtime.io.hdf5.HdfFileChannel;
import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.checksum.ChecksumUtils;
import org.apache.sysds.runtime.io.hdf5.dataset.chunked.Chunk;
import org.apache.sysds.runtime.io.hdf5.dataset.chunked.DatasetInfo;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.exceptions.UnsupportedHdfException;

import java.nio.ByteBuffer;
import java.util.*;

import static org.apache.sysds.runtime.io.hdf5.Constants.UNDEFINED_ADDRESS;
import static org.apache.sysds.runtime.io.hdf5.Utils.readBytesAsUnsignedLong;
import static java.nio.charset.StandardCharsets.US_ASCII;

public class ExtensibleArrayIndex implements ChunkIndex {

    private static final byte[] EXTENSIBLE_ARRAY_HEADER_SIGNATURE = "EAHD".getBytes(US_ASCII);
    private static final byte[] EXTENSIBLE_ARRAY_INDEX_BLOCK_SIGNATURE = "EAIB".getBytes(US_ASCII);
    private static final byte[] EXTENSIBLE_ARRAY_DATA_BLOCK_SIGNATURE = "EADB".getBytes(US_ASCII);
    private static final byte[] EXTENSIBLE_ARRAY_SECONDARY_BLOCK_SIGNATURE = "EASB".getBytes(US_ASCII);

    private final long headerAddress;
    private final int clientId;
    private final boolean filtered; // If the chunks have filters applied
    private final int numberOfElementsInIndexBlock;
    private final int numberOfElements;
    private final int numberOfSecondaryBlocks;
    private final int blockOffsetSize;
    private final int dataBlockSize;
    private final int secondaryBlockSize;

    private final List<Chunk> chunks;
    private final int unfilteredChunkSize;
    private final int[] datasetDimensions;
    private final int[] chunkDimensions;

    private final int minNumberOfElementsInDataBlock;
    private final ExtensibleArrayCounter dataBlockElementCounter;
    private final int minNumberOfDataBlockPointers;
    private final ExtensibleArraySecondaryBlockPointerCounter secondaryBlockPointerCounter;
    private final int maxNumberOfElementsInDataBlockPageBits;
    private final int extensibleArrayElementSize;

    private int elementCounter = 0;

    public ExtensibleArrayIndex(HdfFileChannel hdfFc, long address, DatasetInfo datasetInfo) {
        this.headerAddress = address;
        this.unfilteredChunkSize = datasetInfo.getChunkSizeInBytes();
        this.datasetDimensions = datasetInfo.getDatasetDimensions();
        this.chunkDimensions = datasetInfo.getChunkDimensions();

        final int headerSize = 16 + hdfFc.getSizeOfOffsets() + 6 * hdfFc.getSizeOfLengths();
        final ByteBuffer bb = hdfFc.readBufferFromAddress(address, headerSize);

        verifySignature(bb, EXTENSIBLE_ARRAY_HEADER_SIGNATURE);

        // Version Number
        final byte version = bb.get();
        if (version != 0) {
            throw new HdfException("Unsupported extensible array index version detected. Version: " + version);
        }

        clientId = bb.get();
        if (clientId == 0) {
            filtered = false;
        } else if (clientId == 1) {
            filtered = true;
        } else {
            throw new UnsupportedHdfException("Extensible array unsupported client ID: " + clientId);
        }

        extensibleArrayElementSize = bb.get();

        final int maxNumberOfElementsBits = bb.get();
        blockOffsetSize = maxNumberOfElementsBits / 8; // TODO round up?
        numberOfElementsInIndexBlock = bb.get();
        minNumberOfElementsInDataBlock = bb.get();
        dataBlockElementCounter = new ExtensibleArrayCounter(minNumberOfElementsInDataBlock);
        minNumberOfDataBlockPointers = bb.get();
        secondaryBlockPointerCounter = new ExtensibleArraySecondaryBlockPointerCounter(minNumberOfDataBlockPointers);
        maxNumberOfElementsInDataBlockPageBits = bb.get();

        numberOfSecondaryBlocks = Utils.readBytesAsUnsignedInt(bb, hdfFc.getSizeOfLengths());
        secondaryBlockSize = Utils.readBytesAsUnsignedInt(bb, hdfFc.getSizeOfLengths());
        final int numberOfDataBlocks = Utils.readBytesAsUnsignedInt(bb, hdfFc.getSizeOfLengths());
        dataBlockSize = Utils.readBytesAsUnsignedInt(bb, hdfFc.getSizeOfLengths());

        final int maxIndexSet = Utils.readBytesAsUnsignedInt(bb, hdfFc.getSizeOfLengths());
        chunks = new ArrayList<>(maxIndexSet);

        numberOfElements = Utils.readBytesAsUnsignedInt(bb, hdfFc.getSizeOfLengths());

        final int indexBlockAddress = Utils.readBytesAsUnsignedInt(bb, hdfFc.getSizeOfLengths());

        new ExtensibleArrayIndexBlock(hdfFc, indexBlockAddress);

        // Checksum
        bb.rewind();
        ChecksumUtils.validateChecksum(bb);
    }

    private class ExtensibleArrayIndexBlock {

        private ExtensibleArrayIndexBlock(HdfFileChannel hdfFc, long address) {

            // Figure out the size of the index block
            final int headerSize = 6 + hdfFc.getSizeOfOffsets()
                    // TODO need to handle filtered elements
                    + hdfFc.getSizeOfOffsets() * numberOfElementsInIndexBlock // direct chunk pointers
                    + 6 * extensibleArrayElementSize // Always up to 6 data block pointers are in the index block
                    + numberOfSecondaryBlocks * hdfFc.getSizeOfOffsets() // Secondary block addresses.
                    + 4; // checksum


            final ByteBuffer bb = hdfFc.readBufferFromAddress(address, headerSize);

            verifySignature(bb, EXTENSIBLE_ARRAY_INDEX_BLOCK_SIGNATURE);

            // Version Number
            final byte version = bb.get();
            if (version != 0) {
                throw new HdfException("Unsupported fixed array data block version detected. Version: " + version);
            }

            final int clientId = bb.get();
            if (clientId != ExtensibleArrayIndex.this.clientId) {
                throw new HdfException("Extensible array client ID mismatch. Possible file corruption detected");
            }

            final long headerAddress = readBytesAsUnsignedLong(bb, hdfFc.getSizeOfOffsets());
            if (headerAddress != ExtensibleArrayIndex.this.headerAddress) {
                throw new HdfException("Extensible array data block header address mismatch");
            }

            // Elements in Index block
            boolean readElement = true;
            for (int i = 0; readElement && i < numberOfElementsInIndexBlock; i++) {
                readElement = readElement(bb, hdfFc);
            }

            // Guard against all the elements having already been read
            if (readElement && numberOfElements > numberOfElementsInIndexBlock) {
                // Upto 6 data block pointers directly in the index block
                for (int i = 0; i < 6; i++) {
                    final long dataBlockAddress = readBytesAsUnsignedLong(bb, hdfFc.getSizeOfOffsets());
                    if (dataBlockAddress == UNDEFINED_ADDRESS) {
                        break; // There was less than 6 data blocks for the full dataset
                    }
                    new ExtensibleArrayDataBlock(hdfFc, dataBlockAddress);
                }
            }

            // Now read secondary blocks
            for (int i = 0; i < numberOfSecondaryBlocks; i++) {
                final long secondaryBlockAddress = readBytesAsUnsignedLong(bb, hdfFc.getSizeOfOffsets());
                new ExtensibleArraySecondaryBlock(hdfFc, secondaryBlockAddress);
            }

            // Checksum
            int checksum = bb.getInt();
            // TODO checksums always seem to be 0 or -1?
        }

        private class ExtensibleArrayDataBlock {

            private ExtensibleArrayDataBlock(HdfFileChannel hdfFc, long address) {

                final int numberOfElementsInDataBlock = dataBlockElementCounter.getNextNumberOfChunks();
                final int headerSize = 6 + hdfFc.getSizeOfOffsets() + blockOffsetSize
                        + numberOfElementsInDataBlock * extensibleArrayElementSize // elements (chunks)
                        + 4; // checksum

                final ByteBuffer bb = hdfFc.readBufferFromAddress(address, headerSize);

                verifySignature(bb, EXTENSIBLE_ARRAY_DATA_BLOCK_SIGNATURE);

                // Version Number
                final byte version = bb.get();
                if (version != 0) {
                    throw new HdfException("Unsupported extensible array data block version detected. Version: " + version);
                }

                final int clientId = bb.get();
                if (clientId != ExtensibleArrayIndex.this.clientId) {
                    throw new HdfException("Extensible array client ID mismatch. Possible file corruption detected");
                }

                final long headerAddress = readBytesAsUnsignedLong(bb, hdfFc.getSizeOfOffsets());
                if (headerAddress != ExtensibleArrayIndex.this.headerAddress) {
                    throw new HdfException("Extensible array data block header address mismatch");
                }

                long blockOffset = readBytesAsUnsignedLong(bb, blockOffsetSize);

                // Page bitmap

                // Data block addresses
                boolean readElement = true;
                for (int i = 0; readElement && i < numberOfElementsInDataBlock; i++) {
                    readElement = readElement(bb, hdfFc);
                }

                // Checksum
                bb.rewind();
                ChecksumUtils.validateChecksum(bb);
            }

        }

        private class ExtensibleArraySecondaryBlock {

            private ExtensibleArraySecondaryBlock(HdfFileChannel hdfFc, long address) {

                final int numberOfPointers = secondaryBlockPointerCounter.getNextNumberOfPointers();
                final int secondaryBlockSize = 6 + hdfFc.getSizeOfOffsets() +
                        blockOffsetSize +
                        // Page Bitmap ?
                        numberOfPointers * extensibleArrayElementSize +
                        4; // checksum


                final ByteBuffer bb = hdfFc.readBufferFromAddress(address, secondaryBlockSize);

                verifySignature(bb, EXTENSIBLE_ARRAY_SECONDARY_BLOCK_SIGNATURE);

                // Version Number
                final byte version = bb.get();
                if (version != 0) {
                    throw new HdfException("Unsupported fixed array data block version detected. Version: " + version);
                }

                final int clientId = bb.get();
                if (clientId != ExtensibleArrayIndex.this.clientId) {
                    throw new HdfException("Extensible array client ID mismatch. Possible file corruption detected");
                }

                final long headerAddress = readBytesAsUnsignedLong(bb, hdfFc.getSizeOfOffsets());
                if (headerAddress != ExtensibleArrayIndex.this.headerAddress) {
                    throw new HdfException("Extensible array secondary block header address mismatch");
                }

                final long blockOffset = readBytesAsUnsignedLong(bb, blockOffsetSize);

                // TODO page bitmap

                // Data block addresses
                for (int i = 0; i < numberOfPointers; i++) {
                    long dataBlockAddress = readBytesAsUnsignedLong(bb, hdfFc.getSizeOfOffsets());
                    if (dataBlockAddress == UNDEFINED_ADDRESS) {
                        break; // This is the last secondary block and not full.
                    }
                    new ExtensibleArrayDataBlock(hdfFc, dataBlockAddress);
                }

                // Checksum
                int checksum = bb.getInt();
                if(checksum != UNDEFINED_ADDRESS) {
                    bb.limit(bb.position());
                    bb.rewind();
                    ChecksumUtils.validateChecksum(bb);
                }
            }

        }


        /**
         * Reads an element from the buffer and adds it to the chunks list.
         *
         * @param bb buffer to read from
         * @param hdfFc the HDF file channel
         * @return true if element was read false otherwise
         */
        private boolean readElement(ByteBuffer bb, HdfFileChannel hdfFc) {
            final long chunkAddress = readBytesAsUnsignedLong(bb, hdfFc.getSizeOfOffsets());
            if (chunkAddress != UNDEFINED_ADDRESS) {
                final int[] chunkOffset = Utils.chunkIndexToChunkOffset(elementCounter, chunkDimensions, datasetDimensions);
                if (filtered) { // Filtered
                    final int chunkSizeInBytes = Utils.readBytesAsUnsignedInt(bb, extensibleArrayElementSize - hdfFc.getSizeOfOffsets() - 4);
                    final BitSet filterMask = BitSet.valueOf(new byte[] { bb.get(), bb.get(), bb.get(), bb.get() });
                    chunks.add(new ChunkImpl(chunkAddress, chunkSizeInBytes, chunkOffset, filterMask));
                } else { // Not filtered
                    chunks.add(new ChunkImpl(chunkAddress, unfilteredChunkSize, chunkOffset));
                }
                elementCounter++;
                return true;
            } else {
                return false;
            }
        }

    }

    private void verifySignature(ByteBuffer bb, byte[] expectedSignature) {
        byte[] actualSignature = new byte[expectedSignature.length];
        bb.get(actualSignature, 0, expectedSignature.length);

        // Verify signature
        if (!Arrays.equals(expectedSignature, actualSignature)) {
            String signatureStr = new String(expectedSignature, US_ASCII);
            throw new HdfException("Signature '" + signatureStr + "' not matched, at address ");
        }
    }

    /**
     * This counts the number of elements (chunks) in a data block. The scheme used to assign blocks is described here
     * https://doi.org/10.1007/3-540-48447-7_4
     */
    /* package */ static class ExtensibleArrayCounter {

        private final int minNumberOfElementsInDataBlock;

        private int blockSizeMultiplier = 1;
        private int numberOfBlocks = 1;
        private int blockCounter = 0;
        private boolean increaseNumberOfBlocksNext = false;

        /* package */ ExtensibleArrayCounter(int initialNumberOfElements) {
            this.minNumberOfElementsInDataBlock = initialNumberOfElements;
        }

        public int getNextNumberOfChunks() {
            if (blockCounter < numberOfBlocks) {
                blockCounter++;
            } else if (increaseNumberOfBlocksNext) {
                increaseNumberOfBlocksNext = false;
                numberOfBlocks *= 2;
                blockCounter = 1;
            } else {
                increaseNumberOfBlocksNext = true;
                blockSizeMultiplier *= 2;
                blockCounter = 1;
            }
            return blockSizeMultiplier * minNumberOfElementsInDataBlock;
        }

        @Override
        public String toString() {
            return "ExtensibleArrayCounter{" +
                    "minNumberOfElementsInDataBlock=" + minNumberOfElementsInDataBlock +
                    ", blockSizeMultiplier=" + blockSizeMultiplier +
                    ", numberOfBlocks=" + numberOfBlocks +
                    ", blockCounter=" + blockCounter +
                    ", increaseNumberOfBlocksNext=" + increaseNumberOfBlocksNext +
                    '}';
        }
    }

    /* package */ static class ExtensibleArraySecondaryBlockPointerCounter {

        private static final int REPEATS = 2;

        private int numberOfPointers;
        private int counter = 0;

        /* package */ ExtensibleArraySecondaryBlockPointerCounter(int initialNumberOfPointers) {
            this.numberOfPointers = initialNumberOfPointers;
        }

        public int getNextNumberOfPointers() {
            if (counter < REPEATS) {
                counter++;
            } else {
                numberOfPointers *= 2;
                counter = 1;
            }
            return numberOfPointers;
        }

    }

    @Override
    public Collection<Chunk> getAllChunks() {
        return chunks;
    }
}
