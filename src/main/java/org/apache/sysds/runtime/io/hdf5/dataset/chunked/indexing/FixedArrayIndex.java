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

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class FixedArrayIndex implements ChunkIndex {

    private static final byte[] FIXED_ARRAY_HEADER_SIGNATURE = "FAHD".getBytes(StandardCharsets.US_ASCII);
    private static final byte[] FIXED_ARRAY_DATA_BLOCK_SIGNATURE = "FADB".getBytes(StandardCharsets.US_ASCII);

    private final long address;
    private final int unfilteredChunkSize;

    private final int[] datasetDimensions;
    private final int[] chunkDimensions;

    private final int clientId;
    private final int entrySize;
    private final int pageBits;
    private final int maxNumberOfEntries;
    private final long dataBlockAddress;

    private final List<Chunk> chunks;

    public FixedArrayIndex(HdfFileChannel hdfFc, long address, DatasetInfo datasetInfo) {
        this.address = address;
        this.unfilteredChunkSize = datasetInfo.getChunkSizeInBytes();
        this.datasetDimensions = datasetInfo.getDatasetDimensions();
        this.chunkDimensions = datasetInfo.getChunkDimensions();

        final int headerSize = 12 + hdfFc.getSizeOfOffsets() + hdfFc.getSizeOfLengths();
        final ByteBuffer bb = hdfFc.readBufferFromAddress(address, headerSize);

        byte[] formatSignatureBytes = new byte[4];
        bb.get(formatSignatureBytes, 0, formatSignatureBytes.length);

        // Verify signature
        if (!Arrays.equals(FIXED_ARRAY_HEADER_SIGNATURE, formatSignatureBytes)) {
            throw new HdfException("Fixed array header signature 'FAHD' not matched, at address " + address);
        }

        // Version Number
        final byte version = bb.get();
        if (version != 0) {
            throw new HdfException("Unsupported fixed array index version detected. Version: " + version);
        }

        clientId = bb.get();
        entrySize = bb.get();
        pageBits = bb.get();

        maxNumberOfEntries = Utils.readBytesAsUnsignedInt(bb, hdfFc.getSizeOfLengths());
        dataBlockAddress = Utils.readBytesAsUnsignedLong(bb, hdfFc.getSizeOfOffsets());

        chunks = new ArrayList<>(maxNumberOfEntries);

        // Checksum
        bb.rewind();
        ChecksumUtils.validateChecksum(bb);

        // Building the object fills the chunks. Probably shoudld be changed
        new FixedArrayDataBlock(this, hdfFc, dataBlockAddress);
    }

    private static class FixedArrayDataBlock {

        private FixedArrayDataBlock(FixedArrayIndex fixedArrayIndex, HdfFileChannel hdfFc, long address) {

            // TODO header size ignoring paging
            final int headerSize = 6 + hdfFc.getSizeOfOffsets() + fixedArrayIndex.entrySize * fixedArrayIndex.maxNumberOfEntries + 4;
            final ByteBuffer bb = hdfFc.readBufferFromAddress(address, headerSize);

            byte[] formatSignatureBytes = new byte[4];
            bb.get(formatSignatureBytes, 0, formatSignatureBytes.length);

            // Verify signature
            if (!Arrays.equals(FIXED_ARRAY_DATA_BLOCK_SIGNATURE, formatSignatureBytes)) {
                throw new HdfException("Fixed array data block signature 'FADB' not matched, at address " + address);
            }

            // Version Number
            final byte version = bb.get();
            if (version != 0) {
                throw new HdfException("Unsupported fixed array data block version detected. Version: " + version);
            }

            final int clientId = bb.get();
            if (clientId != fixedArrayIndex.clientId) {
                throw new HdfException("Fixed array client ID mismatch. Possible file corruption detected");
            }

            final long headerAddress = Utils.readBytesAsUnsignedLong(bb, hdfFc.getSizeOfOffsets());
            if (headerAddress != fixedArrayIndex.address) {
                throw new HdfException("Fixed array data block header address missmatch");
            }

            // TODO ignoring paging here might need to revisit

            if (clientId == 0) { // Not filtered
                for (int i = 0; i < fixedArrayIndex.maxNumberOfEntries; i++) {
                    final long chunkAddress = Utils.readBytesAsUnsignedLong(bb, hdfFc.getSizeOfOffsets());
                    final int[] chunkOffset = Utils.chunkIndexToChunkOffset(i, fixedArrayIndex.chunkDimensions, fixedArrayIndex.datasetDimensions);
                    fixedArrayIndex.chunks.add(new ChunkImpl(chunkAddress, fixedArrayIndex.unfilteredChunkSize, chunkOffset));
                }
            } else  if (clientId == 1) { // Filtered
                for (int i = 0; i < fixedArrayIndex.maxNumberOfEntries; i++) {
                    final long chunkAddress = Utils.readBytesAsUnsignedLong(bb, hdfFc.getSizeOfOffsets());
                    final int chunkSizeInBytes = Utils.readBytesAsUnsignedInt(bb, fixedArrayIndex.entrySize - hdfFc.getSizeOfOffsets() - 4);
                    final BitSet filterMask = BitSet.valueOf(new byte[] { bb.get(), bb.get(), bb.get(), bb.get() });
                    final int[] chunkOffset = Utils.chunkIndexToChunkOffset(i,  fixedArrayIndex.chunkDimensions, fixedArrayIndex.datasetDimensions);

                    fixedArrayIndex.chunks.add(new ChunkImpl(chunkAddress, chunkSizeInBytes, chunkOffset, filterMask));
                }
            } else {
                throw new HdfException("Unrecognized client ID  = " + clientId);
            }

            bb.rewind();
            ChecksumUtils.validateChecksum(bb);
        }
    }

    @Override
    public Collection<Chunk> getAllChunks() {
        return chunks;
    }
}
