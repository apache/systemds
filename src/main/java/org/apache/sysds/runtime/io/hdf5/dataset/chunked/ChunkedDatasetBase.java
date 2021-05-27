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


package org.apache.sysds.runtime.io.hdf5.dataset.chunked;

import org.apache.sysds.runtime.io.hdf5.HdfFileChannel;
import org.apache.sysds.runtime.io.hdf5.ObjectHeader;
import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.api.Group;
import org.apache.sysds.runtime.io.hdf5.dataset.DatasetBase;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.filter.FilterManager;
import org.apache.sysds.runtime.io.hdf5.filter.FilterPipeline;
import org.apache.sysds.runtime.io.hdf5.object.message.FilterPipelineMessage;
import org.apache.commons.lang3.concurrent.ConcurrentException;
import org.apache.commons.lang3.concurrent.LazyInitializer;
import org.apache.sysds.runtime.io.hdf5.api.ChunkedDataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

import static java.lang.Math.toIntExact;

public abstract class ChunkedDatasetBase extends DatasetBase implements ChunkedDataset {
    private static final Logger logger = LoggerFactory.getLogger(ChunkedDatasetBase.class);

    protected final FilterPipelineLazyInitializer lazyPipeline;

    public ChunkedDatasetBase(HdfFileChannel hdfFc, long address, String name, Group parent, ObjectHeader oh) {
        super(hdfFc, address, name, parent, oh);
        lazyPipeline = new FilterPipelineLazyInitializer();
    }

    protected void fillDataFromChunk(final Chunk chunk,
                                     final byte[] dataArray,
                                     final int[] chunkDimensions,
                                     final int[] chunkInternalOffsets,
                                     final int[] dataOffsets,
                                     final int fastestChunkDim,
                                     final int elementSize) {

        logger.debug("Filling data from chunk '{}'", chunk);

        // Get the un-filtered (decompressed) data in this chunk
        final byte[] chunkData = decompressChunk(chunk);

        // Now need to figure out how to put this chunks data into the output array
        final int[] chunkOffset = chunk.getChunkOffset();
        final int initialChunkOffset = Utils.dimensionIndexToLinearIndex(chunkOffset, getDimensions());

        if (!isPartialChunk(chunk)) {
            // Not a partial chunk so can always copy the max amount
            final int length = fastestChunkDim * elementSize;
            for (int i = 0; i < chunkInternalOffsets.length; i++) {
                System.arraycopy(
                        chunkData, chunkInternalOffsets[i], // src
                        dataArray, (dataOffsets[i] + initialChunkOffset) * elementSize, // dest
                        length); // length
            }
        } else {
            logger.debug("Handling partial chunk '{}'", chunk);
            // Partial chunk
            final int highestDimIndex = getDimensions().length - 1;

            for (int i = 0; i < chunkInternalOffsets.length; i++) {
                // Quick check first if the data starts outside then we know this part of the chunk can be skipped
                // Does not consider dimensions
                if (dataOffsets[i] > dataArray.length) {
                    continue;
                }
                // Is this part of the chunk outside the dataset including dimensions?
                if(partOfChunkIsOutsideDataset(chunkInternalOffsets[i]/ elementSize, chunkDimensions, chunkOffset)) {
                    continue;
                }

                // Its inside so we need to copy at least something. Now work out how much?
                final int length = elementSize * Math.min(fastestChunkDim,
                        fastestChunkDim - (chunkOffset[highestDimIndex] + chunkDimensions[highestDimIndex] - getDimensions()[highestDimIndex]));

                System.arraycopy(
                        chunkData, chunkInternalOffsets[i], // src
                        dataArray, (dataOffsets[i] + initialChunkOffset) * elementSize, // dest
                        length); // length
            }

        }
    }

    private boolean partOfChunkIsOutsideDataset(final int chunkInternalOffsetIndex,
                                                final int[] chunkDimensions,
                                                final int[] chunkOffset) {

        int[] locationInChunk = Utils.linearIndexToDimensionIndex(chunkInternalOffsetIndex, chunkDimensions);
        for (int j = 0; j < locationInChunk.length - 1; j++) {
            // Check if this dimension would be outside the dataset
            if (chunkOffset[j] + locationInChunk[j] >= getDimensions()[j]) {
                return true;
            }
        }
        // Nothing is outside
        return false;
    }

    /**
     * Calculates the linear offsets into the dataset for each of the chunks internal offsets. It can be thought of as
     * only doing this do the first chunk as to calculate the offsets required for any other chunk you need to add
     * the initial linear offset of that chunk to each of these values.
     *
     * @param chunkInternalOffsets a chunk offset
     * @return offset in the dataset
     */
    protected int[] getDataOffsets(int[] chunkInternalOffsets) {

        final int[] dimensionLinearOffsets = getDimensionLinearOffsets();
        final int[] chunkDimensions = getChunkDimensions();
        final int elementSize = getDataType().getSize();

        final int[] dataOffsets = new int[chunkInternalOffsets.length];
        for (int i = 0; i < chunkInternalOffsets.length; i++) {
            final int[] chunkDimIndex = Utils.linearIndexToDimensionIndex((chunkInternalOffsets[i] / elementSize), chunkDimensions);

            int dataOffset = 0;
            for (int j = 0; j < chunkDimIndex.length; j++) {
                dataOffset += chunkDimIndex[j] * dimensionLinearOffsets[j];
            }
            dataOffsets[i] = dataOffset;
        }
        return dataOffsets;
    }

    @Override
    public ByteBuffer getDataBuffer() {
        logger.trace("Getting data buffer for {}", getPath());

        // Need to load the full buffer into memory so create the array
        final byte[] dataArray = new byte[toIntExact(getSizeInBytes())];
        logger.trace("Created data buffer for '{}' of size {} bytes", getPath(), dataArray.length);

        final int elementSize = getDataType().getSize();

        // Get all chunks because were reading the whole dataset
        final Collection<Chunk> chunks = getAllChunks();

        // These are all the same for every chunk
        final int[] chunkDimensions = getChunkDimensions();
        final int[] chunkInternalOffsets = getChunkInternalOffsets(chunkDimensions, elementSize);
        final int[] dataOffsets = getDataOffsets(chunkInternalOffsets);
        final int fastestChunkDim = chunkDimensions[chunkDimensions.length - 1];

        // Parallel decoding and filling, this is where all the work is done
        chunks.parallelStream().forEach(chunk -> fillDataFromChunk(chunk, dataArray, chunkDimensions,
                chunkInternalOffsets, dataOffsets, fastestChunkDim, elementSize));

        return ByteBuffer.wrap(dataArray);
    }

    /**
     * Gets the number of linear steps to move for one step in the corresponding dimension
     *
     * @return array of the number of linear step to move for one step in each dimension
     */
    private int[] getDimensionLinearOffsets() {
        int dimLength = getDimensions().length;
        int[] dimensionLinearOffsets = new int[dimLength];
        Arrays.fill(dimensionLinearOffsets, 1);
        // dimensionLinearOffsets.length - 1 because a step in the fastest dim is always 1
        for (int i = 0; i < dimensionLinearOffsets.length - 1; i++) {
            for (int j = i + 1; j < dimensionLinearOffsets.length; j++) {
                dimensionLinearOffsets[i] *= getDimensions()[j];
            }
        }
        return dimensionLinearOffsets;
    }

    /**
     * Gets the offsets inside a chunk where a contiguous run of data starts.
     *
     * @param chunkDimensions the dimensions of each chunk
     * @param elementSize number of bytes in a an dataset element
     * @return an array of locations inside the chunk where contiguous data starts
     */
    protected int[] getChunkInternalOffsets(int[] chunkDimensions, int elementSize) {
        final int fastestChunkDim = chunkDimensions[chunkDimensions.length - 1];
        final int numOfOffsets = Arrays.stream(chunkDimensions)
                .limit(chunkDimensions.length - 1L)
                .reduce(1, (a, b) -> a * b);

        final int[] chunkOffsets = new int[numOfOffsets];
        for (int i = 0; i < numOfOffsets; i++) {
            chunkOffsets[i] = i * fastestChunkDim * elementSize;
        }
        return chunkOffsets;
    }

    /**
     * A partial chunk is one that is not completely inside the dataset. i.e. some of its contents are not part of the
     * dataset
     *
     * @param chunk The chunk to test
     * @return true if this is a partial chunk
     */
    private boolean isPartialChunk(Chunk chunk) {
        final int[] datasetDims = getDimensions();
        final int[] chunkOffset = chunk.getChunkOffset();
        final int[] chunkDims = getChunkDimensions();

        for (int i = 0; i < chunkOffset.length; i++) {
            if (chunkOffset[i] + chunkDims[i] > datasetDims[i]) {
                return true;
            }
        }

        return false;
    }

    private byte[] decompressChunk(Chunk chunk) {
        // Get the encoded (i.e. compressed buffer)
        final ByteBuffer encodedBuffer = getDataBuffer(chunk);

        // Get the encoded data from buffer
        final byte[] encodedBytes = new byte[encodedBuffer.remaining()];
        encodedBuffer.get(encodedBytes);

        try {
            final FilterPipeline pipeline = this.lazyPipeline.get();

            if (pipeline == null) {
                // No filters
                logger.debug("No filters returning decoded chunk '{}'", chunk);
                return encodedBytes;
            }

            // Decode using the pipeline applying the filters
            final byte[] decodedBytes = pipeline.decode(encodedBytes);
            logger.debug("Decoded {}", chunk);

            return decodedBytes;
        } catch (ConcurrentException e) {
            throw new HdfException("Failed to get filter pipeline", e);
        }
    }

    private ByteBuffer getDataBuffer(Chunk chunk) {
        try {
            return hdfFc.map(chunk.getAddress(), chunk.getSize());
        } catch (Exception e) {
            throw new HdfException(
                    "Failed to read chunk for dataset '" + getPath() + "' at address " + chunk.getAddress());
        }
    }

    protected final class FilterPipelineLazyInitializer extends LazyInitializer<FilterPipeline> {
        @Override
        protected FilterPipeline initialize() {
            logger.debug("Lazy initializing filter pipeline for '{}'", getPath());

            // If the dataset has filters get the message
            if (oh.hasMessageOfType(FilterPipelineMessage.class)) {
                FilterPipelineMessage filterPipelineMessage = oh.getMessageOfType(FilterPipelineMessage.class);
                return FilterManager.getPipeline(filterPipelineMessage);
            } else {
                // No filters
                return null;
            }
        }
    }


    @Override
    public ByteBuffer getRawChunkBuffer(int[] chunkOffset) {
        final Chunk chunk = getChunk(new ChunkOffset(chunkOffset));
        if(chunk == null) {
            throw new HdfException("No chunk with offset " + Arrays.toString(chunkOffset) +
                    " in dataset: " + getPath());
        }
        return getDataBuffer(chunk);
    }

    private Collection<Chunk> getAllChunks() {
        return getChunkLookup().values();
    }

    private Chunk getChunk(ChunkOffset chunkOffset) {
        return getChunkLookup().get(chunkOffset);
    }

    protected abstract Map<ChunkOffset, Chunk> getChunkLookup();

    @Override
    public boolean isEmpty() {
        return getChunkLookup().isEmpty();
    }

    @Override
    public long getStorageInBytes() {
        return getChunkLookup().values().stream().mapToLong(Chunk::getSize).sum();
    }
}
