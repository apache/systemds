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

import org.apache.sysds.runtime.io.hdf5.Constants;
import org.apache.sysds.runtime.io.hdf5.HdfFileChannel;
import org.apache.sysds.runtime.io.hdf5.ObjectHeader;
import org.apache.sysds.runtime.io.hdf5.api.Group;
import org.apache.sysds.runtime.io.hdf5.dataset.chunked.indexing.*;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;
import org.apache.sysds.runtime.io.hdf5.exceptions.UnsupportedHdfException;
import org.apache.sysds.runtime.io.hdf5.object.message.DataLayoutMessage.ChunkedDataLayoutMessageV4;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.concurrent.ConcurrentException;
import org.apache.commons.lang3.concurrent.LazyInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;
import java.util.function.Function;

import static java.util.stream.Collectors.toMap;

public class ChunkedDatasetV4 extends ChunkedDatasetBase {
    private static final Logger logger = LoggerFactory.getLogger(ChunkedDatasetV4.class);

    private final ChunkedDataLayoutMessageV4 layoutMessage;
    private final ChunkLookupLazyInitializer chunkLookupLazyInitializer;

    public ChunkedDatasetV4(HdfFileChannel hdfFc, long address, String name, Group parent, ObjectHeader oh) {
        super(hdfFc, address, name, parent, oh);

        layoutMessage = oh.getMessageOfType(ChunkedDataLayoutMessageV4.class);
        chunkLookupLazyInitializer = new ChunkLookupLazyInitializer();

        logger.debug("Created chunked v4 dataset. Index type {}", layoutMessage.getIndexingType());
    }

    @Override
    public int[] getChunkDimensions() {
        // TODO understand why there is an extra one on the end of this array
        int[] chunkDimensions = layoutMessage.getChunkDimensions();
        return ArrayUtils.subarray(chunkDimensions, 0, chunkDimensions.length -1);
    }

    @Override
    protected Map<ChunkOffset, Chunk> getChunkLookup() {
        try {
            return chunkLookupLazyInitializer.get();
        } catch (ConcurrentException e) {
            throw new HdfException("Failed to create chunk lookup for: " + getPath(), e);
        }
    }

    private final class ChunkLookupLazyInitializer extends LazyInitializer<Map<ChunkOffset, Chunk>> {
        @Override
        protected Map<ChunkOffset, Chunk> initialize() {
            logger.debug("Creating chunk lookup for '{}'", getPath());

            final DatasetInfo datasetInfo = new DatasetInfo(getChunkSizeInBytes(), getDimensions(), getChunkDimensions());
            final ChunkIndex chunkIndex;

            if(layoutMessage.getAddress() == Constants.UNDEFINED_ADDRESS) {
                logger.debug("No storage allocated for '{}'", getPath());
                chunkIndex = new EmptyChunkIndex();
            } else {

                switch (layoutMessage.getIndexingType()) {
                    case 1: // Single chunk
                        logger.debug("Reading single chunk indexed dataset");
                        chunkIndex = new SingleChunkIndex(layoutMessage, datasetInfo);
                        break;
                    case 2: // Implicit
                        throw new UnsupportedHdfException("Implicit indexing is currently not supported");
                    case 3: // Fixed array
                        logger.debug("Reading fixed array indexed dataset");
                        chunkIndex = new FixedArrayIndex(hdfFc, layoutMessage.getAddress(), datasetInfo);
                        break;
                    case 4: // Extensible Array
                        logger.debug("Reading extensible array indexed dataset");
                        chunkIndex = new ExtensibleArrayIndex(hdfFc, layoutMessage.getAddress(), datasetInfo);
                        break;
                    case 5: // B Tree V2
                        logger.debug("Reading B tree v2 indexed dataset");
                        chunkIndex = new BTreeIndex(hdfFc, layoutMessage.getAddress(), datasetInfo);
                        break;
                    default:
                        throw new HdfException("Unrecognized chunk indexing type = " + layoutMessage.getIndexingType());
                }
            }

            final Collection<Chunk> allChunks = chunkIndex.getAllChunks();

            return allChunks.stream().
                    collect(toMap(chunk -> new ChunkOffset(chunk.getChunkOffset()) // keys
                            , Function.identity())); // values
        }

        private int getChunkSizeInBytes() {
            return Arrays.stream(getChunkDimensions()).reduce(1, Math::multiplyExact) * getDataType().getSize();
        }

    }

}
