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

import org.apache.sysds.runtime.io.hdf5.dataset.chunked.Chunk;
import org.apache.sysds.runtime.io.hdf5.dataset.chunked.DatasetInfo;
import org.apache.sysds.runtime.io.hdf5.object.message.DataLayoutMessage.ChunkedDataLayoutMessageV4;

import java.util.Collection;
import java.util.Collections;

public class SingleChunkIndex implements ChunkIndex {

    private final Chunk singleChunk;

    public SingleChunkIndex(ChunkedDataLayoutMessageV4 layoutMessageV4, DatasetInfo datasetInfo) {
        final int[] chunkOffset = new int[datasetInfo.getDatasetDimensions().length]; // Single chunk so zero offset
        if (layoutMessageV4.isFilteredSingleChunk()) {
            this.singleChunk = new ChunkImpl(layoutMessageV4.getAddress(), layoutMessageV4.getSizeOfFilteredSingleChunk(), chunkOffset, layoutMessageV4.getFilterMaskFilteredSingleChunk());
        } else {
            this.singleChunk = new ChunkImpl(layoutMessageV4.getAddress(), datasetInfo.getChunkSizeInBytes(), chunkOffset);
        }
    }

    @Override
    public Collection<Chunk> getAllChunks() {
        return Collections.singletonList(singleChunk);
    }

}
