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


package org.apache.sysds.runtime.io.hdf5.btree.record;

import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.dataset.chunked.Chunk;
import org.apache.sysds.runtime.io.hdf5.dataset.chunked.DatasetInfo;
import org.apache.sysds.runtime.io.hdf5.dataset.chunked.indexing.ChunkImpl;

import java.nio.ByteBuffer;

public class NonFilteredDatasetChunks extends BTreeDatasetChunkRecord {

    private final Chunk chunk;

    public NonFilteredDatasetChunks(ByteBuffer buffer, DatasetInfo datasetInfo) {
        final long address = Utils.readBytesAsUnsignedLong(buffer, 8); // size of offsets

        int[] chunkOffset = new int[datasetInfo.getDatasetDimensions().length];
        for (int i = 0; i < chunkOffset.length; i++) {
            chunkOffset[i] = Utils.readBytesAsUnsignedInt(buffer, 8) * datasetInfo.getChunkDimensions()[i];
        }

        chunk = new ChunkImpl(address, datasetInfo.getChunkSizeInBytes(), chunkOffset);
    }

    @Override
    public Chunk getChunk() {
        return chunk;
    }
}
