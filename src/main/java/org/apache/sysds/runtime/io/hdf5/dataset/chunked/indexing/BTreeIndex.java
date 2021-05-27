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
import org.apache.sysds.runtime.io.hdf5.btree.BTreeV2;
import org.apache.sysds.runtime.io.hdf5.btree.record.BTreeDatasetChunkRecord;
import org.apache.sysds.runtime.io.hdf5.dataset.chunked.Chunk;
import org.apache.sysds.runtime.io.hdf5.dataset.chunked.DatasetInfo;

import java.util.Collection;

import static java.util.stream.Collectors.toList;

public class BTreeIndex implements ChunkIndex {

    private final BTreeV2<BTreeDatasetChunkRecord> bTreeV2;

    public BTreeIndex(HdfFileChannel hdfFc, long address, DatasetInfo datasetInfo) {
        bTreeV2 = new BTreeV2<>(hdfFc, address, datasetInfo);
    }

    @Override
    public Collection<Chunk> getAllChunks() {
        return bTreeV2.getRecords().stream().map(BTreeDatasetChunkRecord::getChunk).collect(toList());
    }
}
