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
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.BitSet;

public class ChunkImpl implements Chunk {

    private static final BitSet NOT_FILTERED_MASK = BitSet.valueOf(new byte[4]); // No filter mask so just all off

    private final long address;
    private final int size;
    private final int[] chunkOffset;
    private final BitSet filterMask;

    public ChunkImpl(long address, int size, int[] chunkOffset) {
        this(address, size, chunkOffset, NOT_FILTERED_MASK);
    }

    public ChunkImpl(long address, int size, int[] chunkOffset, BitSet filterMask) {
        this.address = address;
        this.size = size;
        this.chunkOffset = ArrayUtils.clone(chunkOffset);
        this.filterMask = filterMask;
    }

    @Override
    public int getSize() {
        return size;
    }

    @Override
    public BitSet getFilterMask() {
        return filterMask;
    }

    @Override
    public int[] getChunkOffset() {
        return ArrayUtils.clone(chunkOffset);
    }

    @Override
    public long getAddress() {
        return address;
    }

    @Override
    public String toString() {
        return "ChunkImpl{" +
                "address=" + address +
                ", size=" + size +
                ", chunkOffset=" + Arrays.toString(chunkOffset) +
                ", filterMask=" + filterMask +
                '}';
    }
}
