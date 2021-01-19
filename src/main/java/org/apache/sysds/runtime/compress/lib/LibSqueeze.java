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

package org.apache.sysds.runtime.compress.lib;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class LibSqueeze {

    // private static final Log LOG = LogFactory.getLog(LibSqueeze.class.getName());

    public static CompressedMatrixBlock squeeze(CompressedMatrixBlock m, int k) {

        CompressedMatrixBlock ret = new CompressedMatrixBlock(true);
        ret.setNumColumns(m.getNumColumns());
        ret.setNumRows(m.getNumRows());

        CompressionSettings cs = new CompressionSettingsBuilder().create();
        List<ColGroup> retCg = (k <= 1) ? singleThreadSqueeze(m, cs) : multiThreadSqueeze(m, cs, k);

        ret.allocateColGroupList(retCg);
        ret.setOverlapping(false);
        ret.setNonZeros(-1);

        // LOG.error(ret);
        if(ret.isOverlapping())
            throw new DMLCompressionException("Squeeze should output compressed nonOverlapping matrix");
        return ret;
    }

    private static List<ColGroup> singleThreadSqueeze(CompressedMatrixBlock m, CompressionSettings cs) {
        List<ColGroup> retCg = new ArrayList<>();

        int blkSz = 1;
        for(int i = 0; i < m.getNumColumns(); i += blkSz) {
            int[] columnIds = new int[Math.min(blkSz, m.getNumColumns() - i)];
            for(int j = 0; j < Math.min(blkSz, m.getNumColumns() - i); j++)
                columnIds[j] = i + j;
            retCg.add(extractNewGroup(m, cs, columnIds));
        }
        return retCg;
    }

    private static List<ColGroup> multiThreadSqueeze(CompressedMatrixBlock m, CompressionSettings cs, int k) {
        List<ColGroup> retCg = new ArrayList<>();
        ExecutorService pool = CommonThreadPool.get(k);
        ArrayList<SqueezeTask> tasks = new ArrayList<>();

        try {
            int blkSz = 1;
            for(int i = 0; i < m.getNumColumns(); i += blkSz) {
                int[] columnIds = new int[Math.min(blkSz, m.getNumColumns() - i)];
                for(int j = 0; j < Math.min(blkSz, m.getNumColumns() - i); j++)
                    columnIds[j] = i + j;
                tasks.add(new SqueezeTask(m, cs, columnIds));
            }

            for(Future<ColGroup> future : pool.invokeAll(tasks))
                retCg.add(future.get());
            pool.shutdown();
        }
        catch(InterruptedException | ExecutionException e) {
            throw new DMLRuntimeException(e);
        }

        return retCg;
    }

    private static ColGroup extractNewGroup(CompressedMatrixBlock m, CompressionSettings cs, int[] columnIds) {

        ABitmap map;
        if(columnIds.length > 1) {

            map = BitmapEncoder.extractBitmap(columnIds, m);
        }
        else {
            MatrixBlock tmp = new MatrixBlock(m.getNumRows(), 1, false).allocateDenseBlock();
            // public static void decompressToBlock(MatrixBlock target, int colIndex, List<ColGroup> colGroups) {
            ColGroup.decompressToBlock(tmp, columnIds[0], m.getColGroups());
            map = BitmapEncoder.extractBitmap(new int[1], tmp, true);
        }
        ColGroup newGroup = ColGroupFactory.compress(columnIds, m.getNumRows(), map, CompressionType.DDC, cs, m);
        return newGroup;
    }

    private static class SqueezeTask implements Callable<ColGroup> {
        private final CompressedMatrixBlock _m;
        private final CompressionSettings _cs;
        private final int[] _columnIds;

        protected SqueezeTask(CompressedMatrixBlock m, CompressionSettings cs, int[] columnIds) {
            _m = m;
            _cs = cs;
            _columnIds = columnIds;
        }

        @Override
        public ColGroup call() {
            return extractNewGroup(_m, _cs, _columnIds);
        }
    }
}
