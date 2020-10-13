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

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupOLE;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupValue;
import org.apache.sysds.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class LibLeftMultBy {
    private static final Log LOG = LogFactory.getLog(LibLeftMultBy.class.getName());

    public static MatrixBlock leftMultByMatrix(List<ColGroup> groups, MatrixBlock that, MatrixBlock ret,
        boolean doTranspose, boolean allocTmp, int rl, int cl, boolean overlapping, int k, Pair<Integer, int[]> v) {

        if(ret == null)
            ret = new MatrixBlock(rl, cl, false, rl * cl);
        else if(!(ret.getNumColumns() == cl && ret.getNumRows() == rl && ret.isAllocated()))
            ret.reset(rl, cl, false, rl * cl);
        that = that instanceof CompressedMatrixBlock ? ((CompressedMatrixBlock) that).decompress() : that;

        // if(that.getNumRows() == 1) {
        // if(k > 1) {
        // return leftMultByVectorTranspose(groups, that, ret, doTranspose, k, v, overlapping);
        // }
        // else {
        // return leftMultByVectorTranspose(groups, that, ret, doTranspose, true, v, overlapping);
        // }
        // }
        // else {
        return leftMultByMatrix(groups, that, ret, k, cl, v, overlapping);
        // }
    }

    public static void leftMultByTransposeSelf(List<ColGroup> groups, MatrixBlock result, int gl, int gu, int k,
        int numColumns, Pair<Integer, int[]> v, boolean overlapping) {
        if(k <= 1 || overlapping) {
            leftMultByTransposeSelf(groups, result, gl, gu, v, overlapping);
        }
        else {
            try {
                ExecutorService pool = CommonThreadPool.get(k);
                ArrayList<MatrixMultTransposeTask> tasks = new ArrayList<>();
                int numgrp = groups.size();
                int blklen = (int) (Math.ceil((double) numgrp / (2 * k)));
                for(int i = 0; i < 2 * k & i * blklen < numColumns; i++)
                    tasks.add(new MatrixMultTransposeTask(groups, result, i * blklen,
                        Math.min((i + 1) * blklen, numgrp), v, overlapping));
                List<Future<Object>> ret = pool.invokeAll(tasks);
                for(Future<Object> tret : ret)
                    tret.get(); // check for errors
                pool.shutdown();
            }
            catch(InterruptedException | ExecutionException e) {
                throw new DMLRuntimeException(e);
            }
        }
    }

    private static MatrixBlock leftMultByMatrix(List<ColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
        int numColumns, Pair<Integer, int[]> v, boolean overlapping) {
        ret.allocateDenseBlock();
        if(that.isInSparseFormat()) {
            ret = leftMultBySparseMatrix(colGroups, that, ret, k, numColumns, v);
        }
        else {
            ret = leftMultByDenseMatrix(colGroups, that, ret, k, numColumns, v, overlapping);
        }

        ret.setNonZeros(ret.getNumColumns() * ret.getNumRows());
        return ret;
    }

    private static MatrixBlock leftMultByDenseMatrix(List<ColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
        int numColumns, Pair<Integer, int[]> v, boolean overlapping) {
        DenseBlock db = that.getDenseBlock();
        if(db == null)
            throw new DMLRuntimeException("Invalid LeftMult By Dense matrix, input matrix was sparse");

        double[] retV = ret.getDenseBlockValues();
        double[] thatV;
        int blockU;
        int blockL = 0;
        for(ColGroup grp : colGroups)
            if(grp instanceof ColGroupUncompressed)
                ((ColGroupUncompressed) grp).leftMultByMatrix(that, ret);

        for(int b = 0; b < db.numBlocks(); b++) {
            int blockSize = db.blockSize(b);
            blockU = Math.min(blockL + blockSize, ret.getNumRows());
            thatV = db.valuesAt(b);

            if(k == 1 || overlapping) {
                // Pair<Integer, int[]> v = getMaxNumValues(colGroups);
                for(int j = 0; j < colGroups.size(); j++) {
                    colGroups.get(j).leftMultByMatrix(thatV,
                        retV,
                        colGroups.get(j).getValues(),
                        that.getNumRows(),
                        ret.getNumColumns(),
                        0,
                        ret.getNumRows(),
                        0);
                }
            }
            else {
                try {
                    ExecutorService pool = CommonThreadPool.get(k);
                    // compute remaining compressed column groups in parallel
                    ArrayList<LeftMatrixMatrixMultTask> tasks = new ArrayList<>();
                    int rowBlockSize = 1;
                    for(int blo = blockL; blo < blockU; blo += rowBlockSize) {
                        tasks.add(new LeftMatrixMatrixMultTask(colGroups, thatV, retV, that.getNumRows(), numColumns,
                            blo, Math.min(blo + rowBlockSize, blockU), blo - blockL, v));
                    }

                    List<Future<Object>> futures = pool.invokeAll(tasks);

                    pool.shutdown();
                    for(Future<Object> future : futures)
                        future.get();
                }
                catch(InterruptedException | ExecutionException e) {
                    throw new DMLRuntimeException(e);
                }
            }
            blockL += blockSize;
        }
        return ret;
    }

    private static MatrixBlock leftMultByVectorTranspose(List<ColGroup> colGroups, MatrixBlock vector,
        MatrixBlock result, boolean doTranspose, boolean allocTmp, Pair<Integer, int[]> v, boolean overlap) {

        MatrixBlock rowVector = vector;
        // Note that transpose here is a metadata operation since the input is a vector.
        if(doTranspose) {
            rowVector = new MatrixBlock(1, vector.getNumRows(), false);
            LibMatrixReorg.transpose(vector, rowVector);
        }

        // initialize and allocate the result
        result.reset();
        result.allocateDenseBlock();

        // setup memory pool for reuse
        if(allocTmp) {
            // Pair<Integer, int[]> v = getMaxNumValues(colGroups);
            ColGroupValue.setupThreadLocalMemory(v.getLeft() + 1); // +1 for efficiency in DDC groups.
            for(int i = 0; i < colGroups.size(); i++) {
                colGroups.get(i).leftMultByRowVector(rowVector.getDenseBlockValues(),
                    result.getDenseBlockValues(),
                    v.getRight()[i]);
            }
        }
        else {

            for(ColGroup grp : colGroups) {
                grp.leftMultByRowVector(rowVector.getDenseBlockValues(), result.getDenseBlockValues(), -1);
            }
        }

        // delegate matrix-vector operation to each column group

        // post-processing
        if(allocTmp)
            ColGroupValue.cleanupThreadLocalMemory();
        result.recomputeNonZeros();

        return result;
    }

    public static MatrixBlock leftMultByVectorTranspose(List<ColGroup> colGroups, MatrixBlock vector,
        MatrixBlock result, boolean doTranspose, int k, Pair<Integer, int[]> v, boolean overlap) {
        // transpose vector if required
        MatrixBlock rowVector = vector;
        if(doTranspose) {
            rowVector = new MatrixBlock(1, vector.getNumRows(), false);
            LibMatrixReorg.transpose(vector, rowVector);
        }

        // initialize and allocate the result
        result.reset();
        result.allocateDenseBlock();

        // multi-threaded execution
        try {
            // compute uncompressed column group in parallel
            // ColGroupUncompressed uc = getUncompressedColGroup();
            // if(uc != null)
            // uc.leftMultByRowVector(rowVector, result, k);

            // compute remaining compressed column groups in parallel
            ExecutorService pool = CommonThreadPool.get(Math.min(colGroups.size(), k));
            ArrayList<LeftMatrixVectorMultTask> tasks = new ArrayList<>();

            // if(overlap){
            tasks.add(new LeftMatrixVectorMultTask(colGroups, rowVector, result, v));
            // } else{
            // ArrayList<ColGroup>[] grpParts = createStaticTaskPartitioning(colGroups, 4 * k, true);
            // for(ArrayList<ColGroup> groups : grpParts)
            // tasks.add(new LeftMatrixVectorMultTask(groups, rowVector, result, v));
            // }

            List<Future<Object>> ret = pool.invokeAll(tasks);
            pool.shutdown();
            for(Future<Object> tmp : ret)
                tmp.get();

        }
        catch(InterruptedException | ExecutionException e) {
            LOG.error(e);
            throw new DMLRuntimeException(e);
        }

        // post-processing
        result.recomputeNonZeros();

        return result;
    }

    private static MatrixBlock leftMultBySparseMatrix(List<ColGroup> colGroups, MatrixBlock that, MatrixBlock ret,
        int k, int numColumns, Pair<Integer, int[]> v) {

        SparseBlock sb = that.getSparseBlock();
        if(sb == null)
            throw new DMLRuntimeException("Invalid Left Mult by Sparse matrix, input matrix was dense");

        for(ColGroup grp : colGroups) {
            if(grp instanceof ColGroupUncompressed)
                ((ColGroupUncompressed) grp).leftMultByMatrix(that, ret);
        }

        if(k == 1) {
            double[][] materialized = new double[colGroups.size()][];
            boolean containsOLE = false;
            for(int i = 0; i < colGroups.size(); i++) {
                materialized[i] = colGroups.get(i).getValues();
                if(colGroups.get(i) instanceof ColGroupOLE) {
                    containsOLE = true;
                }
            }
            double[] materializedRow = containsOLE ? new double[CompressionSettings.BITMAP_BLOCK_SZ * 2] : null;

            for(int r = 0; r < that.getNumRows(); r++) {
                SparseRow row = sb.get(r);
                if(row != null) {

                    for(int j = 0; j < colGroups.size(); j++) {
                        colGroups.get(j).leftMultBySparseMatrix(row.size(),
                            row.indexes(),
                            row.values(),
                            ret.getDenseBlockValues(),
                            v.getRight()[j],
                            materialized[j],
                            that.getNumRows(),
                            ret.getNumColumns(),
                            r,
                            materializedRow);
                    }
                }
            }
        }
        else {
            ExecutorService pool = CommonThreadPool.get(k);
            ArrayList<LeftMatrixSparseMatrixMultTask> tasks = new ArrayList<>();
            try {
                // compute remaining compressed column groups in parallel
                // List<ColGroup>[] parts = createStaticTaskPartitioningForSparseMatrixMult(colGroups, k, false);
                // for(List<ColGroup> part : parts) {
                tasks.add(new LeftMatrixSparseMatrixMultTask(colGroups, sb, ret.getDenseBlockValues(),
                    that.getNumRows(), numColumns, v));
                // }

                List<Future<Object>> futures = pool.invokeAll(tasks);
                pool.shutdown();
                for(Future<Object> future : futures)
                    future.get();
            }
            catch(InterruptedException | ExecutionException e) {
                throw new DMLRuntimeException(e);
            }
        }

        return ret;

    }

    private static void leftMultByTransposeSelf(List<ColGroup> groups, MatrixBlock result, int gl, int gu,
        Pair<Integer, int[]> v, boolean overlapping) {
        final int numRows = groups.get(0).getNumRows();

        // preallocated dense tmp matrix blocks
        MatrixBlock lhs = new MatrixBlock(1, numRows, false);
        MatrixBlock tmpret = new MatrixBlock(1, result.getNumColumns(), false);
        lhs.allocateDenseBlock();
        tmpret.allocateDenseBlock();

        // setup memory pool for reuse
        ColGroupValue.setupThreadLocalMemory(v.getLeft() + 1);

        // approach: for each colgroup, extract uncompressed columns one at-a-time
        // vector-matrix multiplies against remaining col groups
        // for(int i = gl; i < gu; i++) {
        // get current group and relevant col groups
        // ColGroup group = groups.get(i);
        // int[] ixgroup = group.getColIndices();
        // List<ColGroup> tmpList = groups.subList(i, numGroups);

        // if(group instanceof ColGroupDDC // single DDC group
        // && ixgroup.length == 1 && !containsUC && numRows < CompressionSettings.BITMAP_BLOCK_SZ) {
        // // compute vector-matrix partial result
        // leftMultByVectorTranspose(tmpList, (ColGroupDDC) group, tmpret);

        // // write partial results (disjoint non-zeros)
        // LinearAlgebraUtils.copyNonZerosToUpperTriangle(result, tmpret, ixgroup[0]);
        // }
        // else {
        // for all uncompressed lhs columns vectors
        for(int j = 0; j < result.getNumColumns(); j++) {
            ColGroup.decompressToBlock(lhs, j, groups);

            if(!lhs.isEmptyBlock(false)) {
                // tmpret.reset();
                // compute vector-matrix partial result
                // leftMultByMatrix(groups,lhs, tmpret, false, true, 0, 0, overlapping, 1, v );
                leftMultByVectorTranspose(groups, lhs, tmpret, false, true, v, overlapping);
                // LOG.error(tmpret);

                // write partial results (disjoint non-zeros)
                LinearAlgebraUtils.copyNonZerosToUpperTriangle(result, tmpret, j);
            }
            lhs.reset();
            // }
            // }
        }

        // post processing
        ColGroupValue.cleanupThreadLocalMemory();
    }

    private static class LeftMatrixVectorMultTask implements Callable<Object> {
        private final List<ColGroup> _groups;
        private final MatrixBlock _vect;
        private final MatrixBlock _ret;
        private final Pair<Integer, int[]> _v;

        protected LeftMatrixVectorMultTask(List<ColGroup> groups, MatrixBlock vect, MatrixBlock ret,
            Pair<Integer, int[]> v) {
            _groups = groups;
            _vect = vect;
            _ret = ret;
            _v = v;
        }

        @Override
        public Object call() {
            // setup memory pool for reuse
            try {
                ColGroupValue.setupThreadLocalMemory(_v.getLeft() + 1);
                for(int i = 0; i < _groups.size(); i++) {
                    _groups.get(i)
                        .leftMultByRowVector(_vect.getDenseBlockValues(), _ret.getDenseBlockValues(), _v.getRight()[i]);
                }

                ColGroupValue.cleanupThreadLocalMemory();
            }
            catch(Exception e) {
                throw new DMLRuntimeException(e);
            }
            return null;
        }
    }

    private static class LeftMatrixMatrixMultTask implements Callable<Object> {
        private final List<ColGroup> _group;
        private final double[] _that;
        private final double[] _ret;
        private final int _numRows;
        private final int _numCols;
        private final int _rl;
        private final int _ru;
        private final int _vOff;
        private final Pair<Integer, int[]> _v;

        protected LeftMatrixMatrixMultTask(List<ColGroup> group, double[] that, double[] ret, int numRows, int numCols,
            int rl, int ru, int vOff, Pair<Integer, int[]> v) {
            _group = group;
            _that = that;
            _ret = ret;
            _numRows = numRows;
            _numCols = numCols;
            _rl = rl;
            _ru = ru;
            _vOff = vOff;
            _v = v;
        }

        @Override
        public Object call() {
            // setup memory pool for reuse

            double[][] materialized = new double[_group.size()][];
            for(int i = 0; i < _group.size(); i++) {
                materialized[i] = _group.get(i).getValues();
            }
            // Pair<Integer, int[]> v = getMaxNumValues(_group);
            try {
                ColGroupValue.setupThreadLocalMemory(_v.getLeft() + 1);
                for(int j = 0; j < _group.size(); j++) {
                    _group.get(j).leftMultByMatrix(_that, _ret, materialized[j], _numRows, _numCols, _rl, _ru, _vOff);
                }
                ColGroupValue.cleanupThreadLocalMemory();

            }
            catch(Exception e) {
                throw new DMLRuntimeException(e);
            }
            return null;
        }
    }

    private static class LeftMatrixSparseMatrixMultTask implements Callable<Object> {
        private final List<ColGroup> _group;
        private final SparseBlock _that;
        private final double[] _ret;
        private final int _numRows;
        private final int _numCols;
        private final Pair<Integer, int[]> _v;

        protected LeftMatrixSparseMatrixMultTask(List<ColGroup> group, SparseBlock that, double[] ret, int numRows,
            int numCols, Pair<Integer, int[]> v) {
            _group = group;
            _that = that;
            _ret = ret;
            _numRows = numRows;
            _numCols = numCols;
            _v = v;
        }

        @Override
        public Object call() {
            // setup memory pool for reuse

            // double[][] materialized = new double[_group.size()][];
            // for(int i = 0; i < _group.size(); i++) {
            // materialized[i] = _group.get(i).getValues();
            // }

            boolean containsOLE = false;
            for(int j = 0; j < _group.size(); j++) {
                if(_group.get(j) instanceof ColGroupOLE) {
                    containsOLE = true;
                }
            }
            // Temporary Array to store 2 * block size in
            double[] tmpA = containsOLE ? new double[CompressionSettings.BITMAP_BLOCK_SZ * 2] : null;

            ColGroupValue.setupThreadLocalMemory(_v.getLeft());
            try {
                for(int j = 0; j < _group.size(); j++) {
                    double[] materializedV = _group.get(j).getValues();
                    for(int r = 0; r < _that.numRows(); r++) {
                        if(_that.get(r) != null) {
                            _group.get(j).leftMultBySparseMatrix(_that.get(r).size(),
                                _that.get(r).indexes(),
                                _that.get(r).values(),
                                _ret,
                                _v.getRight()[j],
                                materializedV,
                                _numRows,
                                _numCols,
                                r,
                                tmpA);
                        }
                    }
                }
            }
            catch(Exception e) {
                e.printStackTrace();
                throw new DMLRuntimeException(e);
            }
            ColGroupValue.cleanupThreadLocalMemory();
            return null;
        }
    }

    private static class MatrixMultTransposeTask implements Callable<Object> {
        private final List<ColGroup> _groups;
        private final MatrixBlock _ret;
        private final int _gl;
        private final int _gu;
        private final Pair<Integer, int[]> _v;
        private final boolean _overlapping;

        protected MatrixMultTransposeTask(List<ColGroup> groups, MatrixBlock ret, int gl, int gu,
            Pair<Integer, int[]> v, boolean overlapping) {
            _groups = groups;
            _ret = ret;
            _gl = gl;
            _gu = gu;
            _v = v;
            _overlapping = overlapping;
        }

        @Override
        public Object call() {
            leftMultByTransposeSelf(_groups, _ret, _gl, _gu, _v, _overlapping);
            return null;
        }
    }
}
