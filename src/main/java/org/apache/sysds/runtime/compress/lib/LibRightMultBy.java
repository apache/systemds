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
import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupOLE;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupValue;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class LibRightMultBy {
    private static final Log LOG = LogFactory.getLog(LibRightMultBy.class.getName());

    /**
     * Right multiply by matrix. Meaning a left hand side compressed matrix is multiplied with a right hand side
     * uncompressed matrix.
     * 
     * @param colGroups    All Column groups in the compression
     * @param that         The right hand side matrix
     * @param ret          The MatrixBlock to return.
     * @param k            The parallelization degree to use.
     * @param v            The Precalculated counts and Maximum number of tuple entries in the column groups.
     * @param allowOverlap Allow the multiplication to return an overlapped matrix.
     * @return The Result Matrix, modified from the ret parameter.
     */
    public static MatrixBlock rightMultByMatrix(List<ColGroup> colGroups, MatrixBlock that, MatrixBlock ret, int k,
        Pair<Integer, int[]> v, boolean allowOverlap) {

        boolean containsUncompressable = false;
        int distinctCount = 0;
        for(ColGroup g : colGroups) {
            if(g instanceof ColGroupValue) {
                distinctCount += ((ColGroupValue) g).getNumValues();
            }
            else {
                containsUncompressable = true;
            }
        }
        int rl = colGroups.get(0).getNumRows();
        int cl = that.getNumColumns();
        if(!allowOverlap || (containsUncompressable || distinctCount >= rl / 2)) {
            if(ret == null)
                ret = new MatrixBlock(rl, cl, false, rl * cl);
            else if(!(ret.getNumColumns() == cl && ret.getNumRows() == rl && ret.isAllocated()))
                ret.reset(rl, cl, false, rl * cl);
            ret.allocateDenseBlock();
            if(that.isInSparseFormat()) {
                ret = rightMultBySparseMatrix(colGroups, that, ret, k, v);
            }
            else {
                ret = rightMultByDenseMatrix(colGroups, that, ret, k, v);

            }
            ret.setNonZeros(ret.getNumColumns() * ret.getNumRows());
        }
        else {
            // Create an overlapping compressed Matrix Block.
            ret = new CompressedMatrixBlock(true);

            ret.setNumColumns(cl);
            ret.setNumRows(rl);
            CompressedMatrixBlock retC = (CompressedMatrixBlock) ret;
            retC.setOverlapping(true);
            if(that.isInSparseFormat()) {
                ret = rightMultBySparseMatrixCompressed(colGroups, that, retC, k, v);
            }
            else {
                ret = rightMultByDenseMatrixCompressed(colGroups, that, retC, k, v);
            }
        }

        return ret;

    }

    /**
     * Multi-threaded version of rightMultByVector.
     * 
     * @param colGroups The Column groups used int the multiplication
     * @param vector    matrix block vector to multiply with
     * @param result    matrix block result to modify in the multiplication
     * @param k         number of threads to use
     * @param v         The Precalculated counts and Maximum number of tuple entries in the column groups
     */
    public static void rightMultByVector(List<ColGroup> colGroups, MatrixBlock vector, MatrixBlock result, int k,
        Pair<Integer, int[]> v) {
        // initialize and allocate the result
        result.allocateDenseBlock();
        if(k <= 1) {
            rightMultByVector(colGroups, vector, result, v);
            return;
        }

        // multi-threaded execution of all groups
        try {
            // ColGroupUncompressed uc = getUncompressedColGroup();

            // compute uncompressed column group in parallel
            // if(uc != null)
            // uc.rightMultByVector(vector, result, k);

            // compute remaining compressed column groups in parallel
            // note: OLE needs alignment to segment size, otherwise wrong entry
            ExecutorService pool = CommonThreadPool.get(k);
            int rlen = colGroups.get(0).getNumRows();
            int seqsz = CompressionSettings.BITMAP_BLOCK_SZ;
            int blklen = (int) (Math.ceil((double) rlen / k));
            blklen += (blklen % seqsz != 0) ? seqsz - blklen % seqsz : 0;

            ArrayList<RightMatrixVectorMultTask> tasks = new ArrayList<>();
            for(int i = 0; i < k & i * blklen < rlen; i++) {
                tasks.add(new RightMatrixVectorMultTask(colGroups, vector, result, i * blklen,
                    Math.min((i + 1) * blklen, rlen), v));
            }

            List<Future<Long>> ret = pool.invokeAll(tasks);
            pool.shutdown();

            // error handling and nnz aggregation
            long lnnz = 0;
            for(Future<Long> tmp : ret)
                lnnz += tmp.get();
            result.setNonZeros(lnnz);
        }
        catch(InterruptedException | ExecutionException e) {
            throw new DMLRuntimeException(e);
        }
    }

    /**
     * Multiply this matrix block by a column vector on the right.
     * 
     * @param vector right-hand operand of the multiplication
     * @param result buffer to hold the result; must have the appropriate size already
     * @param v      The Precalculated counts and Maximum number of tuple entries in the column groups.
     */
    private static void rightMultByVector(List<ColGroup> colGroups, MatrixBlock vector, MatrixBlock result,
        Pair<Integer, int[]> v) {

        // delegate matrix-vector operation to each column group
        rightMultByVector(colGroups, vector, result, 0, result.getNumRows(), v);

        // post-processing
        result.recomputeNonZeros();
    }

    private static MatrixBlock rightMultBySparseMatrix(List<ColGroup> colGroups, MatrixBlock that, MatrixBlock ret,
        int k, Pair<Integer, int[]> v) {
        SparseBlock sb = that.getSparseBlock();
        double[] retV = ret.getDenseBlockValues();

        if(sb == null)
            throw new DMLRuntimeException("Invalid Right Mult by Sparse matrix, input matrix was dense");

        for(ColGroup grp : colGroups) {
            if(grp instanceof ColGroupUncompressed)
                ((ColGroupUncompressed) grp).rightMultByMatrix(that, ret, 0, ret.getNumColumns());
        }

        // Pair<Integer, int[]> v = Util.getMaxNumValues(colGroups);
        // if(k == 1) {
        for(int j = 0; j < colGroups.size(); j++) {
            double[] preAggregatedB = ((ColGroupValue) colGroups.get(j)).preaggValues(v.getRight()[j],
                sb,
                colGroups.get(j).getValues(),
                0,
                that.getNumColumns(),
                that.getNumColumns());
            colGroups.get(j).rightMultByMatrix(preAggregatedB,
                retV,
                that.getNumColumns(),
                0,
                ret.getNumRows(),
                0,
                that.getNumColumns());

        }
        // }
        // else {
        // ExecutorService pool = CommonThreadPool.get(k);
        // ArrayList<RightMultBySparseMatrixTask> tasks = new ArrayList<>();
        // try {

        // for(int j = 0; j < ret.getNumColumns(); j += CompressionSettings.BITMAP_BLOCK_SZ) {
        // tasks.add(new RightMultBySparseMatrixTask(colGroups, retV, sb, materialized, v, numColumns, j,
        // Math.min(j + CompressionSettings.BITMAP_BLOCK_SZ, ret.getNumColumns())));
        // }

        // List<Future<Object>> futures = pool.invokeAll(tasks);
        // pool.shutdown();
        // for(Future<Object> future : futures)
        // future.get();
        // }
        // catch(InterruptedException | ExecutionException e) {
        // throw new DMLRuntimeException(e);
        // }
        // }

        return ret;
    }

    private static MatrixBlock rightMultByDenseMatrix(List<ColGroup> colGroups, MatrixBlock that, MatrixBlock ret,
        int k, Pair<Integer, int[]> v) {

        // long StartTime = System.currentTimeMillis();
        DenseBlock db = that.getDenseBlock();
        double[] retV = ret.getDenseBlockValues();
        double[] thatV;

        for(ColGroup grp : colGroups) {
            if(grp instanceof ColGroupUncompressed) {
                ((ColGroupUncompressed) grp).rightMultByMatrix(that, ret, 0, ret.getNumRows());
            }
        }

        if(k == 1) {
            ColGroupValue.setupThreadLocalMemory((v.getLeft()));
            for(int b = 0; b < db.numBlocks(); b++) {
                // int blockSize = db.blockSize(b);
                thatV = db.valuesAt(b);
                for(int j = 0; j < colGroups.size(); j++) {
                    int colBlockSize = 128;
                    for(int i = 0; i < that.getNumColumns(); i += colBlockSize) {
                        if(colGroups.get(j) instanceof ColGroupValue) {
                            double[] preAggregatedB = ((ColGroupValue) colGroups.get(j)).preaggValues(v.getRight()[j],
                                thatV,
                                colGroups.get(j).getValues(),
                                i,
                                Math.min(i + colBlockSize, that.getNumColumns()),
                                that.getNumColumns());
                            int blklenRows = CompressionSettings.BITMAP_BLOCK_SZ;
                            for(int n = 0; n * blklenRows < ret.getNumRows(); n++) {
                                colGroups.get(j).rightMultByMatrix(preAggregatedB,
                                    retV,
                                    that.getNumColumns(),
                                    n * blklenRows,
                                    Math.min((n + 1) * blklenRows, ret.getNumRows()),
                                    i,
                                    Math.min(i + colBlockSize, that.getNumColumns()));
                            }
                        }
                    }
                }
            }
            ColGroupValue.cleanupThreadLocalMemory();
        }
        else {

            thatV = db.valuesAt(0);
            ExecutorService pool = CommonThreadPool.get(k);
            ArrayList<RightMatrixMultTask> tasks = new ArrayList<>();
            ArrayList<RightMatrixPreAggregateTask> preTask = new ArrayList<>(colGroups.size());
            // Pair<Integer, int[]> v;
            final int blkz = CompressionSettings.BITMAP_BLOCK_SZ;
            int blklenRows = (int) (Math.ceil((double) ret.getNumRows() / (2 * k)));

            try {
                List<Future<double[]>> ag = pool.invokeAll(preAggregate(colGroups, thatV, that, preTask, v));
                // DDC and RLE
                for(int j = 0; j * blklenRows < ret.getNumRows(); j++) {
                    RightMatrixMultTask rmmt = new RightMatrixMultTask(colGroups, retV, ag, v, that.getNumColumns(),
                        j * blklenRows, Math.min((j + 1) * blklenRows, ret.getNumRows()), 0, that.getNumColumns(),
                        false, false);
                    tasks.add(rmmt);
                }
                blklenRows += (blklenRows % blkz != 0) ? blkz - blklenRows % blkz : 0;
                // OLE!
                for(int j = 0; j * blklenRows < ret.getNumRows(); j++) {
                    RightMatrixMultTask rmmt = new RightMatrixMultTask(colGroups, retV, ag, v, that.getNumColumns(),
                        j * blklenRows, Math.min((j + 1) * blklenRows, ret.getNumRows()), 0, that.getNumColumns(),
                        false, true);
                    tasks.add(rmmt);
                }
                for(Future<Object> future : pool.invokeAll(tasks))
                    future.get();
                tasks.clear();

            }
            catch(InterruptedException | ExecutionException e) {
                throw new DMLRuntimeException(e);
            }
        }

        return ret;
    }

    private static MatrixBlock rightMultByDenseMatrixCompressed(List<ColGroup> colGroups, MatrixBlock that,
        CompressedMatrixBlock ret, int k, Pair<Integer, int[]> v) {

        DenseBlock db = that.getDenseBlock();
        double[] thatV;

        for(ColGroup grp : colGroups) {
            if(grp instanceof ColGroupUncompressed) {
                throw new DMLCompressionException(
                    "Right Mult by dense with compressed output is not efficient to do with uncompressed Compressed ColGroups and therefore not supported.");
            }
        }

        thatV = db.valuesAt(0);
        List<ColGroup> retCg = new ArrayList<ColGroup>();
        int[] newColIndexes = new int[that.getNumColumns()];
        for(int i = 0; i < that.getNumColumns(); i++) {
            newColIndexes[i] = i;
        }
        if(k == 1) {
            for(int j = 0; j < colGroups.size(); j++) {
                ColGroupValue g = (ColGroupValue) colGroups.get(j);
                double[] preAggregatedB = g.preaggValues(v.getRight()[j],
                    thatV,
                    g.getValues(),
                    0,
                    that.getNumColumns(),
                    that.getNumColumns(),
                    new double[v.getRight()[j] * that.getNumColumns()]);
                retCg.add(g.copyAndSet(newColIndexes, preAggregatedB));
            }
        }
        else {
            thatV = db.valuesAt(0);
            ExecutorService pool = CommonThreadPool.get(k);
            ArrayList<RightMatrixPreAggregateTask> preTask = new ArrayList<>(colGroups.size());

            try {
                List<Future<double[]>> ag = pool.invokeAll(preAggregate(colGroups, thatV, that, preTask, v));
                for(int j = 0; j < colGroups.size(); j++) {
                    retCg.add(((ColGroupValue) colGroups.get(j)).copyAndSet(newColIndexes, ag.get(j).get()));
                }
            }
            catch(InterruptedException | ExecutionException e) {
                throw new DMLRuntimeException(e);
            }
        }
        ret.allocateColGroupList(retCg);
        ret.setOverlapping(true);
        ret.setNonZeros(-1);

        return ret;
    }

    private static MatrixBlock rightMultBySparseMatrixCompressed(List<ColGroup> colGroups, MatrixBlock that,
        CompressedMatrixBlock ret, int k, Pair<Integer, int[]> v) {

        // long StartTime = System.currentTimeMillis();
        SparseBlock sb = that.getSparseBlock();

        for(ColGroup grp : colGroups) {
            if(grp instanceof ColGroupUncompressed) {
                throw new DMLCompressionException(
                    "Right Mult by dense with compressed output is not efficient to do with uncompressed Compressed ColGroups and therefore not supported.");
            }
        }

        List<ColGroup> retCg = new ArrayList<ColGroup>();
        int[] newColIndexes = new int[that.getNumColumns()];
        for(int i = 0; i < that.getNumColumns(); i++) {
            newColIndexes[i] = i;
        }
        if(k == 1) {
            for(int j = 0; j < colGroups.size(); j++) {
                ColGroupValue g = (ColGroupValue) colGroups.get(j);
                double[] preAggregatedB = g.preaggValues(v.getRight()[j],
                    sb,
                    colGroups.get(j).getValues(),
                    0,
                    that.getNumColumns(),
                    that.getNumColumns(),
                    new double[v.getRight()[j] * that.getNumColumns()]);
                retCg.add(g.copyAndSet(newColIndexes, preAggregatedB));
            }
        }
        else {
            ExecutorService pool = CommonThreadPool.get(k);
            ArrayList<RightMatrixPreAggregateSparseTask> preTask = new ArrayList<>(colGroups.size());

            try {
                List<Future<double[]>> ag = pool.invokeAll(preAggregate(colGroups, sb, that, preTask, v));
                for(int j = 0; j < colGroups.size(); j++) {
                    retCg.add(((ColGroupValue) colGroups.get(j)).copyAndSet(newColIndexes, ag.get(j).get()));
                }
            }
            catch(InterruptedException | ExecutionException e) {
                throw new DMLRuntimeException(e);
            }
        }
        ret.allocateColGroupList(retCg);
        ret.setOverlapping(true);
        ret.setNonZeros(-1);

        return ret;
    }

    private static ArrayList<RightMatrixPreAggregateTask> preAggregate(List<ColGroup> colGroups, double[] thatV,
        MatrixBlock that, ArrayList<RightMatrixPreAggregateTask> preTask, Pair<Integer, int[]> v) {
        preTask.clear();
        for(int h = 0; h < colGroups.size(); h++) {
            RightMatrixPreAggregateTask pAggT = new RightMatrixPreAggregateTask((ColGroupValue) colGroups.get(h),
                v.getRight()[h], thatV, colGroups.get(h).getValues(), 0, that.getNumColumns(), that.getNumColumns());
            preTask.add(pAggT);
        }
        return preTask;
    }

    private static ArrayList<RightMatrixPreAggregateSparseTask> preAggregate(List<ColGroup> colGroups, SparseBlock sb,
        MatrixBlock that, ArrayList<RightMatrixPreAggregateSparseTask> preTask, Pair<Integer, int[]> v) {
        preTask.clear();
        for(int h = 0; h < colGroups.size(); h++) {
            RightMatrixPreAggregateSparseTask pAggT = new RightMatrixPreAggregateSparseTask(
                (ColGroupValue) colGroups.get(h), v.getRight()[h], sb, colGroups.get(h).getValues(), 0,
                that.getNumColumns(), that.getNumColumns());
            preTask.add(pAggT);
        }
        return preTask;
    }

    private static void rightMultByVector(List<ColGroup> groups, MatrixBlock vect, MatrixBlock ret, int rl, int ru,
        Pair<Integer, int[]> v) {
        // + 1 to enable containing a single 0 value in the dictionary that was not materialized.
        // This is to handle the case of a DDC dictionary not materializing the zero values.
        // A fine tradeoff!
        ColGroupValue.setupThreadLocalMemory(v.getLeft() + 1);

        // boolean cacheDDC1 = ru - rl > CompressionSettings.BITMAP_BLOCK_SZ * 2;

        // process uncompressed column group (overwrites output)
        // if(inclUC) {
        for(ColGroup grp : groups) {
            if(grp instanceof ColGroupUncompressed)
                ((ColGroupUncompressed) grp).rightMultByVector(vect, ret, rl, ru);
        }

        // process cache-conscious DDC1 groups (adds to output)

        // if(cacheDDC1) {
        // ArrayList<ColGroupDDC1> tmp = new ArrayList<>();
        // for(ColGroup grp : groups)
        // if(grp instanceof ColGroupDDC1)
        // tmp.add((ColGroupDDC1) grp);
        // if(!tmp.isEmpty())
        // ColGroupDDC1.rightMultByVector(tmp.toArray(new ColGroupDDC1[0]), vect, ret, rl, ru);
        // }
        // process remaining groups (adds to output)
        double[] values = ret.getDenseBlockValues();
        for(ColGroup grp : groups) {
            if(!(grp instanceof ColGroupUncompressed)) {
                grp.rightMultByVector(vect.getDenseBlockValues(), values, rl, ru, grp.getValues());
            }
        }

        ColGroupValue.cleanupThreadLocalMemory();

    }

    private static class RightMatrixMultTask implements Callable<Object> {
        private final List<ColGroup> _colGroups;
        // private final double[] _thatV;
        private final double[] _retV;
        private final List<Future<double[]>> _aggB;
        private final Pair<Integer, int[]> _v;
        private final int _numColumns;

        private final int _rl;
        private final int _ru;
        private final int _cl;
        private final int _cu;
        private final boolean _mem;
        private final boolean _skipOle;

        protected RightMatrixMultTask(List<ColGroup> groups, double[] retV, List<Future<double[]>> aggB,
            Pair<Integer, int[]> v, int numColumns, int rl, int ru, int cl, int cu, boolean mem, boolean skipOle) {
            _colGroups = groups;
            // _thatV = thatV;
            _retV = retV;
            _aggB = aggB;
            _v = v;
            _numColumns = numColumns;
            _rl = rl;
            _ru = ru;
            _cl = cl;
            _cu = cu;
            _mem = mem;
            _skipOle = skipOle;
        }

        @Override
        public Object call() {
            try {
                if(_mem)
                    ColGroupValue.setupThreadLocalMemory((_v.getLeft()));
                for(int j = 0; j < _colGroups.size(); j++) {
                    if(_colGroups.get(j) instanceof ColGroupOLE) {
                        if(_skipOle) {
                            _colGroups.get(j)
                                .rightMultByMatrix(_aggB.get(j).get(), _retV, _numColumns, _rl, _ru, _cl, _cu);
                        }
                    }
                    else {
                        if(!_skipOle) {
                            _colGroups.get(j)
                                .rightMultByMatrix(_aggB.get(j).get(), _retV, _numColumns, _rl, _ru, _cl, _cu);
                        }
                    }
                }
                if(_mem)
                    ColGroupValue.cleanupThreadLocalMemory();
                return null;
            }
            catch(Exception e) {
                LOG.error(e);
                throw new DMLRuntimeException(e);
            }
        }
    }

    private static class RightMatrixPreAggregateTask implements Callable<double[]> {
        private final ColGroupValue _colGroup;
        private final int _numVals;
        private final double[] _b;
        private final double[] _dict;

        private final int _cl;
        private final int _cu;
        private final int _cut;

        protected RightMatrixPreAggregateTask(ColGroupValue colGroup, int numVals, double[] b, double[] dict, int cl,
            int cu, int cut) {
            _colGroup = colGroup;
            _numVals = numVals;
            _b = b;
            _dict = dict;
            _cl = cl;
            _cu = cu;
            _cut = cut;
        }

        @Override
        public double[] call() {
            try {
                return _colGroup.preaggValues(_numVals, _b, _dict, _cl, _cu, _cut);
            }
            catch(Exception e) {
                LOG.error(e);
                throw new DMLRuntimeException(e);
            }
        }
    }

    private static class RightMatrixPreAggregateSparseTask implements Callable<double[]> {
        private final ColGroupValue _colGroup;
        private final int _numVals;
        private final SparseBlock _b;
        private final double[] _dict;

        private final int _cl;
        private final int _cu;
        private final int _cut;

        protected RightMatrixPreAggregateSparseTask(ColGroupValue colGroup, int numVals, SparseBlock b, double[] dict,
            int cl, int cu, int cut) {
            _colGroup = colGroup;
            _numVals = numVals;
            _b = b;
            _dict = dict;
            _cl = cl;
            _cu = cu;
            _cut = cut;
        }

        @Override
        public double[] call() {
            try {
                return _colGroup.preaggValues(_numVals, _b, _dict, _cl, _cu, _cut);
            }
            catch(Exception e) {
                LOG.error(e);
                throw new DMLRuntimeException(e);
            }
        }
    }

    private static class RightMatrixVectorMultTask implements Callable<Long> {
        private final List<ColGroup> _groups;
        private final MatrixBlock _vect;
        private final MatrixBlock _ret;
        private final int _rl;
        private final int _ru;
        private final Pair<Integer, int[]> _v;

        protected RightMatrixVectorMultTask(List<ColGroup> groups, MatrixBlock vect, MatrixBlock ret, int rl, int ru,
            Pair<Integer, int[]> v) {
            _groups = groups;
            _vect = vect;
            _ret = ret;
            _rl = rl;
            _ru = ru;
            _v = v;
        }

        @Override
        public Long call() {
            try {
                rightMultByVector(_groups, _vect, _ret, _rl, _ru, _v);
                return _ret.recomputeNonZeros(_rl, _ru - 1, 0, 0);
            }
            catch(Exception e) {
                LOG.error(e);
                throw new DMLRuntimeException(e);
            }
        }
    }
}
