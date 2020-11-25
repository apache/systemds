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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Mean;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class LibCompAgg {

    // private static final Log LOG = LogFactory.getLog(LibCompAgg.class.getName());

    /** Threshold for when to parallelize the aggregation functions. */
    private static final long MIN_PAR_AGG_THRESHOLD = 8 * 1024 * 1024; // 8MB

    /** Thread pool matrix Block for materializing decompressed groups. */
    private static ThreadLocal<MatrixBlock> memPool = new ThreadLocal<MatrixBlock>() {
        @Override
        protected MatrixBlock initialValue() {
            return null;
        }
    };

    public static MatrixBlock aggregateUnary(CompressedMatrixBlock m1, MatrixBlock ret, AggregateUnaryOperator op,
        int blen, MatrixIndexes indexesIn, boolean inCP) {

        fillStart(ret, op);

        // core unary aggregate
        if(op.getNumThreads() > 1 && m1.getExactSizeOnDisk() > MIN_PAR_AGG_THRESHOLD) {
            // multi-threaded execution of all groups
            ArrayList<ColGroup>[] grpParts = createStaticTaskPartitioning(m1.getColGroups(),
                (op.indexFn instanceof ReduceCol) ? 1 : op.getNumThreads(),
                false);

            ColGroupUncompressed uc = m1.getUncompressedColGroup();

            try {
                // compute uncompressed column group in parallel (otherwise bottleneck)
                if(uc != null)
                    uc.unaryAggregateOperations(op, ret);
                // compute all compressed column groups
                ExecutorService pool = CommonThreadPool.get(op.getNumThreads());
                ArrayList<UnaryAggregateTask> tasks = new ArrayList<>();
                if(op.indexFn instanceof ReduceCol && grpParts.length > 0) {
                    final int blkz = CompressionSettings.BITMAP_BLOCK_SZ;
                    int blklen = (int) Math.ceil((double) m1.getNumRows() / op.getNumThreads());
                    blklen += (blklen % blkz != 0) ? blkz - blklen % blkz : 0;
                    for(int i = 0; i < op.getNumThreads() & i * blklen < m1.getNumRows(); i++) {
                        tasks.add(new UnaryAggregateTask(grpParts[0], ret, i * blklen,
                            Math.min((i + 1) * blklen, m1.getNumRows()), op));

                    }
                }
                else
                    for(ArrayList<ColGroup> grp : grpParts) {
                        if(grp != null)
                            tasks.add(new UnaryAggregateTask(grp, ret, 0, m1.getNumRows(), op));
                    }
                List<Future<MatrixBlock>> rtasks = pool.invokeAll(tasks);
                pool.shutdown();

                // aggregate partial results
                if(op.indexFn instanceof ReduceAll) {
                    if(op.aggOp.increOp.fn instanceof KahanFunction) {
                        KahanObject kbuff = new KahanObject(ret.quickGetValue(0, 0), 0);
                        for(Future<MatrixBlock> rtask : rtasks) {
                            double tmp = rtask.get().quickGetValue(0, 0);
                            ((KahanFunction) op.aggOp.increOp.fn).execute2(kbuff, tmp);
                        }
                        ret.quickSetValue(0, 0, kbuff._sum);
                    }
                    else if(op.aggOp.increOp.fn instanceof Mean) {
                        double val = ret.quickGetValue(0, 0);
                        for(Future<MatrixBlock> rtask : rtasks) {
                            double tmp = rtask.get().quickGetValue(0, 0);
                            val = val + tmp;
                        }
                        ret.quickSetValue(0, 0, val);
                    }
                    else {
                        double val = ret.quickGetValue(0, 0);
                        for(Future<MatrixBlock> rtask : rtasks) {
                            double tmp = rtask.get().quickGetValue(0, 0);
                            val = op.aggOp.increOp.fn.execute(val, tmp);
                        }
                        ret.quickSetValue(0, 0, val);
                    }
                }
            }
            catch(InterruptedException | ExecutionException e) {
                throw new DMLRuntimeException(e);
            }
        }
        else {
            if(m1.getColGroups() != null) {

                for(ColGroup grp : m1.getColGroups())
                    if(grp instanceof ColGroupUncompressed)
                        ((ColGroupUncompressed) grp).unaryAggregateOperations(op, ret);
                aggregateUnaryOperations(op, m1.getColGroups(), ret, 0, m1.getNumRows());
            }
        }

        // special handling zeros for rowmins/rowmax
        if(op.indexFn instanceof ReduceCol && op.aggOp.increOp.fn instanceof Builtin) {
            int[] rnnz = new int[m1.getNumRows()];
            for(ColGroup grp : m1.getColGroups())
                grp.countNonZerosPerRow(rnnz, 0, m1.getNumRows());
            Builtin builtin = (Builtin) op.aggOp.increOp.fn;
            for(int i = 0; i < m1.getNumRows(); i++)
                if(rnnz[i] < m1.getNumColumns())
                    ret.quickSetValue(i, 0, builtin.execute(ret.quickGetValue(i, 0), 0));
        }

        // special handling of mean
        if(op.aggOp.increOp.fn instanceof Mean) {
            if(op.indexFn instanceof ReduceAll) {
                ret.quickSetValue(0, 0, ret.quickGetValue(0, 0) / (m1.getNumColumns() * m1.getNumRows()));
            }
            else if(op.indexFn instanceof ReduceCol) {
                for(int i = 0; i < m1.getNumRows(); i++) {
                    ret.quickSetValue(i, 0, ret.quickGetValue(i, 0) / m1.getNumColumns());
                }
            }
            else if(op.indexFn instanceof ReduceRow)
                for(int i = 0; i < m1.getNumColumns(); i++) {
                    ret.quickSetValue(0, i, ret.quickGetValue(0, i) / m1.getNumRows());
                }
        }

        // drop correction if necessary
        if(op.aggOp.existsCorrection() && inCP)
            ret.dropLastRowsOrColumns(op.aggOp.correction);

        ret.recomputeNonZeros();
        return ret;
    }

    public static MatrixBlock aggregateUnaryOverlapping(CompressedMatrixBlock m1, MatrixBlock ret,
        AggregateUnaryOperator op, int blen, MatrixIndexes indexesIn, boolean inCP) {

        if(!(op.aggOp.increOp.fn instanceof KahanPlusSq || (op.aggOp.increOp.fn instanceof Builtin &&
            (((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MIN ||
                ((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MAX)))) {
            throw new DMLRuntimeException("Overlapping aggregates is not valid for op: " + op.aggOp.increOp.fn);
        }

        fillStart(ret, op);

        try {
            // compute all compressed column groups
            ExecutorService pool = CommonThreadPool.get(op.getNumThreads());
            ArrayList<UnaryAggregateOverlappingTask> tasks = new ArrayList<>();
            final int blklen = Math.min(m1.getNumRows() /op.getNumThreads(), CompressionSettings.BITMAP_BLOCK_SZ) ;
            // final int blklen = CompressionSettings.BITMAP_BLOCK_SZ ;/// m1.getNumColumns();

            for(int i = 0; i * blklen < m1.getNumRows(); i++) {
                tasks.add(new UnaryAggregateOverlappingTask(m1.getColGroups(), ret, i * blklen,
                    Math.min((i + 1) * blklen, m1.getNumRows()), op));
            }

            List<Future<MatrixBlock>> rtasks = pool.invokeAll(tasks);
            pool.shutdown();

            if(op.indexFn instanceof ReduceAll || (ret.getNumColumns() == 1 && ret.getNumRows() == 1)) {
                if(op.aggOp.increOp.fn instanceof KahanFunction) {
                    KahanObject kbuff = new KahanObject(ret.quickGetValue(0, 0), 0);
                    KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
                    for(Future<MatrixBlock> rtask : rtasks) {
                        double tmp = rtask.get().quickGetValue(0, 0);
                        kplus.execute2(kbuff, tmp);
                    }
                    ret.quickSetValue(0, 0, kbuff._sum);
                }
                else {
                    double val = ret.quickGetValue(0, 0);
                    for(Future<MatrixBlock> rtask : rtasks) {
                        double tmp = rtask.get().quickGetValue(0, 0);
                        val = op.aggOp.increOp.fn.execute(val, tmp);
                    }
                    ret.quickSetValue(0, 0, val);
                }

                ret.recomputeNonZeros();
            }
            else if(op.indexFn instanceof ReduceCol) {
                long nnz = 0;
                for(int i = 0; i * blklen < m1.getNumRows(); i++) {
                    MatrixBlock tmp = rtasks.get(i).get();
                    for(int row = 0, off = i * blklen; row < tmp.getNumRows(); row++, off++) {
                        ret.quickSetValue(off, 0, tmp.quickGetValue(row, 0));
                        nnz += ret.quickGetValue(off, 0) == 0 ? 0 : 1;
                    }
                }
                ret.setNonZeros(nnz);
            }
            else {
                for(Future<MatrixBlock> rtask : rtasks) {
                    LibMatrixBincell.bincellOp(rtask.get(),
                        ret,
                        ret,
                        (op.aggOp.increOp.fn instanceof KahanFunction) ? new BinaryOperator(
                            Plus.getPlusFnObject()) : op.aggOp.increOp);
                }
            }
            memPool.remove();
        }
        catch(InterruptedException | ExecutionException e) {
            throw new DMLRuntimeException(e);
        }
        if(op.aggOp.existsCorrection() && inCP)
            ret.dropLastRowsOrColumns(op.aggOp.correction);

        return ret;
    }

    @SuppressWarnings("unchecked")
    private static ArrayList<ColGroup>[] createStaticTaskPartitioning(List<ColGroup> colGroups, int k,
        boolean inclUncompressed) {
        // special case: single uncompressed col group
        if(colGroups.size() == 1 && colGroups.get(0) instanceof ColGroupUncompressed) {
            return new ArrayList[0];
        }

        // initialize round robin col group distribution
        // (static task partitioning to reduce mem requirements/final agg)
        int numTasks = Math.min(k, colGroups.size());
        ArrayList<ColGroup>[] grpParts = new ArrayList[numTasks];
        int pos = 0;
        for(ColGroup grp : colGroups) {
            if(grpParts[pos] == null)
                grpParts[pos] = new ArrayList<>();
            if(inclUncompressed || !(grp instanceof ColGroupUncompressed)) {
                grpParts[pos].add(grp);
                pos = (pos == numTasks - 1) ? 0 : pos + 1;
            }
        }

        return grpParts;
    }

    private static void aggregateUnaryOperations(AggregateUnaryOperator op, List<ColGroup> groups, MatrixBlock ret,
        int rl, int ru) {

        // note: UC group never passed into this function
        double[] c = ret.getDenseBlockValues();
        for(ColGroup grp : groups)
            if(grp != null && !(grp instanceof ColGroupUncompressed))
                grp.unaryAggregateOperations(op, c, rl, ru);

    }

    private static class UnaryAggregateTask implements Callable<MatrixBlock> {
        private final List<ColGroup> _groups;
        private final int _rl;
        private final int _ru;
        private final MatrixBlock _ret;
        private final AggregateUnaryOperator _op;

        protected UnaryAggregateTask(List<ColGroup> groups, MatrixBlock ret, int rl, int ru,
            AggregateUnaryOperator op) {
            _groups = groups;
            _op = op;
            _rl = rl;
            _ru = ru;

            if(_op.indexFn instanceof ReduceAll) { // sum
                _ret = new MatrixBlock(ret.getNumRows(), ret.getNumColumns(), false);
                _ret.allocateDenseBlock();
                if(_op.aggOp.increOp.fn instanceof Builtin)
                    System.arraycopy(ret.getDenseBlockValues(),
                        0,
                        _ret.getDenseBlockValues(),
                        0,
                        ret.getNumRows() * ret.getNumColumns());
            }
            else { // colSums
                _ret = ret;
            }
        }

        @Override
        public MatrixBlock call() {
            aggregateUnaryOperations(_op, _groups, _ret, _rl, _ru);
            return _ret;
        }
    }

    private static class UnaryAggregateOverlappingTask implements Callable<MatrixBlock> {
        private final List<ColGroup> _groups;
        private final int _rl;
        private final int _ru;
        private final MatrixBlock _ret;
        private final AggregateUnaryOperator _op;

        protected UnaryAggregateOverlappingTask(List<ColGroup> groups, MatrixBlock ret, int rl, int ru,
            AggregateUnaryOperator op) {
            _groups = groups;
            _op = op;
            _rl = rl;
            _ru = ru;
            if(_op.indexFn instanceof ReduceAll) {
                _ret = new MatrixBlock(ret.getNumRows(), ret.getNumColumns(), false);
                _ret.allocateDenseBlock();
            }
            else if(_op.indexFn instanceof ReduceCol) {
                _ret = new MatrixBlock(ru - rl, ret.getNumColumns(), false);
                _ret.allocateDenseBlock();
            }
            else {
                _ret = new MatrixBlock(ret.getNumRows(), ret.getNumColumns(), false);
                _ret.allocateDenseBlock();
            }
            if(_op.aggOp.increOp.fn instanceof Builtin) {
                System.arraycopy(ret
                    .getDenseBlockValues(), 0, _ret.getDenseBlockValues(), 0, _ret.getDenseBlockValues().length);
            }

        }

        @Override
        public MatrixBlock call() {
            MatrixBlock tmp = memPool.get();
            if(tmp == null) {
                memPool.set(new MatrixBlock(_ru - _rl, _groups.get(0).getNumCols(), false, -1).allocateBlock());
                tmp = memPool.get();
            }
            else {
                tmp = memPool.get();
                tmp.reset(_ru - _rl, _groups.get(0).getNumCols(), false, -1);
            }

            for(ColGroup g : _groups) {
                g.decompressToBlock(tmp, _rl, _ru, 0, g.getValues());
            }

            LibMatrixAgg.aggregateUnaryMatrix(tmp, _ret, _op);
            return _ret;
        }
    }

    private static void fillStart(MatrixBlock ret, AggregateUnaryOperator op) {
        if(op.aggOp.increOp.fn instanceof Builtin) {
            Double val = null;
            switch(((Builtin) op.aggOp.increOp.fn).getBuiltinCode()) {
                case MAX:
                    val = Double.NEGATIVE_INFINITY;
                    break;
                case MIN:
                    val = Double.POSITIVE_INFINITY;
                    break;
                default:
                    break;
            }
            if(val != null) {
                ret.getDenseBlock().set(val);
            }
        }
    }
}
