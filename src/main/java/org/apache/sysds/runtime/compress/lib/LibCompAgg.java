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
import org.apache.sysds.runtime.functionobjects.IndexFunction;
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

    private static final long MIN_PAR_AGG_THRESHOLD = 8 * 1024 * 1024;

    private static ThreadLocal<MatrixBlock> memPool = new ThreadLocal<MatrixBlock>() {
        @Override
        protected MatrixBlock initialValue() {
            return null;
        }
    };

    public static MatrixBlock aggregateUnary(CompressedMatrixBlock inputMatrix, MatrixBlock outputMatrix,
        AggregateUnaryOperator op, int blen, MatrixIndexes indexesIn, boolean inCP) {

        if(inputMatrix.getColGroups() != null) {
            fillStart(outputMatrix, op);

            if(inputMatrix.isOverlapping() &&
                (op.aggOp.increOp.fn instanceof KahanPlusSq || (op.aggOp.increOp.fn instanceof Builtin &&
                    (((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MIN ||
                        ((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MAX))))
                aggregateUnaryOverlapping(inputMatrix, outputMatrix, op, indexesIn, inCP);
            else
                aggregateUnaryNormalCompressedMatrixBlock(inputMatrix, outputMatrix, op, blen, indexesIn, inCP);
        }

        if(op.aggOp.existsCorrection() && inCP)
            outputMatrix.dropLastRowsOrColumns(op.aggOp.correction);

        outputMatrix.recomputeNonZeros();
        memPool.remove();

        return outputMatrix;
    }

    private static void aggregateUnaryNormalCompressedMatrixBlock(CompressedMatrixBlock inputMatrix,
        MatrixBlock outputMatrix, AggregateUnaryOperator op, int blen, MatrixIndexes indexesIn, boolean inCP) {

        aggregateUncompressedColGroups(inputMatrix, outputMatrix, op);

        if(isValidForParallelProcessing(inputMatrix, op))
            tryToAggregateInParallel(inputMatrix, outputMatrix, op);
        else
            aggregateSingleThreaded(inputMatrix, outputMatrix, op);

        postProcessAggregate(inputMatrix, outputMatrix, op);

    }

    private static boolean isValidForParallelProcessing(CompressedMatrixBlock m1, AggregateUnaryOperator op) {
        return op.getNumThreads() > 1 && m1.getExactSizeOnDisk() > MIN_PAR_AGG_THRESHOLD;
    }

    private static void aggregateUncompressedColGroups(CompressedMatrixBlock m1, MatrixBlock ret,
        AggregateUnaryOperator op) {
        ColGroupUncompressed uc = m1.getUncompressedColGroup();
        if(uc != null)
            uc.unaryAggregateOperations(op, ret);
    }

    private static void tryToAggregateInParallel(CompressedMatrixBlock m1, MatrixBlock ret, AggregateUnaryOperator op) {
        int k = getNumberOfThreadsToUse(op);
        if(k == 1)
            aggregateUnaryOperations(op, m1.getColGroups(), ret, 0, m1.getNumRows(), m1.getNumColumns());
        else
            aggregateInParallel(m1, ret, op, k);

    }

    private static void aggregateInParallel(CompressedMatrixBlock m1, MatrixBlock ret, AggregateUnaryOperator op,
        int k) {

        List<List<ColGroup>> grpParts = createTaskPartitionNotIncludingUncompressable(m1.getColGroups(), k);

        ExecutorService pool = CommonThreadPool.get(op.getNumThreads());
        ArrayList<UnaryAggregateTask> tasks = new ArrayList<>();

        try {
            // compute all compressed column groups
            if(op.indexFn instanceof ReduceCol && grpParts.size() > 0) {
                final int blkz = CompressionSettings.BITMAP_BLOCK_SZ;
                int blklen = Math.min((int) Math.ceil((double) m1.getNumRows() / op.getNumThreads()), blkz / 2);
                blklen += (blklen % blkz != 0) ? blkz - blklen % blkz : 0;
                for(int i = 0; i < op.getNumThreads() & i * blklen < m1.getNumRows(); i++) {
                    tasks.add(new UnaryAggregateTask(grpParts.get(0), ret, i * blklen,
                        Math.min((i + 1) * blklen, m1.getNumRows()), op, m1.getNumColumns()));

                }
            }
            else
                for(List<ColGroup> grp : grpParts) {
                    if(grp != null)
                        tasks.add(new UnaryAggregateTask(grp, ret, 0, m1.getNumRows(), op, m1.getNumColumns()));
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

    private static void aggregateSingleThreaded(CompressedMatrixBlock m1, MatrixBlock ret, AggregateUnaryOperator op) {
        aggregateUnaryOperations(op, m1.getColGroups(), ret, 0, m1.getNumRows(), m1.getNumColumns());
    }

    private static void divideByNumberOfCellsForMean(CompressedMatrixBlock m1, MatrixBlock ret, IndexFunction idxFn) {
        if(idxFn instanceof ReduceAll)
            divideByNumberOfCellsForMeanAll(m1, ret);
        else if(idxFn instanceof ReduceCol)
            divideByNumberOfCellsForMeanRows(m1, ret);
        else if(idxFn instanceof ReduceRow)
            divideByNumberOfCellsForMeanCols(m1, ret);
    }

    private static void divideByNumberOfCellsForMeanRows(CompressedMatrixBlock m1, MatrixBlock ret) {
        for(int i = 0; i < m1.getNumRows(); i++) {
            ret.quickSetValue(i, 0, ret.quickGetValue(i, 0) / m1.getNumColumns());
        }
    }

    private static void divideByNumberOfCellsForMeanCols(CompressedMatrixBlock m1, MatrixBlock ret) {
        for(int i = 0; i < m1.getNumColumns(); i++) {
            ret.quickSetValue(0, i, ret.quickGetValue(0, i) / m1.getNumRows());
        }
    }

    private static void divideByNumberOfCellsForMeanAll(CompressedMatrixBlock m1, MatrixBlock ret) {
        ret.quickSetValue(0, 0, ret.quickGetValue(0, 0) / (m1.getNumColumns() * m1.getNumRows()));
    }

    private static void postProcessAggregate(CompressedMatrixBlock m1, MatrixBlock ret, AggregateUnaryOperator op) {
        if(op.aggOp.increOp.fn instanceof Mean)
            divideByNumberOfCellsForMean(m1, ret, op.indexFn);

    }

    private static void aggregateUnaryOverlapping(CompressedMatrixBlock m1, MatrixBlock ret, AggregateUnaryOperator op,
        MatrixIndexes indexesIn, boolean inCP) {

        try {
            List<Future<MatrixBlock>> rtasks = generateUnaryAggregateOverlappingFutures(m1, ret, op);
            reduceOverlappingFutures(rtasks, ret, op);
        }
        catch(InterruptedException | ExecutionException e) {
            throw new DMLRuntimeException(e);
        }

    }

    private static void reduceOverlappingFutures(List<Future<MatrixBlock>> rtasks, MatrixBlock ret,
        AggregateUnaryOperator op) throws InterruptedException, ExecutionException {
        if(isReduceAll(ret, op.indexFn))
            reduceAllOverlappingFutures(rtasks, ret, op);
        else if(op.indexFn instanceof ReduceRow)
            reduceColOverlappingFutures(rtasks, ret, op);
        else
            reduceRowOverlappingFutures(rtasks, ret, op);
    }

    private static void reduceColOverlappingFutures(List<Future<MatrixBlock>> rtasks, MatrixBlock ret,
        AggregateUnaryOperator op) throws InterruptedException, ExecutionException {
        for(Future<MatrixBlock> rtask : rtasks) {
            LibMatrixBincell.bincellOp(rtask.get(),
                ret,
                ret,
                (op.aggOp.increOp.fn instanceof KahanFunction) ? new BinaryOperator(
                    Plus.getPlusFnObject()) : op.aggOp.increOp);
        }
    }

    private static void reduceRowOverlappingFutures(List<Future<MatrixBlock>> rtasks, MatrixBlock ret,
        AggregateUnaryOperator op) throws InterruptedException, ExecutionException {
        for(Future<MatrixBlock> rtask : rtasks) {
            rtask.get();
        }
    }

    private static boolean isReduceAll(MatrixBlock ret, IndexFunction idxFn) {
        return idxFn instanceof ReduceAll || (ret.getNumColumns() == 1 && ret.getNumRows() == 1);
    }

    private static void reduceAllOverlappingFutures(List<Future<MatrixBlock>> rtasks, MatrixBlock ret,
        AggregateUnaryOperator op) throws InterruptedException, ExecutionException {

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
    }

    private static List<Future<MatrixBlock>> generateUnaryAggregateOverlappingFutures(CompressedMatrixBlock m1,
        MatrixBlock ret, AggregateUnaryOperator op) throws InterruptedException {

        ExecutorService pool = CommonThreadPool.get(op.getNumThreads());
        ArrayList<UnaryAggregateOverlappingTask> tasks = new ArrayList<>();

        final int blklen = Math.min(m1.getNumRows() / op.getNumThreads(), CompressionSettings.BITMAP_BLOCK_SZ);

        for(int i = 0; i * blklen < m1.getNumRows(); i++)
            tasks.add(new UnaryAggregateOverlappingTask(m1, ret, i * blklen,
                Math.min((i + 1) * blklen, m1.getNumRows()), op));

        List<Future<MatrixBlock>> futures = pool.invokeAll(tasks);
        pool.shutdown();
        return futures;
    }

    private static List<List<ColGroup>> createTaskPartitionNotIncludingUncompressable(List<ColGroup> colGroups, int k) {
        int numTasks = Math.min(k, colGroups.size());
        List<List<ColGroup>> grpParts = new ArrayList<>();
        for(int i = 0; i < numTasks; i++) {
            grpParts.add(new ArrayList<>());
        }
        int pos = 0;
        for(ColGroup grp : colGroups) {
            if(!(grp instanceof ColGroupUncompressed)) {
                List<ColGroup> g = grpParts.get(pos);
                g.add(grp);
                pos = (pos + 1) % numTasks;
            }
        }

        return grpParts;
    }

    private static int getNumberOfThreadsToUse(AggregateUnaryOperator op) {
        return (op.indexFn instanceof ReduceCol) ? 1 : op.getNumThreads();
    }

    private static void aggregateUnaryOperations(AggregateUnaryOperator op, List<ColGroup> groups, MatrixBlock ret,
        int rl, int ru, int numColumns) {

        int[] rnnz = (op.indexFn instanceof ReduceCol && op.aggOp.increOp.fn instanceof Builtin) ? new int[ru -
            rl] : null;
        int numberDenseColumns = 0;
        for(ColGroup grp : groups) {
            if(grp != null && !(grp instanceof ColGroupUncompressed)) {
                grp.unaryAggregateOperations(op, ret, rl, ru);
                if(grp.isDense()) {
                    numberDenseColumns += grp.getNumCols();
                }
                else if(op.indexFn instanceof ReduceCol && op.aggOp.increOp.fn instanceof Builtin) {
                    grp.countNonZerosPerRow(rnnz, rl, ru);
                }
            }
        }

        if(op.indexFn instanceof ReduceCol && op.aggOp.increOp.fn instanceof Builtin) {
            for(int row = rl; row < ru; row++) {
                if(rnnz[row] + numberDenseColumns < numColumns) {
                    ret.quickSetValue(row, 0, op.aggOp.increOp.fn.execute(ret.quickGetValue(row, 0), 0.0));
                }
            }

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

    private static class UnaryAggregateTask implements Callable<MatrixBlock> {
        private final List<ColGroup> _groups;
        private final int _rl;
        private final int _ru;
        private final MatrixBlock _ret;
        private final int _numColumns;
        private final AggregateUnaryOperator _op;

        protected UnaryAggregateTask(List<ColGroup> groups, MatrixBlock ret, int rl, int ru, AggregateUnaryOperator op,
            int numColumns) {
            _groups = groups;
            _op = op;
            _rl = rl;
            _ru = ru;
            _numColumns = numColumns;

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
            aggregateUnaryOperations(_op, _groups, _ret, _rl, _ru, _numColumns);
            return _ret;
        }
    }

    private static class UnaryAggregateOverlappingTask implements Callable<MatrixBlock> {
        private final CompressedMatrixBlock _m1;
        private final int _rl;
        private final int _ru;
        private final MatrixBlock _ret;
        private final AggregateUnaryOperator _op;

        protected UnaryAggregateOverlappingTask(CompressedMatrixBlock m1, MatrixBlock ret, int rl, int ru,
            AggregateUnaryOperator op) {
            _m1 = m1;
            _op = op;
            _rl = rl;
            _ru = ru;
            _ret = ret;

        }

        private MatrixBlock setupOutputMatrix() {
            MatrixBlock outputBlock;
            if(_op.indexFn instanceof ReduceAll)
                outputBlock = new MatrixBlock(_ret.getNumRows(), _ret.getNumColumns(), false).allocateDenseBlock();
            else if(_op.indexFn instanceof ReduceCol)
                outputBlock = new MatrixBlock(_ru - _rl, _ret.getNumColumns(), false).allocateDenseBlock();
            else
                outputBlock = new MatrixBlock(_ret.getNumRows(), _ret.getNumColumns(), false).allocateDenseBlock();

            if(_op.aggOp.increOp.fn instanceof Builtin)
                if(_op.indexFn instanceof ReduceCol)
                    System.arraycopy(_ret.getDenseBlockValues(),
                        _rl * _ret.getNumColumns(),
                        outputBlock.getDenseBlockValues(),
                        0,
                        outputBlock.getDenseBlockValues().length);
                else
                    System.arraycopy(_ret.getDenseBlockValues(),
                        0,
                        outputBlock.getDenseBlockValues(),
                        0,
                        _ret.getDenseBlockValues().length);

            return outputBlock;
        }

        private MatrixBlock getTmp() {
            MatrixBlock tmp = memPool.get();
            if(tmp == null) {
                memPool.set(new MatrixBlock(_ru - _rl, _m1.getNumColumns(), false, -1).allocateBlock());
                tmp = memPool.get();
            }
            else {
                tmp = memPool.get();
                tmp.reset(_ru - _rl, _m1.getNumColumns(), false, -1);
            }
            return tmp;
        }

        private MatrixBlock decompressToTemp() {
            MatrixBlock tmp = getTmp();
            for(ColGroup g : _m1.getColGroups())
                g.decompressToBlockSafe(tmp, _rl, _ru, 0, g.getValues(), false);
            tmp.setNonZeros(_rl + _ru);
            return tmp;
        }

        @Override
        public MatrixBlock call() {
            MatrixBlock tmp = decompressToTemp();

            MatrixBlock outputBlock = setupOutputMatrix();
            LibMatrixAgg.aggregateUnaryMatrix(tmp, outputBlock, _op);

            if(_op.indexFn instanceof ReduceCol) {
                double[] retValues = _ret.getDenseBlockValues();
                int currentIndex = _rl * _ret.getNumColumns();
                double[] outputBlockValues = outputBlock.getDenseBlockValues();
                System.arraycopy(outputBlockValues, 0, retValues, currentIndex, outputBlockValues.length);

                return null;
            }
            else {

                return outputBlock;
            }
        }
    }
}
