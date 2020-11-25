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
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.Dictionary;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.Equals;
import org.apache.sysds.runtime.functionobjects.GreaterThan;
import org.apache.sysds.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysds.runtime.functionobjects.LessThan;
import org.apache.sysds.runtime.functionobjects.LessThanEquals;
import org.apache.sysds.runtime.functionobjects.NotEquals;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * This class is used for relational operators that return binary values depending on individual cells values in the
 * compression. This indicate that the resulting vectors/matrices are amenable to compression since they only contain
 * two distinct values, true or false.
 * 
 */
public class LibRelationalOp {
    // private static final Log LOG = LogFactory.getLog(LibRelationalOp.class.getName());

    /** Thread pool matrix Block for materializing decompressed groups. */
    private static ThreadLocal<MatrixBlock> memPool = new ThreadLocal<MatrixBlock>() {
        @Override
        protected MatrixBlock initialValue() {
            return null;
        }
    };

    public static MatrixBlock relationalOperation(ScalarOperator sop, CompressedMatrixBlock m1, boolean overlapping) {

        List<ColGroup> colGroups = m1.getColGroups();
        if(overlapping) {
            if(sop.fn instanceof LessThan || sop.fn instanceof LessThanEquals || sop.fn instanceof GreaterThan ||
                sop.fn instanceof GreaterThanEquals || sop.fn instanceof Equals || sop.fn instanceof NotEquals) {
                return overlappingRelativeRelationalOperation(sop, m1);
            }
            else {
                throw new DMLCompressionException("Invalid arguments to relational Operation");
            }
        }
        else {
            CompressedMatrixBlock ret = new CompressedMatrixBlock(m1.getNumRows(), m1.getNumColumns(), true);
            List<ColGroup> newColGroups = new ArrayList<>();
            for(ColGroup grp : colGroups) {
                newColGroups.add(grp.scalarOperation(sop));
            }
            ret.allocateColGroupList(newColGroups);
            ret.setNonZeros(-1);
            ret.setOverlapping(false);
            return (MatrixBlock) ret;
        }

    }

    private static MatrixBlock overlappingRelativeRelationalOperation(ScalarOperator sop, CompressedMatrixBlock m1) {

        List<ColGroup> colGroups = m1.getColGroups();
        boolean less = ((sop.fn instanceof LessThan || sop.fn instanceof LessThanEquals) &&
            sop instanceof LeftScalarOperator) ||
            (sop instanceof RightScalarOperator &&
                (sop.fn instanceof GreaterThan || sop.fn instanceof GreaterThanEquals));
        double v = sop.getConstant();
        MinMaxGroup[] minMax = new MinMaxGroup[colGroups.size()];
        double maxS = 0.0;
        double minS = 0.0;
        Builtin min = Builtin.getBuiltinFnObject(BuiltinCode.MIN);
        Builtin max = Builtin.getBuiltinFnObject(BuiltinCode.MAX);
        int id = 0;
        for(ColGroup grp : colGroups) {
            double infN = Double.NEGATIVE_INFINITY;
            double infP = Double.POSITIVE_INFINITY;
            double minG = grp.computeMxx(infP, min);
            double maxG = grp.computeMxx(infN, max);
            minS += minG;
            maxS += maxG;
            minMax[id++] = new MinMaxGroup(minG, maxG, grp);
        }

        // Shortcut:
        // If we know worst case min and worst case max and the values to compare to in all cases is
        // less then or greater than worst then we can return a full matrix with either 1 or 0.

        if(v < minS || v > maxS) {
            if(sop.fn instanceof Equals) {
                return makeConstZero(m1.getNumRows(), m1.getNumColumns());
            }
            else if(sop.fn instanceof NotEquals) {
                return makeConstOne(m1.getNumRows(), m1.getNumColumns());
            }
            else if(less) {
                if(v < minS || ((sop.fn instanceof LessThanEquals || sop.fn instanceof GreaterThan) && v <= minS))
                    return makeConstOne(m1.getNumRows(), m1.getNumColumns());
                else
                    return makeConstZero(m1.getNumRows(), m1.getNumColumns());
            }
            else {
                if(v > minS || ((sop.fn instanceof LessThanEquals || sop.fn instanceof GreaterThan) && v >= minS))
                    return makeConstOne(m1.getNumRows(), m1.getNumColumns());
                else
                    return makeConstZero(m1.getNumRows(), m1.getNumColumns());
            }
        }
        else {
            return processNonConstant(sop, minMax, minS, maxS, m1.getNumRows(), m1.getNumColumns(), less);
        }

    }

    private static MatrixBlock makeConstOne(int rows, int cols) {
        List<ColGroup> newColGroups = new ArrayList<>();
        int[] colIndexes = new int[cols];
        for(int i = 0; i < colIndexes.length; i++) {
            colIndexes[i] = i;
        }
        double[] values = new double[cols];
        Arrays.fill(values, 1);

        newColGroups.add(new ColGroupConst(colIndexes, rows, new Dictionary(values)));
        CompressedMatrixBlock ret = new CompressedMatrixBlock(rows, cols, true);
        ret.allocateColGroupList(newColGroups);
        ret.setNonZeros(cols * rows);
        ret.setOverlapping(false);
        return ret;
    }

    private static MatrixBlock makeConstZero(int rows, int cols) {
        MatrixBlock sb = new MatrixBlock(rows, cols, true, 0);
        return sb;
    }

    private static MatrixBlock processNonConstant(ScalarOperator sop, MinMaxGroup[] minMax, double minS, double maxS,
        final int rows, final int cols, boolean less) {

        // BitSet res = new BitSet(ret.getNumColumns() * ret.getNumRows());
        MatrixBlock res = new MatrixBlock(rows, cols, true, 0).allocateBlock();
        int k = OptimizerUtils.getConstrainedNumThreads(-1);
        int outRows = rows;
        long nnz = 0;
        if(k == 1) {
            final int b = CompressionSettings.BITMAP_BLOCK_SZ / cols;
            final int blkz = (outRows < b) ? outRows : b;

            MatrixBlock tmp = new MatrixBlock(blkz, cols, false, -1).allocateBlock();
            for(int i = 0; i * blkz < outRows; i++) {
                for(MinMaxGroup mmg : minMax) {
                    mmg.g.decompressToBlock(tmp, i * blkz, Math.min((i + 1) * blkz, rows), 0, mmg.values);
                }
                for(int row = 0; row < blkz && row < rows - i * blkz; row++) {
                    int off = (row + i * blkz);
                    for(int col = 0; col < cols; col++) {
                        res.quickSetValue(off, col, sop.executeScalar(tmp.quickGetValue(row, col)));
                        if(res.quickGetValue(off, col) != 0) {
                            nnz++;
                        }
                    }
                }
            }
            tmp.reset();
            res.setNonZeros(nnz);
        }
        else {
            final int blkz = CompressionSettings.BITMAP_BLOCK_SZ / cols;
            ExecutorService pool = CommonThreadPool.get(k);
            ArrayList<RelationalTask> tasks = new ArrayList<>();

            try {
                for(int i = 0; i * blkz < outRows; i++) {
                    RelationalTask rt = new RelationalTask(minMax, i, blkz, res, rows, cols, sop);
                    tasks.add(rt);
                }
                List<Future<Object>> futures = pool.invokeAll(tasks);
                pool.shutdown();
                for(Future<Object> f : futures)
                    f.get();
            }
            catch(InterruptedException | ExecutionException e) {
                e.printStackTrace();
                throw new DMLRuntimeException(e);
            }

        }
        memPool.remove();

        return res;
    }

    protected static class MinMaxGroup implements Comparable<MinMaxGroup> {
        double min;
        double max;
        ColGroup g;
        double[] values;

        public MinMaxGroup(double min, double max, ColGroup g) {
            this.min = min;
            this.max = max;
            this.g = g;
            this.values = g.getValues();
        }

        @Override
        public int compareTo(MinMaxGroup o) {
            double t = max - min;
            double ot = o.max - o.min;
            return Double.compare(t, ot);
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("MMG: ");
            sb.append("[" + min + "," + max + "]");
            sb.append(" " + g.getClass().getSimpleName());
            return sb.toString();
        }
    }

    private static class RelationalTask implements Callable<Object> {
        private final MinMaxGroup[] _minMax;
        private final int _i;
        private final int _blkz;
        private final MatrixBlock _res;
        private final int _rows;
        private final int _cols;
        private final ScalarOperator _sop;

        protected RelationalTask(MinMaxGroup[] minMax, int i, int blkz, MatrixBlock res, int rows, int cols,
            ScalarOperator sop) {
            _minMax = minMax;
            _i = i;
            _blkz = blkz;
            _res = res;
            _rows = rows;
            _cols = cols;
            _sop = sop;
        }

        @Override
        public Object call() {
            MatrixBlock tmp = memPool.get();
            if(tmp == null) {
                memPool.set(new MatrixBlock(_blkz, _cols, false, -1).allocateBlock());
                tmp = memPool.get();
            }
            else {
                tmp = memPool.get();
                tmp.reset(_blkz, _cols, false, -1);
            }

            for(MinMaxGroup mmg : _minMax) {
                mmg.g.decompressToBlock(tmp, _i * _blkz, Math.min((_i + 1) * _blkz, mmg.g.getNumRows()), 0, mmg.values);
            }

            for(int row = 0, off = _i * _blkz; row < _blkz && row < _rows - _i * _blkz; row++, off++) {
                for(int col = 0; col < _cols; col++) {
                    _res.appendValue(off, col, _sop.executeScalar(tmp.quickGetValue(row, col)));
                }
            }
            return null;
        }
    }
}
