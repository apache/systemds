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

import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupValue;
import org.apache.sysds.runtime.compress.colgroup.Dictionary;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.Power2;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class LibScalar {

    // private static final Log LOG = LogFactory.getLog(LibScalar.class.getName());
    private static final int MINIMUM_PARALLEL_SIZE = 8096;

    public static MatrixBlock scalarOperations(ScalarOperator sop, CompressedMatrixBlock m1, CompressedMatrixBlock ret,
        boolean overlapping) {
        // LOG.error(sop);
        if(sop instanceof LeftScalarOperator) {
            if(sop.fn instanceof Minus) {
                m1 = (CompressedMatrixBlock) scalarOperations(new RightScalarOperator(Multiply.getMultiplyFnObject(),
                    -1), m1, ret, overlapping);
                return scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), sop.getConstant()),
                    m1,
                    ret,
                    overlapping);
            }
            else if(sop.fn instanceof Power2) {
                throw new DMLCompressionException("Left Power does not make sense.");
                // List<ColGroup> newColGroups = new ArrayList<>();
                // double v = sop.executeScalar(0);

                // double[] values = new double[m1.getNumColumns()];
                // Arrays.fill(values, v);

                // int[] colIndexes = new int[m1.getNumColumns()];
                // for(int i = 0; i < colIndexes.length; i++) {
                    // colIndexes[i] = i;
                // }
                // newColGroups.add(new ColGroupConst(colIndexes, ret.getNumRows(), new Dictionary(values)));
                // ret.allocateColGroupList(newColGroups);
                // ret.setNonZeros(ret.getNumColumns() * ret.getNumRows());
                // return ret;
            }

        }

        List<ColGroup> colGroups = m1.getColGroups();
        if(overlapping && !(sop.fn instanceof Multiply)) {
            if(sop.fn instanceof Plus || sop.fn instanceof Minus) {

                // If the colGroup is overlapping we know there are no incompressable colGroups.
                List<ColGroup> newColGroups = new ArrayList<>();
                for(ColGroup grp : colGroups) {
                    ColGroupValue g = (ColGroupValue) grp;
                    newColGroups.add(g.copy());
                }
                int[] colIndexes = newColGroups.get(0).getColIndices();
                double v = sop.executeScalar(0);
                double[] values = new double[colIndexes.length];
                Arrays.fill(values, v);
                newColGroups.add(new ColGroupConst(colIndexes, ret.getNumRows(), new Dictionary(values)));
                ret.allocateColGroupList(newColGroups);
                ret.setOverlapping(true);
                ret.setNonZeros(-1);
            }
        }
        else {

            if(sop.getNumThreads() > 1) {
                parallelScalarOperations(sop, colGroups, ret, sop.getNumThreads());
            }
            else {
                // Apply the operation to each of the column groups.
                // Most implementations will only modify metadata.
                List<ColGroup> newColGroups = new ArrayList<>();
                for(ColGroup grp : colGroups) {
                    newColGroups.add(grp.scalarOperation(sop));
                }
                ret.allocateColGroupList(newColGroups);
            }
            ret.setNonZeros(-1);
            ret.setOverlapping(m1.isOverlapping());
        }

        return ret;

    }

    private static void parallelScalarOperations(ScalarOperator sop, List<ColGroup> colGroups,
        CompressedMatrixBlock ret, int k) {
        ExecutorService pool = CommonThreadPool.get(k);
        List<ScalarTask> tasks = partition(sop, colGroups);
        try {
            List<Future<List<ColGroup>>> rtasks = pool.invokeAll(tasks);
            pool.shutdown();
            List<ColGroup> newColGroups = new ArrayList<>();
            for(Future<List<ColGroup>> f : rtasks) {
                newColGroups.addAll(f.get());
            }
            ret.allocateColGroupList(newColGroups);
        }
        catch(InterruptedException | ExecutionException e) {
            throw new DMLRuntimeException(e);
        }
    }

    private static List<ScalarTask> partition(ScalarOperator sop, List<ColGroup> colGroups) {
        ArrayList<ScalarTask> tasks = new ArrayList<>();
        ArrayList<ColGroup> small = new ArrayList<>();
        for(ColGroup grp : colGroups) {
            if(grp instanceof ColGroupUncompressed) {
                ArrayList<ColGroup> uc = new ArrayList<>();
                uc.add(grp);
                tasks.add(new ScalarTask(uc, sop));
            }
            else {
                int nv = ((ColGroupValue) grp).getNumValues() * grp.getColIndices().length;
                if(nv < MINIMUM_PARALLEL_SIZE) {
                    small.add(grp);
                }
                else {
                    ArrayList<ColGroup> large = new ArrayList<>();
                    large.add(grp);
                    tasks.add(new ScalarTask(large, sop));
                }
            }
            if(small.size() > 10) {
                tasks.add(new ScalarTask(small, sop));
                small = new ArrayList<>();
            }
        }
        if(small.size() > 0) {
            tasks.add(new ScalarTask(small, sop));
        }
        return tasks;
    }

    private static class ScalarTask implements Callable<List<ColGroup>> {
        private final List<ColGroup> _colGroups;
        private final ScalarOperator _sop;

        protected ScalarTask(List<ColGroup> colGroups, ScalarOperator sop) {
            _colGroups = colGroups;
            _sop = sop;
        }

        @Override
        public List<ColGroup> call() {
            List<ColGroup> res = new ArrayList<>();
            for(ColGroup x : _colGroups) {
                res.add(x.scalarOperation(_sop));
            }
            return res;
        }
    }
}
