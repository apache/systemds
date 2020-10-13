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

package org.apache.sysds.runtime.compress.colgroup;

import java.util.Iterator;

import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

import edu.emory.mathcs.backport.java.util.Arrays;

public class ColGroupConst extends ColGroupValue {

    private static final long serialVersionUID = 3204391661346504L;

    /**
     * Constructor for serialization
     */
    protected ColGroupConst() {
        super();
    }

    /**
     * Constructs an Constant Colum Group, that contains only one tuple, with the given value.
     * 
     * @param colIndices The Colum indexes for the column group.
     * @param numRows    The number of rows contained in the group.
     * @param dict       The dictionary containing one tuple for the entire compression.
     */
    public ColGroupConst(int[] colIndices, int numRows, ADictionary dict) {
        super(colIndices, numRows, dict);
    }

    @Override
    public int[] getCounts(int[] out) {
        out[0] = _numRows;
        return out;
    }

    @Override
    public int[] getCounts(int rl, int ru, int[] out) {
        out[0] = ru - rl;
        return out;
    }

    @Override
    protected void computeSum(double[] c, KahanFunction kplus) {
        c[0] += _dict.sum(getCounts(), _colIndexes.length, kplus);
    }

    @Override
    protected void computeRowSums(double[] c, KahanFunction kplus, int rl, int ru, boolean mean) {
        KahanObject kbuff = new KahanObject(0, 0);
        KahanPlus kplus2 = KahanPlus.getKahanPlusFnObject();
        double[] vals = _dict.sumAllRowsToDouble(kplus, kbuff, _colIndexes.length);
        for(int rix = rl; rix < ru; rix++) {
            setandExecute(c, kbuff, kplus2, vals[0], rix * (2 + (mean ? 1 : 0)));
        }
    }

    @Override
    protected void computeColSums(double[] c, KahanFunction kplus) {
        _dict.colSum(c, getCounts(), _colIndexes, kplus);
    }

    @Override
    protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru) {
        throw new DMLCompressionException(
            "Row max not supported for Const since Const is used for overlapping ColGroups, You have to materialize rows and then calculate row max");
    }

    @Override
    public CompressionType getCompType() {
        return CompressionType.CONST;
    }

    @Override
    protected ColGroupType getColGroupType() {
        return ColGroupType.CONST;
    }

    @Override
    public long estimateInMemorySize() {
        return ColGroupSizes.estimateInMemorySizeCONST(getNumCols(), getNumValues(), isLossy());
    }

    @Override
    public void decompressToBlock(MatrixBlock target, int rl, int ru) {
        final int ncol = getNumCols();
        final double[] values = getValues();

        for(int i = rl; i < ru; i++)
            for(int j = 0; j < ncol; j++) {
                double v = target.quickGetValue(i, _colIndexes[j]);
                target.setValue(i, _colIndexes[j], values[j] + v);
            }
    }

    @Override
    public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
        int ncol = getNumCols();
        double[] values = getValues();
        for(int i = 0; i < _numRows; i++) {
            for(int colIx = 0; colIx < ncol; colIx++) {
                int origMatrixColIx = getColIndex(colIx);
                int col = colIndexTargets[origMatrixColIx];
                double cellVal = values[colIx];
                target.quickSetValue(i, col, target.quickGetValue(i, col) + cellVal);
            }
        }
    }

    @Override
    public void decompressToBlock(MatrixBlock target, int colpos) {
        double[] c = target.getDenseBlockValues();

        int nnz = 0;
        double v = _dict.getValue(Arrays.binarySearch(_colIndexes, colpos));
        if(v != 0) {
            for(int i = 0; i < c.length; i++)
                c[i] += v;
            nnz = _numRows;
        }
        target.setNonZeros(nnz);
    }

    @Override
    public double get(int r, int c) {
        return _dict.getValue(Arrays.binarySearch(_colIndexes, c));
    }

    @Override
    public void rightMultByVector(double[] b, double[] c, int rl, int ru, double[] dictVals) {
        double[] vals = preaggValues(1, b, dictVals);
        for(int i = 0; i < c.length; i++) {
            c[i] += vals[0];
        }
    }

    @Override
    public void rightMultByMatrix(double[] preAggregatedB, double[] c, int thatNrColumns, int rl, int ru, int cl,
        int cu) {

        for(int i = rl * thatNrColumns; i < ru * thatNrColumns; i += thatNrColumns)
            for(int j = i + cl; j < i + cu; j++)
                c[j] += preAggregatedB[j % thatNrColumns];

    }

    @Override
    public void rightMultBySparseMatrix(SparseRow[] rows, double[] c, int numVals, double[] dictVals, int nrColumns,
        int rl, int ru) {
        throw new DMLCompressionException(
            "Depreciated and not supported right mult by sparse matrix Please preAggregate before calling");
    }

    private double preAggregate(double[] a, int aRows) {
        double vals = 0;
        for(int i = 0, off = _numRows * aRows; i < _numRows; i++, off++) {
            vals += a[off];
        }
        return vals;
    }

    @Override
    public void leftMultByRowVector(double[] a, double[] c, int numVals) {
        double preAggVals = preAggregate(a, 0);
        double[] dictVals = getValues();
        for(int i = 0; i < _colIndexes.length; i++) {
            c[i] += preAggVals * dictVals[i];
        }
    }

    @Override
    public void leftMultByRowVector(double[] a, double[] c, int numVals, double[] values) {
        double preAggVals = preAggregate(a, 0);
        for(int i = 0; i < _colIndexes.length; i++) {
            c[i] += preAggVals * values[i];
        }
    }

    @Override
    public void leftMultByMatrix(double[] a, double[] c, double[] values, int numRows, int numCols, int rl, int ru,
        int vOff) {
        for(int i = rl; i < ru; i++) {
            double preAggVals = preAggregate(a, i);
            int offC = i * numCols;
            for(int j = 0; j < _colIndexes.length; j++) {
                c[offC + j] += preAggVals * values[j];
            }
        }
    }

    @Override
    public void leftMultBySparseMatrix(int spNrVals, int[] indexes, double[] sparseV, double[] c, int numVals,
        double[] values, int numRows, int numCols, int row, double[] MaterializedRow) {
        double v = 0;
        for(int i = 0; i < spNrVals; i++) {
            v += sparseV[i];
        }
        int offC = row * numCols;
        for(int j = 0; j < _colIndexes.length; j++) {
            c[offC + j] += v * values[j];
        }
    }

    @Override
    public ColGroup scalarOperation(ScalarOperator op) {
        return new ColGroupConst(_colIndexes, _numRows, applyScalarOp(op));
    }

    @Override
    public ColGroup binaryRowOp(BinaryOperator op, double[] v, boolean sparseSafe) {
        return new ColGroupConst(_colIndexes, _numRows, applyBinaryRowOp(op.fn, v, true));
    }

    @Override
    public Iterator<IJV> getIterator(int rl, int ru, boolean inclZeros, boolean rowMajor) {
        throw new DMLCompressionException("Unsupported Iterator of Const ColGroup");
    }

    @Override
    public ColGroupRowIterator getRowIterator(int rl, int ru) {
        throw new DMLCompressionException("Unsupported Row iterator of Const ColGroup");
    }

    @Override
    public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {

        double[] values = _dict.getValues();
        int base = 0;
        for(int i = 0; i < values.length; i++) {
            base += values[i] == 0 ? 0 : 1;
        }
        for(int i = 0; i < ru - rl; i++) {
            rnnz[i] = base;
        }
    }
}
