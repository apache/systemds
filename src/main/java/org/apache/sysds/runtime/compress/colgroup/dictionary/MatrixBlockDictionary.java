package org.apache.sysds.runtime.compress.colgroup.dictionary;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

public class MatrixBlockDictionary extends ADictionary {

    private MatrixBlock _data;

    public MatrixBlockDictionary(MatrixBlock data) {
        _data = data;
    }

    public MatrixBlock getMatrixBlock() {
        return _data;
    }

    @Override
    public double[] getValues() {
        LOG.warn("Inefficient force dense format.");
        _data.sparseToDense();
        return _data.getDenseBlockValues();
    }

    @Override
    public double getValue(int i) {
        final int nCol = _data.getNumColumns();
        LOG.warn("inefficient get value at index");
        return _data.quickGetValue(i / nCol, i % nCol);
    }

    @Override
    public int hasZeroTuple(int nCol) {
        if(_data.isInSparseFormat()) {
            SparseBlock sb = _data.getSparseBlock();
            for(int i = 0; i < _data.getNumRows(); i++) {
                if(sb.isEmpty(i)) {
                    return i;
                }
            }
        }
        else {
            throw new NotImplementedException();
        }
        return -1;
    }

    @Override
    public long getInMemorySize() {
        return 8 + _data.estimateSizeInMemory();
    }

    public static long getInMemorySize(int numberValues, int numberColumns, double sparsity) {
        return 8 + MatrixBlock.estimateSizeInMemory(numberValues, numberColumns, sparsity);
    }

    @Override
    public double aggregate(double init, Builtin fn) {
        if(fn.getBuiltinCode() == BuiltinCode.MAX)
            return fn.execute(init, _data.max());
        else if(fn.getBuiltinCode() == BuiltinCode.MIN)
            return fn.execute(init, _data.min());
        else
            throw new NotImplementedException();
    }

    @Override
    public double[] aggregateTuples(Builtin fn, int nCol) {
        double[] ret = new double[_data.getNumRows()];
        if(_data.isEmpty())
            return ret;
        else if(_data.isInSparseFormat()) {
            SparseBlock sb = _data.getSparseBlock();
            for(int i = 0; i < _data.getNumRows(); i++) {
                if(!sb.isEmpty(i)) {
                    final int apos = sb.pos(i);
                    final int alen = sb.size(i) + apos;
                    final double[] avals = sb.values(i);
                    ret[i] = avals[apos];
                    for(int j = apos + 1; j < alen; j++)
                        ret[i] = fn.execute(ret[i], avals[j]);

                    if(sb.size(i) < _data.getNumColumns())
                        ret[i] = fn.execute(ret[i], 0);
                }
                else
                    ret[i] = fn.execute(ret[i], 0);
            }
        }
        else if(nCol == 1)
            return _data.getDenseBlockValues();
        else {
            double[] values = _data.getDenseBlockValues();
            int off = 0;
            for(int k = 0; k < _data.getNumRows(); k++) {
                ret[k] = values[off++];
                for(int j = 1; j < _data.getNumColumns(); j++)
                    ret[k] = fn.execute(ret[k], values[off++]);
            }
        }
        return ret;
    }

    @Override
    public void aggregateCols(double[] c, Builtin fn, int[] colIndexes) {
        if(_data.isEmpty()) {
            for(int j = 0; j < colIndexes.length; j++) {
                final int idx = colIndexes[j];
                c[idx] = fn.execute(c[idx], 0);
            }
        }
        else if(_data.isInSparseFormat()) {
            MatrixBlock t = LibMatrixReorg.transposeInPlace(_data, 1);
            if(!t.isInSparseFormat()) {
                throw new NotImplementedException();
            }
            SparseBlock sbt = t.getSparseBlock();

            for(int i = 0; i < _data.getNumColumns(); i++) {
                final int idx = colIndexes[i];
                if(!sbt.isEmpty(i)) {
                    final int apos = sbt.pos(i);
                    final int alen = sbt.size(i) + apos;
                    final double[] avals = sbt.values(i);
                    for(int j = apos; j < alen; j++)
                        c[idx] = fn.execute(c[idx], avals[j]);
                    if(avals.length != _data.getNumRows())
                        c[idx] = fn.execute(c[idx], 0);
                }
                else
                    c[idx] = fn.execute(c[idx], 0);
            }
        }
        else {
            double[] values = _data.getDenseBlockValues();
            int off = 0;
            for(int k = 0; k < _data.getNumRows(); k++) {
                for(int j = 0; j < _data.getNumColumns(); j++) {
                    final int idx = colIndexes[j];
                    c[idx] = fn.execute(c[idx], values[off++]);
                }
            }
        }
    }

    @Override
    public int size() {
        return (int) _data.getNonZeros();
    }

    @Override
    public ADictionary apply(ScalarOperator op) {
        MatrixBlock res = _data.scalarOperations(op, new MatrixBlock());
        return new MatrixBlockDictionary(res);
    }

    @Override
    public ADictionary applyScalarOp(ScalarOperator op, double newVal, int numCols) {
        MatrixBlock res = _data.scalarOperations(op, new MatrixBlock());
        MatrixBlock res2 = res.append(new MatrixBlock(1, 1, newVal), new MatrixBlock());
        return new MatrixBlockDictionary(res2);
    }

    @Override
    public ADictionary applyBinaryRowOpLeft(ValueFunction fn, double[] v, boolean sparseSafe, int[] colIndexes) {
        throw new NotImplementedException();
    }

    @Override
    public ADictionary applyBinaryRowOpRight(ValueFunction fn, double[] v, boolean sparseSafe, int[] colIndexes) {
        throw new NotImplementedException();
    }

    @Override
    public ADictionary clone() {
        MatrixBlock ret = new MatrixBlock();
        ret.copy(_data);
        return new MatrixBlockDictionary(ret);
    }

    @Override
    public ADictionary cloneAndExtend(int len) {
        throw new NotImplementedException();
    }

    @Override
    public boolean isLossy() {
        return false;
    }

    @Override
    public int getNumberOfValues(int ncol) {
        return _data.getNumRows();
    }

    @Override
    public double[] sumAllRowsToDouble(boolean square, int nrColumns) {
        double[] ret = new double[_data.getNumRows()];

        if(_data.isEmpty())
            return ret;
        else if(_data.isInSparseFormat()) {
            SparseBlock sb = _data.getSparseBlock();
            for(int i = 0; i < _data.getNumRows(); i++) {
                if(!sb.isEmpty(i)) {
                    final int apos = sb.pos(i);
                    final int alen = sb.size(i) + apos;
                    final double[] avals = sb.values(i);
                    for(int j = apos; j < alen; j++) {
                        ret[i] += (square) ? avals[j] * avals[j] : avals[j];
                    }
                }
            }
        }
        else {
            double[] values = _data.getDenseBlockValues();
            int off = 0;
            for(int k = 0; k < _data.getNumRows(); k++) {
                for(int j = 0; j < _data.getNumColumns(); j++) {
                    final double v = values[off++];
                    ret[k] += (square) ? v * v : v;
                }
            }
        }
        return ret;
    }

    @Override
    public double sumRow(int k, boolean square, int nrColumns) {
        throw new NotImplementedException();
    }

    @Override
    public double[] colSum(int[] counts, int nCol) {
        if(_data.isEmpty())
            return null;
        double[] ret = new double[nCol];
        if(_data.isInSparseFormat()) {
            SparseBlock sb = _data.getSparseBlock();
            for(int i = 0; i < _data.getNumRows(); i++) {
                if(!sb.isEmpty(i)) {
                    // double tmpSum = 0;
                    final int count = counts[i];
                    final int apos = sb.pos(i);
                    final int alen = sb.size(i) + apos;
                    final int[] aix = sb.indexes(i);
                    final double[] avals = sb.values(i);
                    for(int j = apos; j < alen; j++) {
                        ret[aix[j]] += count * avals[j];
                    }
                }
            }
        }
        else {
            double[] values = _data.getDenseBlockValues();
            int off = 0;
            for(int k = 0; k < _data.getNumRows(); k++) {
                final int countK = counts[k];
                for(int j = 0; j < _data.getNumColumns(); j++) {
                    final double v = values[off++];
                    ret[j] += v * countK;
                }
            }
        }
        return ret;
    }

    @Override
    public void colSum(double[] c, int[] counts, int[] colIndexes, boolean square) {
        if(_data.isEmpty())
            return;
        if(_data.isInSparseFormat()) {
            SparseBlock sb = _data.getSparseBlock();
            for(int i = 0; i < _data.getNumRows(); i++) {
                if(!sb.isEmpty(i)) {
                    // double tmpSum = 0;
                    final int count = counts[i];
                    final int apos = sb.pos(i);
                    final int alen = sb.size(i) + apos;
                    final int[] aix = sb.indexes(i);
                    final double[] avals = sb.values(i);
                    for(int j = apos; j < alen; j++) {
                        c[colIndexes[aix[j]]] += square ? count * avals[j] * avals[j] : count * avals[j];
                    }
                }
            }
        }
        else {
            double[] values = _data.getDenseBlockValues();
            int off = 0;
            for(int k = 0; k < _data.getNumRows(); k++) {
                final int countK = counts[k];
                for(int j = 0; j < _data.getNumColumns(); j++) {
                    final double v = values[off++];
                    c[colIndexes[j]] += square ? v * v * countK : v * countK;
                }
            }
        }
    }

    @Override
    public double sum(int[] counts, int ncol) {
        double tmpSum = 0;
        if(_data.isEmpty())
            return tmpSum;
        if(_data.isInSparseFormat()) {
            SparseBlock sb = _data.getSparseBlock();
            for(int i = 0; i < _data.getNumRows(); i++) {
                if(!sb.isEmpty(i)) {
                    final int count = counts[i];
                    final int apos = sb.pos(i);
                    final int alen = sb.size(i) + apos;
                    final double[] avals = sb.values(i);
                    for(int j = apos; j < alen; j++) {
                        tmpSum += count * avals[j];
                    }
                }
            }
        }
        else {
            double[] values = _data.getDenseBlockValues();
            int off = 0;
            for(int k = 0; k < _data.getNumRows(); k++) {
                final int countK = counts[k];
                for(int j = 0; j < _data.getNumColumns(); j++) {
                    final double v = values[off++];
                    tmpSum += v * countK;
                }
            }
        }
        return tmpSum;
    }

    @Override
    public double sumsq(int[] counts, int ncol) {
        double tmpSum = 0;
        if(_data.isEmpty())
            return tmpSum;
        if(_data.isInSparseFormat()) {
            SparseBlock sb = _data.getSparseBlock();
            for(int i = 0; i < _data.getNumRows(); i++) {
                if(!sb.isEmpty(i)) {
                    final int count = counts[i];
                    final int apos = sb.pos(i);
                    final int alen = sb.size(i) + apos;
                    final double[] avals = sb.values(i);
                    for(int j = apos; j < alen; j++) {
                        tmpSum += count * avals[j] * avals[j];
                    }
                }
            }
        }
        else {
            double[] values = _data.getDenseBlockValues();
            int off = 0;
            for(int k = 0; k < _data.getNumRows(); k++) {
                final int countK = counts[k];
                for(int j = 0; j < _data.getNumColumns(); j++) {
                    final double v = values[off++];
                    tmpSum += v * v * countK;
                }
            }
        }
        return tmpSum;
    }

    @Override
    public String getString(int colIndexes) {
        return _data.toString();
    }

    @Override
    public void addMaxAndMin(double[] ret, int[] colIndexes) {
        throw new NotImplementedException();
    }

    @Override
    public ADictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
        throw new NotImplementedException();
    }

    @Override
    public ADictionary reExpandColumns(int max) {
        throw new NotImplementedException();
    }

    @Override
    public boolean containsValue(double pattern) {
        return _data.containsValue(pattern);
    }

    @Override
    public long getNumberNonZeros(int[] counts, int nCol) {
        if(_data.isEmpty())
            return 0;
        long nnz = 0;
        if(_data.isInSparseFormat()) {
            SparseBlock sb = _data.getSparseBlock();
            for(int i = 0; i < _data.getNumRows(); i++)
                if(sb.isEmpty(i))
                    nnz += sb.size(i) * counts[i];

        }
        else {
            double[] values = _data.getDenseBlockValues();
            int off = 0;
            for(int i = 0; i < _data.getNumRows(); i++) {
                int countThisTuple = 0;
                for(int j = 0; j < _data.getNumColumns(); j++) {
                    double v = values[off++];
                    if(v != 0)
                        countThisTuple++;
                }
                nnz += countThisTuple * counts[i];
            }
        }
        return nnz;
    }

    @Override
    public long getNumberNonZerosContained() {
        throw new NotImplementedException();
    }

    @Override
    public void addToEntry(Dictionary d, int fr, int to, int nCol) {
        throw new NotImplementedException();
    }

    @Override
    public double[] getMostCommonTuple(int[] counts, int nCol) {
        throw new NotImplementedException();
    }

    @Override
    public ADictionary subtractTuple(double[] tuple) {
        throw new NotImplementedException();
    }

    @Override
    public MatrixBlockDictionary getAsMatrixBlockDictionary(int nCol) {
        // Simply return this.
        return this;
    }

    @Override
    public String toString() {
        return "MatrixBlock Dictionary :" + _data.toString();
    }
}
