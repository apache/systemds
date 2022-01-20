package org.apache.sysds.runtime.compress.colgroup;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * Class to encapsulate information about a column group that is first delta encoded then encoded with dense
 * dictionary encoding (DeltaDDC).
 */
public class ColGroupDeltaDDC extends ColGroupDDC {

    /**
     * Constructor for serialization
     *
     * @param numRows number of rows
     */
    protected ColGroupDeltaDDC(int numRows) {
        super(numRows);
    }

    protected ColGroupDeltaDDC(int[] colIndices, int numRows, ADictionary dict, AMapToData data, int[] cachedCounts) {
        super(colIndices, numRows, dict, data, cachedCounts);
        _zeros = false;
        _data = data;
    }

    public CompressionType getCompType() {
        return CompressionType.DeltaDDC;
    }

    @Override
    protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
                                                         double[] values) {
        final int nCol = _colIndexes.length;
        for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
            final double[] c = db.values(offT);
            final int off = db.pos(offT) + offC;
            final int rowIndex = _data.getIndex(i) * nCol;
            final int prevOff = (off == 0) ? off : off - nCol;
            for(int j = 0; j < nCol; j++) {
                // Here we use the values in the previous row to compute current values along with the delta
                double newValue = c[prevOff + j] + values[rowIndex + j];
                c[off + _colIndexes[j]] += newValue;
            }
        }
    }

    @Override
    protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
                                                          double[] values) {
        throw new NotImplementedException();
    }

    @Override
    public AColGroup scalarOperation(ScalarOperator op) {
        return new ColGroupDeltaDDC(_colIndexes, _numRows, _dict.applyScalarOp(op), _data,
                getCachedCounts());
    }
}
