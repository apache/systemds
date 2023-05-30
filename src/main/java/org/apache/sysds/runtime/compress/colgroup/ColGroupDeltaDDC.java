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


import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DeltaDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

/**
 * Class to encapsulate information about a column group that is first delta encoded then encoded with dense dictionary
 * encoding (DeltaDDC).
 */
public class ColGroupDeltaDDC extends AColGroupCompressed implements AMapToDataGroup  {

    private static final long serialVersionUID = -1045556313148564147L;
    private AMapToData _data;
    private DeltaDictionary _dict;

    //TODO (optional): create an abstract class ADeltaDictionary
    private ColGroupDeltaDDC(IColIndex colIndexes, DeltaDictionary dict, AMapToData data) {
        super(colIndexes);
        _data = data;
        _dict =  dict;
    }

    //TODO (optional): create an abstract class ADeltaDictionary
    public static AColGroup create(IColIndex colIndexes, DeltaDictionary dict, AMapToData data) {
        // TODO : If my understanding is correct, the DeltaDictionary will contain deltas in _values, therefore, the only delta contained in that case is zero, and we are dealing with a matrix which is "constant" in a sense that all elements are the same.
        // TODO : _data will contain actual indexes. Deltas will be mapped to indexes.
        if (dict.getValues().length == 1 && dict.getValue(0)==0) {
            return new ColGroupEmpty(colIndexes);
        }
        return new ColGroupDeltaDDC(colIndexes, dict, data);
    }


    @Override
    protected AColGroup copyAndSet(IColIndex colIndexes) {
        throw new NotImplementedException();
    }

    @Override
    public double getIdx(int r, int colIdx) {
        throw new NotImplementedException();
    }

    @Override
    public int getNumValues() {
        throw new NotImplementedException();
    }

    @Override
    public CompressionType getCompType() {
        throw new NotImplementedException();
    }

    @Override
    protected ColGroupType getColGroupType() {
        throw new NotImplementedException();
    }

    @Override
    public void decompressToDenseBlock(DenseBlock db, int rl, int ru, int offR, int offC) {
        throw new NotImplementedException();
    }

    @Override
    public void decompressToSparseBlock(SparseBlock sb, int rl, int ru, int offR, int offC) {
        throw new NotImplementedException();
    }

    @Override
    public AColGroup rightMultByMatrix(MatrixBlock right, IColIndex allCols) {
        throw new NotImplementedException();
    }

    @Override
    public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
        throw new NotImplementedException();
    }

    @Override
    public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result, int nRows) {
        throw new NotImplementedException();
    }

    @Override
    public void tsmmAColGroup(AColGroup other, MatrixBlock result) {
        throw new NotImplementedException();
    }

    @Override
    public AColGroup scalarOperation(ScalarOperator op) {
        throw new NotImplementedException();
    }

    @Override
    public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
        throw new NotImplementedException();
    }

    @Override
    public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
        throw new NotImplementedException();
    }

    @Override
    protected AColGroup sliceSingleColumn(int idx) {
        throw new NotImplementedException();
    }

    @Override
    protected AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols) {
        throw new NotImplementedException();
    }

    @Override
    public AColGroup sliceRows(int rl, int ru) {
        throw new NotImplementedException();
    }

    @Override
    public boolean containsValue(double pattern) {
        throw new NotImplementedException();
    }

    @Override
    public long getNumberNonZeros(int nRows) {
        throw new NotImplementedException();
    }

    @Override
    public AColGroup replace(double pattern, double replace) {
        throw new NotImplementedException();
    }

    @Override
    public void computeColSums(double[] c, int nRows) {
        throw new NotImplementedException();
    }

    @Override
    public CM_COV_Object centralMoment(CMOperator op, int nRows) {
        throw new NotImplementedException();
    }

    @Override
    public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
        throw new NotImplementedException();
    }

    @Override
    public double getCost(ComputationCostEstimator e, int nRows) {
        throw new NotImplementedException();
    }

    @Override
    public AColGroup unaryOperation(UnaryOperator op) {
        throw new NotImplementedException();
    }

    @Override
    public AColGroup append(AColGroup g) {
        throw new NotImplementedException();
    }

    @Override
    protected AColGroup appendNInternal(AColGroup[] groups) {
        throw new NotImplementedException();
    }

    @Override
    public ICLAScheme getCompressionScheme() {
        throw new NotImplementedException();
    }

    @Override
    protected double computeMxx(double c, Builtin builtin) {
        throw new NotImplementedException();
    }

    @Override
    protected void computeColMxx(double[] c, Builtin builtin) {
        throw new NotImplementedException();
    }

    @Override
    protected void computeSum(double[] c, int nRows) {
        throw new NotImplementedException();
    }

    @Override
    protected void computeSumSq(double[] c, int nRows) {
        throw new NotImplementedException();
    }

    @Override
    protected void computeColSumsSq(double[] c, int nRows) {
        throw new NotImplementedException();
    }

    @Override
    protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
        throw new NotImplementedException();
    }

    @Override
    protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
        throw new NotImplementedException();
    }

    @Override
    protected void computeProduct(double[] c, int nRows) {
        throw new NotImplementedException();
    }

    @Override
    protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
        throw new NotImplementedException();
    }

    @Override
    protected void computeColProduct(double[] c, int nRows) {
        throw new NotImplementedException();
    }

    @Override
    protected double[] preAggSumRows() {
        throw new NotImplementedException();
    }

    @Override
    protected double[] preAggSumSqRows() {
        throw new NotImplementedException();
    }

    @Override
    protected double[] preAggProductRows() {
        throw new NotImplementedException();
    }

    @Override
    protected double[] preAggBuiltinRows(Builtin builtin) {
        throw new NotImplementedException();
    }

    @Override
    protected void tsmm(double[] result, int numColumns, int nRows) {
        throw new NotImplementedException();
    }

    @Override
    public AMapToData getMapToData() {
        throw new NotImplementedException();
    }


// 	private static final long serialVersionUID = -1045556313148564147L;

// 	/** Constructor for serialization */
// 	protected ColGroupDeltaDDC() {
// 	}

// 	private ColGroupDeltaDDC(int[] colIndexes, ADictionary dict, AMapToData data, int[] cachedCounts) {
// 		super();
// 		LOG.info("Carefully use of DeltaDDC since implementation is not finished.");
// 		_colIndexes = colIndexes;
// 		_dict = dict;
// 		_data = data;
// 	}

// 	public static AColGroup create(int[] colIndices, ADictionary dict, AMapToData data, int[] cachedCounts) {
// 		if(dict == null)
// 			throw new NotImplementedException("Not implemented constant delta group");
// 		else
// 			return new ColGroupDeltaDDC(colIndices, dict, data, cachedCounts);
// 	}

// 	public CompressionType getCompType() {
// 		return CompressionType.DeltaDDC;
// 	}

// 	@Override
// 	protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
// 		double[] values) {
// 		final int nCol = _colIndexes.length;
// 		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
// 			final double[] c = db.values(offT);
// 			final int off = db.pos(offT) + offC;
// 			final int rowIndex = _data.getIndex(i) * nCol;
// 			final int prevOff = (off == 0) ? off : off - nCol;
// 			for(int j = 0; j < nCol; j++) {
// 				// Here we use the values in the previous row to compute current values along with the delta
// 				double newValue = c[prevOff + j] + values[rowIndex + j];
// 				c[off + _colIndexes[j]] += newValue;
// 			}
// 		}
// 	}

// 	@Override
// 	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
// 		double[] values) {
// 		throw new NotImplementedException();
// 	}

// 	@Override
// 	public AColGroup scalarOperation(ScalarOperator op) {
// 		return new ColGroupDeltaDDC(_colIndexes, _dict.applyScalarOp(op), _data, getCachedCounts());
// 	}
}
