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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.utils.MemoryEstimates;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

public class ColGroupLinearFunctional extends AColGroupCompressed {

	private static final long serialVersionUID = -2811822570758221975L;

	// Needed for numerical robustness when checking if a value is contained in a column
	private final static double CONTAINS_VALUE_THRESHOLD = 1e-6;

	protected double[] _coefficents;

	protected int _numRows;

	/**
	 * Constructs a Linear Functional Column Group that compresses its content using a linear functional.
	 *
	 * @param colIndices  The Column indexes for the column group.
	 * @param coefficents Array where the first `colIndices.length` entries are the intercepts and the next
	 *                    `colIndices.length` entries are the slopes
	 * @param numRows     Number of rows encoded within this column group.
	 */
	private ColGroupLinearFunctional(IColIndex colIndices, double[] coefficents, int numRows) {
		super(colIndices);
		this._coefficents = coefficents;
		this._numRows = numRows;
	}

	/**
	 * Generate a linear functional column group.
	 *
	 * @param colIndices  The specific column indexes that is contained in this column group.
	 * @param coefficents Array where the first `colIndices.length` entries are the intercepts and the next
	 *                    `colIndices.length` entries are the slopes
	 * @param numRows     Number of rows encoded within this column group.
	 * @return A LinearFunctional column group.
	 */
	public static AColGroup create(IColIndex colIndices, double[] coefficents, int numRows) {
		if(coefficents.length != 2 * colIndices.size())
			throw new DMLCompressionException("Invalid size of values compared to columns");

		boolean allSlopesConstant = true;
		for(int j = 0; j < colIndices.size(); j++) {
			if(coefficents[colIndices.size() + j] != 0) {
				allSlopesConstant = false;
				break;
			}
		}

		if(allSlopesConstant) {
			boolean allInterceptsZero = true;
			for(int j = 0; j < colIndices.size(); j++) {
				if(coefficents[j] != 0) {
					allInterceptsZero = false;
					break;
				}
			}

			if(allInterceptsZero)
				return new ColGroupEmpty(colIndices);
			else {
				double[] intercepts = new double[colIndices.size()];
				System.arraycopy(coefficents, 0, intercepts, 0, colIndices.size());
				return ColGroupConst.create(colIndices, intercepts);
			}
		}
		else
			return new ColGroupLinearFunctional(colIndices, coefficents, numRows);
	}

	public double getInterceptForColumn(int colIdx) {
		return this._coefficents[colIdx];
	}

	public double getSlopeForColumn(int colIdx) {
		return this._coefficents[this._colIndexes.size() + colIdx];
	}

	public int getNumRows() {
		return _numRows;
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		throw new NotImplementedException();
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.LinearFunctional;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.LinearFunctional;
	}

	@Override
	public double getMin() {
		double min = Double.POSITIVE_INFINITY;

		for(int col = 0; col < getNumCols(); col++) {
			double intercept = getInterceptForColumn(col);
			double slope = getSlopeForColumn(col);
			if(slope >= 0 && (intercept + slope) < min) {
				min = intercept + slope;
			}
			else if(slope < 0 && (intercept + _numRows * slope) < min) {
				min = intercept + _numRows * slope;
			}
		}

		return min;
	}

	@Override
	public double getMax() {
		double max = Double.NEGATIVE_INFINITY;

		for(int col = 0; col < getNumCols(); col++) {
			double intercept = getInterceptForColumn(col);
			double slope = getSlopeForColumn(col);
			if(slope >= 0 && (intercept + _numRows * slope) > max) {
				max = intercept + _numRows * slope;
			}
			else if(slope < 0 && (intercept + slope) > max) {
				max = intercept + slope;
			}
		}

		return max;
	}

	@Override
	public void decompressToDenseBlock(DenseBlock db, int rl, int ru, int offR, int offC) {
		final int nCol = getNumCols();
		final double[] accumulators = new double[nCol];

		// copy intercepts into accumulators array
		System.arraycopy(_coefficents, 0, accumulators, 0, nCol);

		int offT = rl + offR;
		for(int row = rl; row < ru; row++, offT++) {
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;

			for(int j = 0; j < nCol; j++) {
				accumulators[j] += getSlopeForColumn(j);
				c[off + _colIndexes.get(j)] += accumulators[j];
			}
		}
	}

	@Override
	public void decompressToSparseBlock(SparseBlock ret, int rl, int ru, int offR, int offC) {
		final int nCol = _colIndexes.size();
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
			for(int j = 0; j < nCol; j++)
				ret.append(offT, _colIndexes.get(j) + offC, getIdx(i, j));
		}
	}

	@Override
	public double getIdx(int r, int colIdx) {
		return getInterceptForColumn(colIdx) + getSlopeForColumn(colIdx) * (r + 1);
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		double[] coefficients_new = new double[_coefficents.length];

		if(op.fn instanceof Plus || op.fn instanceof Minus) {
			// copy slopes into new array, since they do not change if we add/subtract a scalar
			System.arraycopy(_coefficents, 0, coefficients_new, getNumCols(), getNumCols());
			// absorb plus/minus into intercept
			for(int col = 0; col < getNumCols(); col++)
				coefficients_new[col] = op.executeScalar(_coefficents[col]);

			return create(_colIndexes, coefficients_new, _numRows);
		}
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			// multiply/divide changes intercepts & slopes
			for(int j = 0; j < _coefficents.length; j++)
				coefficients_new[j] = op.executeScalar(_coefficents[j]);

			return create(_colIndexes, coefficients_new, _numRows);
		}
		else {
			throw new NotImplementedException();
		}

	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		return binaryRowOp(op, v, isRowSafe, true);
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		return binaryRowOp(op, v, isRowSafe, false);
	}

	private AColGroup binaryRowOp(BinaryOperator op, double[] v, boolean isRowSafe, boolean left) {
		double[] coefficients_new = new double[_coefficents.length];

		if(op.fn instanceof Plus || op.fn instanceof Minus) {
			// copy slopes into new array, since they do not change if we add/subtract a scalar
			System.arraycopy(_coefficents, 0, coefficients_new, getNumCols(), getNumCols());

			// absorb plus/minus into intercept
			if(left) {
				for(int col = 0; col < getNumCols(); col++)
					coefficients_new[col] = op.fn.execute(v[_colIndexes.get(col)], _coefficents[col]);
			}
			else {
				for(int col = 0; col < getNumCols(); col++)
					coefficients_new[col] = op.fn.execute(_coefficents[col], v[_colIndexes.get(col)]);
			}

			return create(_colIndexes, coefficients_new, _numRows);
		}
		else if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			// multiply/divide changes intercepts & slopes
			if(left) {
				for(int col = 0; col < getNumCols(); col++) {
					// update intercept
					coefficients_new[col] = op.fn.execute(v[_colIndexes.get(col)], _coefficents[col]);
					// update slope
					coefficients_new[col + getNumCols()] = op.fn.execute(v[_colIndexes.get(col)],
						_coefficents[col + getNumCols()]);
				}
			}
			else {
				for(int col = 0; col < getNumCols(); col++) {
					// update intercept
					coefficients_new[col] = op.fn.execute(_coefficents[col], v[_colIndexes.get(col)]);
					// update slope
					coefficients_new[col + getNumCols()] = op.fn.execute(_coefficents[col + getNumCols()],
						v[_colIndexes.get(col)]);
				}
			}

			return create(_colIndexes, coefficients_new, _numRows);
		}
		else {
			throw new NotImplementedException();
		}
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
		for(int col = 0; col < getNumCols(); col++) {
			double intercept = getInterceptForColumn(col);
			double slope = getSlopeForColumn(col);
			c[0] += nRows * (intercept + (nRows + 1) * slope / 2);
		}
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		for(int col = 0; col < getNumCols(); col++) {
			double intercept = getInterceptForColumn(col);
			double slope = getSlopeForColumn(col);
			c[_colIndexes.get(col)] += nRows * (intercept + (nRows + 1) * slope / 2);
		}
	}

	@Override
	protected void computeSumSq(double[] c, int nRows) {
		for(int col = 0; col < getNumCols(); col++) {
			double intercept = getInterceptForColumn(col);
			double slope = getSlopeForColumn(col);
			// Given the intercept and slope of a column, the sum of the squared components of the column reads
			// \sum_{i=1}^n (intercept + slope * i)^2
			// We get a closed form expression by expanding the binomial and using the fact that
			// \sum_{i=1}^n i = n(n+1)/2 and \sum_{i=1}^n i^2 = n(n+1)(2n+1)/6

			c[0] += nRows * (Math.pow(intercept, 2) + (nRows + 1) * slope * intercept +
				(nRows + 1) * (2 * nRows + 1) * Math.pow(slope, 2) / 6);
		}
	}

	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		for(int col = 0; col < getNumCols(); col++) {
			double intercept = getInterceptForColumn(col);
			double slope = getSlopeForColumn(col);
			c[_colIndexes.get(col)] += nRows * (Math.pow(intercept, 2) + (nRows + 1) * slope * intercept +
				(nRows + 1) * (2 * nRows + 1) * Math.pow(slope, 2) / 6);
		}
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		double intercept_sum = preAgg[0];
		double slope_sum = preAgg[1];

		for(int rix = rl; rix < ru; rix++)
			c[rix] += intercept_sum + slope_sum * (rix + 1);
	}

	@Override
	public int getNumValues() {
		return 0;
	}

	@Override
	public AColGroup rightMultByMatrix(MatrixBlock right, IColIndex allCols, int k) {
		final int nColR = right.getNumColumns();
		final IColIndex outputCols = allCols != null ? allCols : ColIndexFactory.create(nColR);

		// TODO: add specialization for sparse/dense matrix blocks
		MatrixBlock result = new MatrixBlock(_numRows, nColR, false);
		for(int j = 0; j < nColR; j++) {
			double bias_accum = 0.0;
			double slope_accum = 0.0;

			for(int c = 0; c < _colIndexes.size(); c++) {
				bias_accum += right.get(_colIndexes.get(c), j) * getInterceptForColumn(c);
				slope_accum += right.get(_colIndexes.get(c), j) * getSlopeForColumn(c);
			}

			for(int r = 0; r < _numRows; r++) {
				result.set(r, j, bias_accum + (r + 1) * slope_accum);
			}
		}

		// returns an uncompressed ColGroup
		return ColGroupUncompressed.create(result, outputCols);
	}

	@Override
	public void tsmm(double[] ret, int numColumns, int nRows) {
		// runs in O(tCol^2) since dot-products take O(1) time to compute when both vectors are linearly compressed
		final int tCol = _colIndexes.size();

		final double sumIndices = nRows * (nRows + 1) / 2.0;
		final double sumSquaredIndices = nRows * (nRows + 1) * (2 * nRows + 1) / 6.0;
		for(int row = 0; row < tCol; row++) {
			final double alpha1 = nRows * getInterceptForColumn(row) + sumIndices * getSlopeForColumn(row);
			final double alpha2 = sumIndices * getInterceptForColumn(row) + sumSquaredIndices * getSlopeForColumn(row);
			final int offRet = _colIndexes.get(row) * numColumns;
			for(int col = row; col < tCol; col++) {
				ret[offRet + _colIndexes.get(col)] += alpha1 * getInterceptForColumn(col) + alpha2 * getSlopeForColumn(col);
			}
		}
	}

	@Override
	public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		throw new DMLCompressionException("This method should never be called");
	}

	@Override
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result, int nRows) {
		if(lhs instanceof ColGroupEmpty)
			return;

		MatrixBlock tmpRet = new MatrixBlock(lhs.getNumCols(), _colIndexes.size(), 0);

		if(lhs instanceof ColGroupUncompressed) {
			ColGroupUncompressed lhsUC = (ColGroupUncompressed) lhs;
			int numRowsLeft = lhsUC.getData().getNumRows();

			double[] colSumsAndWeightedColSums = new double[2 * lhs.getNumCols()];
			for(int j = 0, offTmp = 0; j < lhs.getNumCols(); j++, offTmp += 2) {
				for(int i = 0; i < numRowsLeft; i++) {
					colSumsAndWeightedColSums[offTmp] += lhs.getIdx(i, j);
					colSumsAndWeightedColSums[offTmp + 1] += (i + 1) * lhs.getIdx(i, j);
				}
			}

			MatrixBlock sumMatrix = new MatrixBlock(lhs.getNumCols(), 2, colSumsAndWeightedColSums);
			MatrixBlock coefficientMatrix = new MatrixBlock(2, _colIndexes.size(), _coefficents);

			LibMatrixMult.matrixMult(sumMatrix, coefficientMatrix, tmpRet);
		}
		else if(lhs instanceof ColGroupLinearFunctional) {
			ColGroupLinearFunctional lhsLF = (ColGroupLinearFunctional) lhs;

			final double sumIndices = _numRows * (_numRows + 1) / 2.0;
			final double sumSquaredIndices = _numRows * (_numRows + 1) * (2 * _numRows + 1) / 6.0;

			MatrixBlock weightMatrix = new MatrixBlock(2, 2,
				new double[] {_numRows, sumIndices, sumIndices, sumSquaredIndices});
			MatrixBlock coefficientMatrixLhs = new MatrixBlock(2, lhsLF._colIndexes.size(), lhsLF._coefficents);
			MatrixBlock coefficientMatrixRhs = new MatrixBlock(2, _colIndexes.size(), _coefficents);

			coefficientMatrixLhs = LibMatrixReorg.transposeInPlace(coefficientMatrixLhs,
				InfrastructureAnalyzer.getLocalParallelism());

			// We simply compute a matrix multiplication chain in coefficient space, i.e.,
			// t(L) %*% R = t(coeff(L)) %*% W %*% coeff(R)
			// where W is a weight matrix capturing the size of the shared dimension (weightMatrix above)
			// and coeff(X) denotes the 2 x n matrix of the m x n matrix X.
			MatrixBlock tmp = new MatrixBlock(lhs.getNumCols(), 2, false);
			LibMatrixMult.matrixMult(coefficientMatrixLhs, weightMatrix, tmp);
			LibMatrixMult.matrixMult(tmp, coefficientMatrixRhs, tmpRet);
		}
		else if(lhs instanceof APreAgg) {
			// TODO: implement
			throw new NotImplementedException();
		}
		else {
			throw new NotImplementedException();
		}

		ColGroupUtils.copyValuesColGroupMatrixBlocks(lhs, this, tmpRet, result);
	}

	@Override
	public void tsmmAColGroup(AColGroup other, MatrixBlock result) {
		throw new DMLCompressionException("Should not be called");
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
	public boolean containsValue(double pattern) {
		for(int col = 0; col < getNumCols(); col++) {
			if(colContainsValue(col, pattern))
				return true;
		}

		return false;
	}

	public boolean colContainsValue(int col, double pattern) {
		if(pattern == getInterceptForColumn(col))
			return Math.abs(getSlopeForColumn(col)) < CONTAINS_VALUE_THRESHOLD;

		double div = (pattern - getInterceptForColumn(col)) / getSlopeForColumn(col);
		double diffToNextInt = Math.min(Math.ceil(div) - div, div - Math.floor(div));

		return Math.abs(diffToNextInt) < CONTAINS_VALUE_THRESHOLD;
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		throw new NotImplementedException();
	}

	public static ColGroupLinearFunctional read(DataInput in, int nRows) throws IOException {
		IColIndex cols = ColIndexFactory.read(in);
		double[] coefficients = ColGroupIO.readDoubleArray(2 * cols.size(), in);
		return new ColGroupLinearFunctional(cols, coefficients, nRows);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		for(double d : _coefficents)
			out.writeDouble(d);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += MemoryEstimates.doubleArrayCost(_coefficents.length);
		ret += 4L; // _numRows
		return ret;
	}

	@Override
	protected void computeProduct(double[] c, int nRows) {
		if(containsValue(0)) {
			c[0] = 0;
			return;
		}

		for(int col = 0; col < getNumCols(); col++) {
			double intercept = getInterceptForColumn(col);
			double slope = getSlopeForColumn(col);

			for(int i = 0; i < nRows; i++) {
				c[0] *= intercept + slope * (i + 1);
			}
		}
	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		for(int rix = rl; rix < ru; rix++) {
			for(int col = 0; col < getNumCols(); col++) {
				double intercept = getInterceptForColumn(col);
				double slope = getSlopeForColumn(col);
				c[rix] *= intercept + slope * (rix + 1);
			}
		}
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		for(int col = 0; col < getNumCols(); col++) {
			if(colContainsValue(col, 0)) {
				c[_colIndexes.get(col)] = 0;
			}
			else {
				double intercept = getInterceptForColumn(col);
				double slope = getSlopeForColumn(col);
				for(int i = 0; i < nRows; i++) {
					c[_colIndexes.get(col)] *= intercept + slope * (i + 1);
				}
			}
		}
	}

	@Override
	protected double[] preAggSumRows() {
		double intercept_sum = 0;
		for(int col = 0; col < getNumCols(); col++)
			intercept_sum += getInterceptForColumn(col);

		double slope_sum = 0;
		for(int col = 0; col < getNumCols(); col++)
			slope_sum += getSlopeForColumn(col);

		return new double[] {intercept_sum, slope_sum};
	}

	@Override
	protected double[] preAggSumSqRows() {
		return null;
	}

	@Override
	protected double[] preAggProductRows() {
		return null;
	}

	@Override
	protected double[] preAggBuiltinRows(Builtin builtin) {
		throw new NotImplementedException();
	}

	@Override
	public long estimateInMemorySize() {
		return ColGroupSizes.estimateInMemorySizeLinearFunctional(getNumCols(), _colIndexes.isContiguous());
	}

	@Override
	public CmCovObject centralMoment(CMOperator op, int nRows) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		throw new NotImplementedException();
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		LOG.warn("Cost calculation for LinearFunctional ColGroup is not precise");
		final int nCols = getNumCols();
		// We store 2 tuples in this column group, namely intercepts and slopes
		return e.getCost(nRows, nRows, nCols, 2, 1.0);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s", " Intercepts: " + Arrays.toString(getIntercepts())));
		sb.append(String.format("\n%15s", " Slopes: " + Arrays.toString(getSlopes())));
		return sb.toString();
	}

	public double[] getIntercepts() {
		double[] intercepts = new double[getNumCols()];
		for(int col = 0; col < getNumCols(); col++)
			intercepts[col] = getInterceptForColumn(col);

		return intercepts;
	}

	public double[] getSlopes() {
		double[] slopes = new double[getNumCols()];
		for(int col = 0; col < getNumCols(); col++)
			slopes[col] = getSlopeForColumn(col);

		return slopes;
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup copyAndSet(IColIndex colIndexes) {
		return ColGroupLinearFunctional.create(colIndexes, _coefficents, _numRows);
	}

	@Override
	public AColGroup append(AColGroup g) {
		return null;
	}

	@Override
	public AColGroup appendNInternal(AColGroup[] g, int blen, int rlen) {
		throw new NotImplementedException();
	}

	@Override
	public ICLAScheme getCompressionScheme() {
		return null;
	}

	@Override
	public AColGroup recompress() {
		return this;
	}

	@Override
	public CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
		throw new NotImplementedException("Not Implemented Compressed SizeInfo for Linear col group");
	}

	@Override
	public boolean sameIndexStructure(AColGroupCompressed that) {
		throw new NotImplementedException();
	}

	@Override
	protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup reduceCols() {
		throw new NotImplementedException();
	}

	@Override
	public double getSparsity() {
		return 1.0;
	}

	@Override
	public void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	public void decompressToDenseBlockTransposed(DenseBlock db, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	public void decompressToSparseBlockTransposed(SparseBlockMCSR sb, int nColOut) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
		throw new NotImplementedException("Unimplemented method 'splitReshape'");
	}

}
