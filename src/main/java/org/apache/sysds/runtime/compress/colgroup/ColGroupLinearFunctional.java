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
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.DataConverter;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

public class ColGroupLinearFunctional extends AColGroupCompressed {

	private static final long serialVersionUID = -2811822570758221975L;

	protected double[][] _coefficents;

	protected int _numRows;

	/** Constructor for serialization */
	protected ColGroupLinearFunctional() {
		super();
	}

	/**
	 * Constructs a Linear Functional Column Group that compresses its content using a linear functional.
	 *
	 * @param colIndices  The Column indexes for the column group.
	 * @param coefficents 2D-Array where coefficients[0] is the list of intercepts and coefficients[1] the list of slopes.
	 * @param numRows Number of rows encoded within this column group.
	 */
	private ColGroupLinearFunctional(int[] colIndices, double[][] coefficents, int numRows) {
		super(colIndices);
		this._coefficents = coefficents;
		this._numRows = numRows;
	}

	/**
	 * Generate a linear functional column group.
	 *
	 * @param colIndices   The specific column indexes that is contained in this column group.
	 * @param coefficents 2D-Array where coefficients[0] is the list of intercepts and coefficients[1] the list of slopes.
	 * @param numRows Number of rows encoded within this column group.
	 * @return A LinearFunctional column group.
	 */
	public static AColGroup create(int[] colIndices, double[][] coefficents, int numRows) {
		if(coefficents.length != 2 || colIndices.length != coefficents[0].length)
			throw new DMLCompressionException("Invalid size of values compared to columns");
		return new ColGroupLinearFunctional(colIndices, coefficents, numRows);
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
			double intercept = _coefficents[0][col];
			double slope = _coefficents[1][col];
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
			double intercept = _coefficents[0][col];
			double slope = _coefficents[1][col];
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
		int offT = rl + offR;
		final int nCol = _colIndexes.length;
		int offS = rl * nCol;
		for(int row = rl; row < ru; row++, offT++, offS += nCol) {
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			for(int j = 0; j < nCol; j++)
				c[off + _colIndexes[j]] += getIdx(row, j);
		}
	}

	@Override
	public void decompressToSparseBlock(SparseBlock ret, int rl, int ru, int offR, int offC) {
		final int nCol = _colIndexes.length;
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
			for(int j = 0; j < nCol; j++)
				ret.append(offT, _colIndexes[j] + offC, getIdx(i, j));
		}
	}

	@Override
	public double getIdx(int r, int colIdx) {
		return _coefficents[0][colIdx] + _coefficents[1][colIdx] * (r + 1);
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
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
			double intercept = _coefficents[0][col];
			double slope = _coefficents[1][col];
			c[0] += nRows * (intercept + (nRows + 1) * slope / 2);
		}
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		for(int col = 0; col < getNumCols(); col++) {
			double intercept = _coefficents[0][col];
			double slope = _coefficents[1][col];
			c[_colIndexes[col]] += nRows * (intercept + (nRows + 1) * slope / 2);
		}
	}

	@Override
	protected void computeSumSq(double[] c, int nRows) {
		for(int col = 0; col < getNumCols(); col++) {
			double intercept = _coefficents[0][col];
			double slope = _coefficents[1][col];
			c[0] += nRows * (Math.pow(intercept, 2) + (nRows + 1) * slope * intercept +
				(nRows + 1) * (2 * nRows + 1) * Math.pow(slope, 2) / 6);
		}
	}

	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		for(int col = 0; col < getNumCols(); col++) {
			double intercept = _coefficents[0][col];
			double slope = _coefficents[1][col];
			c[_colIndexes[col]] += nRows * (Math.pow(intercept, 2) + (nRows + 1) * slope * intercept +
				(nRows + 1) * (2 * nRows + 1) * Math.pow(slope, 2) / 6);
		}
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		double intercept_sum = 0;
		for(double intercept : _coefficents[0])
			intercept_sum += intercept;

		double slope_sum = 0;
		for(double slope : _coefficents[1])
			slope_sum += slope;

		for(int rix = rl; rix < ru; rix++)
			c[rix] += intercept_sum + slope_sum * (rix + 1);
	}

	@Override
	public int getNumValues() {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup rightMultByMatrix(MatrixBlock right) {
		final int nColR = right.getNumColumns();
		final int[] outputCols = Util.genColsIndices(nColR);

		MatrixBlock result = new MatrixBlock(_numRows, nColR, false);
		for(int j = 0; j < nColR; j++) {
			double bias_accum = 0.0;
			double slope_accum = 0.0;

			for(int c = 0; c < _colIndexes.length; c++) {
				bias_accum += right.getValue(_colIndexes[c], j) * _coefficents[0][c];
				slope_accum += right.getValue(_colIndexes[c], j) * _coefficents[1][c];
			}

			for(int r = 0; r < _numRows; r++) {
				result.setValue(r, j, bias_accum + (r+1) * slope_accum);
			}
		}

		// returns an uncompressed ColGroup
		return ColGroupUncompressed.create(result, outputCols);
	}

	@Override
	public void tsmm(double[] ret, int numColumns, int nRows) {
		// runs in O(nRows^2) since dot-products take O(1) time to compute when both vectors are linearly compressed
		final int tCol = _colIndexes.length;

		final double sumIndices = nRows * (nRows + 1)/2.0;
		final double sumSquaredIndices = nRows * (nRows + 1) * (2*nRows + 1)/6.0;
		for(int row = 0, offTmp = 0; row < tCol; row++, offTmp += tCol) {
			final double alpha1 = nRows * _coefficents[0][row] + sumIndices * _coefficents[1][row];
			final double alpha2 = sumIndices * _coefficents[0][row] + sumSquaredIndices * _coefficents[1][row];
			final int offRet = _colIndexes[row] * numColumns;
			for(int col = row; col < tCol; col++) {
				ret[offRet + _colIndexes[col]] += alpha1 * _coefficents[0][col] + alpha2 * _coefficents[1][col];
			}
		}
	}

	@Override
	public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		throw new DMLCompressionException("This method should never be called");
	}

	@Override
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result) {
		if(lhs instanceof ColGroupEmpty) {
			return;
		}

		MatrixBlock tmpRet = new MatrixBlock(lhs.getNumCols(), _colIndexes.length, 0);

		if(lhs instanceof ColGroupUncompressed) {
			ColGroupUncompressed lhsUC = (ColGroupUncompressed) lhs;
			int numRowsLeft = lhsUC.getData().getNumRows();

			double[] colSumsAndWeightedColSums = new double[2 * lhs.getNumCols()];
			for(int j = 0, offTmp = 0; j < lhs.getNumCols(); j++, offTmp += 2) {
				for(int i = 0; i < numRowsLeft; i++) {
					colSumsAndWeightedColSums[offTmp] += lhs.getIdx(i, j);
					colSumsAndWeightedColSums[offTmp + 1] += (i+1) * lhs.getIdx(i, j);
				}
			}

			MatrixBlock sumMatrix = new MatrixBlock(lhs.getNumCols(), 2, colSumsAndWeightedColSums);
			MatrixBlock coefficientMatrix = DataConverter.convertToMatrixBlock(_coefficents);

			LibMatrixMult.matrixMult(sumMatrix, coefficientMatrix, tmpRet);
		} else if(lhs instanceof ColGroupLinearFunctional) {
			ColGroupLinearFunctional lhsLF = (ColGroupLinearFunctional) lhs;

			final double sumIndices = _numRows * (_numRows + 1)/2.0;
			final double sumSquaredIndices = _numRows * (_numRows + 1) * (2*_numRows + 1)/6.0;

			MatrixBlock weightMatrix = new MatrixBlock(2, 2, new double[] {_numRows, sumIndices, sumIndices, sumSquaredIndices});
			MatrixBlock coefficientMatrixLhs = DataConverter.convertToMatrixBlock(lhsLF._coefficents);
			MatrixBlock coefficientMatrixRhs = DataConverter.convertToMatrixBlock(_coefficents);

			coefficientMatrixLhs = LibMatrixReorg.transposeInPlace(coefficientMatrixLhs,
				InfrastructureAnalyzer.getLocalParallelism());

			// We simply compute a matrix multiplication chain in coefficient space, i.e.,
			// 					t(L) %*% R = t(coeff(L)) %*% W %*% coeff(R)
			// where W is a weight matrix capturing the size of the shared dimension (weightMatrix above)
			// and coeff(X) denotes the 2 x n matrix of the m x n matrix X.
			MatrixBlock tmp = new MatrixBlock(lhs.getNumCols(), 2, false);
			LibMatrixMult.matrixMult(coefficientMatrixLhs, weightMatrix, tmp);
			LibMatrixMult.matrixMult(tmp, coefficientMatrixRhs, tmpRet);
		} else if(lhs instanceof APreAgg) {
			// TODO: implement
			throw new NotImplementedException();
		} else {
			throw new NotImplementedException();
		}

		final double[] resV = result.getDenseBlockValues();
		if(tmpRet.isEmpty())
			return;
		else if(tmpRet.isInSparseFormat()) {
			SparseBlock sb = tmpRet.getSparseBlock();
			for(int row = 0; row < lhs._colIndexes.length; row++) {
				if(sb.isEmpty(row))
					continue;
				final int apos = sb.pos(row);
				final int alen = sb.size(row) + apos;
				final int[] aix = sb.indexes(row);
				final double[] avals = sb.values(row);
				final int offRes = lhs._colIndexes[row] * result.getNumColumns();
				for(int col = apos; col < alen; col++)
					resV[offRes + _colIndexes[aix[col]]] += avals[col];
			}
		}
		else {
			double[] tmpRetV = tmpRet.getDenseBlockValues();
			for(int row = 0; row < lhs.getNumCols(); row++) {
				final int offRes = lhs._colIndexes[row] * result.getNumColumns();
				final int offTmp = row * getNumCols();
				for(int col = 0; col < getNumCols(); col++) {
					resV[offRes + _colIndexes[col]] += tmpRetV[offTmp + col];
				}
			}
		}
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
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, int[] outputCols) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup copy() {
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
	public void readFields(DataInput in) throws IOException {
		throw new NotImplementedException();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		throw new NotImplementedException();
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _coefficents[0].length * _coefficents.length * 8L;
		ret += 4L; // _numRows
		return ret;
	}

	@Override
	protected void computeProduct(double[] c, int nRows) {
		c[0] = 1;
		for(int colIndex : _colIndexes) {
			double intercept = _coefficents[0][colIndex];
			double slope = _coefficents[1][colIndex];
			for(int i = 0; i < nRows; i++) {
				c[0] *= intercept + slope * (i + 1);
			}
		}
	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		for(int rix = rl; rix < ru; rix++) {
			c[rix] = 1;
			for(int colIndex : _colIndexes) {
				double intercept = _coefficents[0][colIndex];
				double slope = _coefficents[1][colIndex];
				c[rix] *= intercept + slope * (rix + 1);
			}
		}
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		for(int colIndex : _colIndexes) {
			c[colIndex] = 1;
			double intercept = _coefficents[0][colIndex];
			double slope = _coefficents[1][colIndex];
			for(int i = 0; i < nRows; i++) {
				c[colIndex] *= intercept + slope * (i + 1);
			}
		}
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
	public long estimateInMemorySize() {
		return ColGroupSizes.estimateInMemorySizeLinearFunctional(getNumCols());
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
		final int nCols = getNumCols();
		// We store 2 tuples in this column group, namely intercepts and slopes
		return e.getCost(nRows, nRows, nCols, 2, 1.0);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s", " Intercepts: \t" + Arrays.toString(_coefficents[0])));
		sb.append(String.format("\n%15s", " Slopes: \t" + Arrays.toString(_coefficents[1])));
		return sb.toString();
	}
}
