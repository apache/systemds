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
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class ColGroupLinearFunctional extends AColGroupCompressed {

	private static final long serialVersionUID = -2534788786668777356L;

	protected double[][] _coefficents;

	protected int _numRows;

	/** Constructor for serialization */
	protected ColGroupLinearFunctional() {
		super();
	}

	/**
	 * Constructs a Column Group that compresses its content using a linear functional.
	 *
	 * @param colIndices  The Column indexes for the column group.
	 * @param coefficents The dictionary containing one tuple for the entire compression.
	 */
	private ColGroupLinearFunctional(int[] colIndices, double[][] coefficents, int numRows) {
		super(colIndices);
		this._coefficents = coefficents;
		this._numRows = numRows;
	}

	/**
	 * Generate a constant column group.
	 *
	 * @param colIndices   The specific column indexes that is contained in this column group.
	 * @param coefficents  The coefficents vector
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

		for(int colIndex : _colIndexes) {
			double intercept = _coefficents[0][colIndex];
			double slope = _coefficents[1][colIndex];
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

		for(int colIndex : _colIndexes) {
			double intercept = _coefficents[0][colIndex];
			double slope = _coefficents[1][colIndex];
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
//		return create(_colIndexes, _dict.applyScalarOp(op));
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		throw new NotImplementedException();
//		return create(_colIndexes, _dict.applyUnaryOp(op));
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		throw new NotImplementedException();
//		return create(_colIndexes, _dict.binOpLeft(op, v, _colIndexes));
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		throw new NotImplementedException();
//		return create(_colIndexes, _dict.binOpRight(op, v, _colIndexes));
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
		for(int colIndex : _colIndexes) {
			double intercept = _coefficents[0][colIndex];
			double slope = _coefficents[1][colIndex];
			c[0] += nRows * (intercept + (nRows + 1) * slope / 2);
		}
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		for(int colIndex : _colIndexes) {
			double intercept = _coefficents[0][colIndex];
			double slope = _coefficents[1][colIndex];
			c[colIndex] += nRows * (intercept + (nRows + 1) * slope / 2);
		}
	}

	@Override
	protected void computeSumSq(double[] c, int nRows) {
		for(int colIndex : _colIndexes) {
			double intercept = _coefficents[0][colIndex];
			double slope = _coefficents[1][colIndex];
			c[0] += nRows * (Math.pow(intercept, 2) + (nRows + 1) * slope * intercept +
				(nRows + 1) * (2 * nRows + 1) * Math.pow(slope, 2) / 6);
		}
	}

	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		for(int colIndex : _colIndexes) {
			double intercept = _coefficents[0][colIndex];
			double slope = _coefficents[1][colIndex];
			c[colIndex] += nRows * (Math.pow(intercept, 2) + (nRows + 1) * slope * intercept +
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

	private synchronized MatrixBlock forceValuesToMatrixBlock() {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup rightMultByMatrix(MatrixBlock right) {
		throw new NotImplementedException();
	}

	@Override
	public void tsmm(double[] ret, int numColumns, int nRows) {
		final int tCol = _colIndexes.length;

		final double sumIndices = nRows * (nRows + 1)/2.0;
		final double sumSquaredIndices = nRows * (nRows + 1) * (2*nRows + 1)/6.0;
		for(int row = 0, offTmp = 0; row < tCol; row++, offTmp += tCol) {
			final int rowIdx = _colIndexes[row];
			final double alpha1 = nRows * _coefficents[0][rowIdx] + sumIndices * _coefficents[1][rowIdx];
			final double alpha2 = sumIndices * _coefficents[0][rowIdx] + sumSquaredIndices * _coefficents[1][rowIdx];
			final int offRet = _colIndexes[row] * numColumns;
			for(int col = row; col < tCol; col++) {
				final int colIdx = _colIndexes[col];
				ret[offRet + _colIndexes[col]] = alpha1 * _coefficents[0][colIdx] + alpha2 * _coefficents[1][colIdx];
			}

		}
	}

	@Override
	public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		throw new DMLCompressionException("This method should never be called");
	}

	@Override
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result) {
		throw new DMLCompressionException("Should not be called");
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
//		CM_COV_Object ret = new CM_COV_Object();
//		op.fn.execute(ret, _dict.getValue(0), nRows);
//		return ret;
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		throw new NotImplementedException();
//		ADictionary d = _dict.rexpandCols(max, ignore, cast, _colIndexes.length);
//		if(d == null)
//			return ColGroupEmpty.create(max);
//		else
//			return create(max, d);
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		throw new NotImplementedException();
//		final int nCols = getNumCols();
//		return e.getCost(nRows, 1, nCols, 1, 1.0);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s", "Coefficents: " + _coefficents.toString()));
		sb.append(_coefficents.length);
		return sb.toString();
	}
}
