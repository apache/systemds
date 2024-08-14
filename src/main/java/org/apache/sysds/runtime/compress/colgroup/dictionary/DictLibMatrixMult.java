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

package org.apache.sysds.runtime.compress.colgroup.dictionary;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Utility interface for dictionary matrix multiplication
 */
public class DictLibMatrixMult {
	static final Log LOG = LogFactory.getLog(DictLibMatrixMult.class.getName());

	private DictLibMatrixMult() {
		// private constructor to avoid init.
	}

	/**
	 * Add to the upper triangle, but twice if on the diagonal
	 * 
	 * @param nCols number cols in res
	 * @param row   the row to add to
	 * @param col   the col to add to
	 * @param res   the double array to add to
	 * @param val   the value to add
	 */
	public static void addToUpperTriangle(int nCols, int row, int col, double[] res, double val) {
		if(row == col) // diagonal add twice
			res[row * nCols + col] += val + val;
		else if(row > col) // swap because in lower triangle
			res[col * nCols + row] += val;
		else
			res[row * nCols + col] += val;
	}

	/**
	 * Matrix multiply with scaling (left side transposed)
	 * 
	 * @param left         Left side dictionary that is not physically transposed but should be treated if it is.
	 * @param right        Right side dictionary that is not transposed and should be used as is.
	 * @param leftRows     Left side row offsets
	 * @param rightColumns Right side column offsets
	 * @param result       The result matrix, normal allocation.
	 * @param counts       The scaling factors
	 */
	public static void MMDictsWithScaling(IDictionary left, IDictionary right, IColIndex leftRows,
		IColIndex rightColumns, MatrixBlock result, int[] counts) {
		left.MMDictScaling(right, leftRows, rightColumns, result, counts);
	}

	/**
	 * Perform the full tsmm with the dictionary (allocating into the entire output matrix.)
	 * 
	 * @param dict   The dictionary to tsmm
	 * @param counts The frequency of each dictionary entry
	 * @param rows   The rows of the dictionary
	 * @param cols   The cols of the dictionary
	 * @param ret    The output to add the results to
	 */
	public static void TSMMDictionaryWithScaling(IDictionary dict, int[] counts, IColIndex rows, IColIndex cols,
		MatrixBlock ret) {
		dict.TSMMWithScaling(counts, rows, cols, ret);
	}

	/**
	 * Matrix Multiply the two dictionaries, note that the left side is considered transposed but not allocated
	 * transposed making the multiplication a: t(left) %*% right
	 * 
	 * @param left      The left side dictionary
	 * @param right     The right side dictionary
	 * @param rowsLeft  The row indexes on the left hand side
	 * @param colsRight The column indexes on the right hand side
	 * @param result    The result matrix to put the results into.
	 */
	public static void MMDicts(IDictionary left, IDictionary right, IColIndex rowsLeft, IColIndex colsRight,
		MatrixBlock result) {
		left.MMDict(right, rowsLeft, colsRight, result);
	}

	/**
	 * Does two matrix multiplications in one go but only add to the upper triangle.
	 * 
	 * the two multiplications are:
	 * 
	 * t(left) %*% right
	 * 
	 * t(right) %*% left
	 * 
	 * In practice this operation then only does one of these multiplications but all results that would end up in the
	 * lower triangle is transposed and added to the upper triangle.
	 * 
	 * Furthermore all results that would end up on the diagonal is added twice to adhere with the two multiplications
	 * 
	 * @param left      Left dictionary to multiply
	 * @param right     Right dictionary to multiply
	 * @param rowsLeft  rows for the left dictionary
	 * @param colsRight cols for the right dictionary
	 * @param result    the result
	 */
	public static void TSMMToUpperTriangle(IDictionary left, IDictionary right, IColIndex rowsLeft, IColIndex colsRight,
		MatrixBlock result) {
		left.TSMMToUpperTriangle(right, rowsLeft, colsRight, result);
	}

	/**
	 * Does two matrix multiplications in one go but only add to the upper triangle with scaling.
	 * 
	 * the two multiplications are:
	 * 
	 * t(left) %*% right
	 * 
	 * t(right) %*% left
	 * 
	 * In practice this operation then only does one of these multiplications but all results that would end up in the
	 * lower triangle is transposed and added to the upper triangle.
	 * 
	 * Furthermore all results that would end up on the diagonal is added twice to adhere with the two multiplications
	 * 
	 * @param left      Left dictionary to multiply
	 * @param right     Right dictionary to multiply
	 * @param rowsLeft  Rows for the left dictionary
	 * @param colsRight Cols for the right dictionary
	 * @param scale     A multiplier to each dictionary entry
	 * @param result    The result
	 */
	public static void TSMMToUpperTriangleScaling(IDictionary left, IDictionary right, IColIndex rowsLeft,
		IColIndex colsRight, int[] scale, MatrixBlock result) {
		left.TSMMToUpperTriangleScaling(left, rowsLeft, colsRight, scale, result);
	}

	protected static void TSMMDictsDenseWithScaling(double[] dv, IColIndex rowsLeft, IColIndex colsRight, int[] scaling,
		MatrixBlock result) {
		final int commonDim = Math.min(dv.length / rowsLeft.size(), dv.length / colsRight.size());
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * rowsLeft.size();
			final int offR = k * colsRight.size();
			final int scale = scaling[k];
			for(int i = 0; i < rowsLeft.size(); i++) {
				final int offOut = rowsLeft.get(i) * resCols;
				final double vl = dv[offL + i] * scale;
				if(vl != 0) {
					for(int j = 0; j < colsRight.size(); j++)
						resV[offOut + colsRight.get(j)] += vl * dv[offR + j];
				}
			}
		}
	}

	protected static void TSMMDictsSparseWithScaling(SparseBlock sb, IColIndex rowsLeft, IColIndex colsRight,
		int[] scaling, MatrixBlock result) {

		final int commonDim = sb.numRows();
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();

		for(int i = 0; i < commonDim; i++) {
			if(sb.isEmpty(i))
				continue;
			final int apos = sb.pos(i);
			final int alen = sb.size(i) + apos;
			final int[] aix = sb.indexes(i);
			final double[] avals = sb.values(i);
			final int scale = scaling[i];
			for(int k = apos; k < alen; k++) {
				final double v = avals[k] * scale;
				final int offOut = rowsLeft.get(aix[k]) * resCols;
				for(int j = apos; j < alen; j++)
					resV[offOut + colsRight.get(aix[j])] += v * avals[j];
			}
		}
	}

	protected static void MMDictsDenseDense(double[] left, double[] right, IColIndex rowsLeft, IColIndex colsRight,
		MatrixBlock result) {
		final int leftSide = rowsLeft.size();
		final int rightSide = colsRight.size();
		final int commonDim = Math.min(left.length / leftSide, right.length / rightSide);
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();

		for(int k = 0; k < commonDim; k++) {
			final int offL = k * leftSide;
			final int offR = k * rightSide;
			for(int i = 0; i < leftSide; i++) {
				final int offOut = rowsLeft.get(i) * resCols;
				final double vl = left[offL + i];
				if(vl != 0) {
					for(int j = 0; j < rightSide; j++)
						resV[offOut + colsRight.get(j)] += vl * right[offR + j];
				}
			}
		}
	}

	protected static void MMDictsScalingDenseDense(double[] left, double[] right, IColIndex rowsLeft,
		IColIndex colsRight, MatrixBlock result, int[] scaling) {
		final int leftSide = rowsLeft.size();
		final int rightSide = colsRight.size();
		final int commonDim = Math.min(left.length / leftSide, right.length / rightSide);
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * leftSide;
			final int offR = k * rightSide;
			final int s = scaling[k];
			for(int i = 0; i < leftSide; i++) {
				final int offOut = rowsLeft.get(i) * resCols;
				final double vl = left[offL + i] * s;
				if(vl != 0) {
					for(int j = 0; j < rightSide; j++)
						resV[offOut + colsRight.get(j)] += vl * right[offR + j];
				}
			}
		}
	}

	protected static void MMDictsSparseDense(SparseBlock left, double[] right, IColIndex rowsLeft, IColIndex colsRight,
		MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.numRows(), right.length / colsRight.size());
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i))
				continue;
			final int apos = left.pos(i);
			final int alen = left.size(i) + apos;
			final int[] aix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int offRight = i * colsRight.size();
			for(int k = apos; k < alen; k++) {
				final int offOut = rowsLeft.get(aix[k]) * result.getNumColumns();
				final double v = leftVals[k];
				for(int j = 0; j < colsRight.size(); j++)
					resV[offOut + colsRight.get(j)] += v * right[offRight + j];
			}
		}
	}

	protected static void MMDictsScalingSparseDense(SparseBlock left, double[] right, IColIndex rowsLeft,
		IColIndex colsRight, MatrixBlock result, int[] scaling) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.numRows(), right.length / colsRight.size());
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i))
				continue;
			final int apos = left.pos(i);
			final int alen = left.size(i) + apos;
			final int[] aix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int offRight = i * colsRight.size();
			final int s = scaling[i];
			for(int k = apos; k < alen; k++) {
				final int offOut = rowsLeft.get(aix[k]) * result.getNumColumns();
				final double v = leftVals[k] * s;
				for(int j = 0; j < colsRight.size(); j++)
					resV[offOut + colsRight.get(j)] += v * right[offRight + j];
			}
		}
	}

	protected static void MMDictsDenseSparse(double[] left, SparseBlock right, IColIndex rowsLeft, IColIndex colsRight,
		MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int leftSize = rowsLeft.size();
		final int commonDim = Math.min(left.length / leftSize, right.numRows());

		for(int i = 0; i < commonDim; i++) {
			if(right.isEmpty(i))
				continue;
			final int apos = right.pos(i);
			final int alen = right.size(i) + apos;
			final int[] aix = right.indexes(i);
			final double[] rightVals = right.values(i);
			final int offLeft = i * leftSize;
			for(int j = 0; j < leftSize; j++) {
				final int offOut = rowsLeft.get(j) * result.getNumColumns();
				final double v = left[offLeft + j];
				if(v != 0) {
					for(int k = apos; k < alen; k++)
						resV[offOut + colsRight.get(aix[k])] += v * rightVals[k];
				}
			}
		}
	}

	protected static void MMDictsScalingDenseSparse(double[] left, SparseBlock right, IColIndex rowsLeft,
		IColIndex colsRight, MatrixBlock result, int[] scaling) {
		final double[] resV = result.getDenseBlockValues();
		final int leftSize = rowsLeft.size();
		final int commonDim = Math.min(left.length / leftSize, right.numRows());

		for(int i = 0; i < commonDim; i++) {
			if(right.isEmpty(i))
				continue;
			final int apos = right.pos(i);
			final int alen = right.size(i) + apos;
			final int[] aix = right.indexes(i);
			final double[] rightVals = right.values(i);
			final int offLeft = i * leftSize;
			final int s = scaling[i];
			for(int j = 0; j < leftSize; j++) {
				final int offOut = rowsLeft.get(j) * result.getNumColumns();
				final double v = left[offLeft + j] * s;
				if(v != 0) {
					for(int k = apos; k < alen; k++)
						resV[offOut + colsRight.get(aix[k])] += v * rightVals[k];
				}
			}
		}
	}

	protected static void MMDictsSparseSparse(SparseBlock left, SparseBlock right, IColIndex rowsLeft,
		IColIndex colsRight, MatrixBlock result) {
		final int commonDim = Math.min(left.numRows(), right.numRows());
		final double[] resV = result.getDenseBlockValues();
		final int resCols = result.getNumColumns();
		// remember that the left side is transposed...
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i) || right.isEmpty(i))
				continue;
			final int leftAPos = left.pos(i);
			final int leftAlen = left.size(i) + leftAPos;
			final int[] leftAix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int rightAPos = right.pos(i);
			final int rightAlen = right.size(i) + rightAPos;
			final int[] rightAix = right.indexes(i);
			final double[] rightVals = right.values(i);

			for(int k = leftAPos; k < leftAlen; k++) {
				final int offOut = rowsLeft.get(leftAix[k]) * resCols;
				final double v = leftVals[k];
				for(int j = rightAPos; j < rightAlen; j++)
					resV[offOut + colsRight.get(rightAix[j])] += v * rightVals[j];
			}
		}
	}

	protected static void MMDictsScalingSparseSparse(SparseBlock left, SparseBlock right, IColIndex rowsLeft,
		IColIndex colsRight, MatrixBlock result, int[] scaling) {
		final int commonDim = Math.min(left.numRows(), right.numRows());
		final double[] resV = result.getDenseBlockValues();
		final int resCols = result.getNumColumns();
		// remember that the left side is transposed...
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i) || right.isEmpty(i))
				continue;
			final int leftAPos = left.pos(i);
			final int leftAlen = left.size(i) + leftAPos;
			final int[] leftAix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int rightAPos = right.pos(i);
			final int rightAlen = right.size(i) + rightAPos;
			final int[] rightAix = right.indexes(i);
			final double[] rightVals = right.values(i);

			final int s = scaling[i];

			for(int k = leftAPos; k < leftAlen; k++) {
				final int offOut = rowsLeft.get(leftAix[k]) * resCols;
				final double v = leftVals[k] * s;
				for(int j = rightAPos; j < rightAlen; j++)
					resV[offOut + colsRight.get(rightAix[j])] += v * rightVals[j];
			}
		}
	}

	protected static void MMToUpperTriangleSparseSparse(SparseBlock left, SparseBlock right, IColIndex rowsLeft,
		IColIndex colsRight, MatrixBlock result) {
		final int commonDim = Math.min(left.numRows(), right.numRows());
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i) || right.isEmpty(i))
				continue;

			final int leftAPos = left.pos(i);
			final int leftAlen = left.size(i) + leftAPos;
			final int[] leftAix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int rightAPos = right.pos(i);
			final int rightAlen = right.size(i) + rightAPos;
			final int[] rightAix = right.indexes(i);
			final double[] rightVals = right.values(i);

			for(int k = leftAPos; k < leftAlen; k++) {
				final int rowOut = rowsLeft.get(leftAix[k]);
				final double vl = leftVals[k];
				for(int j = rightAPos; j < rightAlen; j++) {
					final double vr = rightVals[j];
					final int colOut = colsRight.get(rightAix[j]);
					addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
				}
			}
		}
	}

	protected static void MMToUpperTriangleDenseSparse(double[] left, SparseBlock right, IColIndex rowsLeft,
		IColIndex colsRight, MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.length / rowsLeft.size(), right.numRows());
		final int resCols = result.getNumColumns();
		for(int i = 0; i < commonDim; i++) {
			if(right.isEmpty(i))
				continue;
			final int apos = right.pos(i);
			final int alen = right.size(i) + apos;
			final int[] aix = right.indexes(i);
			final double[] rightVals = right.values(i);
			final int offLeft = i * rowsLeft.size();
			for(int j = 0; j < rowsLeft.size(); j++) {
				final int rowOut = rowsLeft.get(j);
				final double vl = left[offLeft + j];
				if(vl != 0) {
					for(int k = apos; k < alen; k++) {
						final double vr = rightVals[k];
						final int colOut = colsRight.get(aix[k]);
						addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
					}
				}
			}
		}
	}

	protected static void MMToUpperTriangleSparseDense(SparseBlock left, double[] right, IColIndex rowsLeft,
		IColIndex colsRight, MatrixBlock result) {
		final int loc = location(rowsLeft, colsRight);
		if(loc < 0)
			MMToUpperTriangleSparseDenseAllUpperTriangle(left, right, rowsLeft, colsRight, result);
		else if(loc > 0)
			MMToUpperTriangleSparseDenseAllLowerTriangle(left, right, rowsLeft, colsRight, result);
		else
			MMToUpperTriangleSparseDenseDiagonal(left, right, rowsLeft, colsRight, result);
	}

	protected static void MMToUpperTriangleSparseDenseAllUpperTriangle(SparseBlock left, double[] right,
		IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.numRows(), right.length / colsRight.size());
		final int resCols = result.getNumColumns();
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i))
				continue;
			final int apos = left.pos(i);
			final int alen = left.size(i) + apos;
			final int[] aix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int offRight = i * colsRight.size();
			for(int k = apos; k < alen; k++) {
				final int rowOut = rowsLeft.get(aix[k]);
				final double vl = leftVals[k];
				for(int j = 0; j < colsRight.size(); j++)
					resV[colsRight.get(j) * resCols + rowOut] += vl * right[offRight + j];
			}
		}
	}

	protected static void MMToUpperTriangleSparseDenseAllLowerTriangle(SparseBlock left, double[] right,
		IColIndex rowsLeft, IColIndex colsRight, MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.numRows(), right.length / colsRight.size());
		final int resCols = result.getNumColumns();
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i))
				continue;
			final int apos = left.pos(i);
			final int alen = left.size(i) + apos;
			final int[] aix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int offRight = i * colsRight.size();
			for(int k = apos; k < alen; k++) {
				final int rowOut = rowsLeft.get(aix[k]) * resCols;
				final double vl = leftVals[k];
				for(int j = 0; j < colsRight.size(); j++)
					resV[colsRight.get(j) + rowOut] += vl * right[offRight + j];
			}
		}
	}

	protected static void MMToUpperTriangleSparseDenseDiagonal(SparseBlock left, double[] right, IColIndex rowsLeft,
		IColIndex colsRight, MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.numRows(), right.length / colsRight.size());
		final int resCols = result.getNumColumns();
		// generic
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i))
				continue;
			final int apos = left.pos(i);
			final int alen = left.size(i) + apos;
			final int[] aix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int offRight = i * colsRight.size();
			for(int k = apos; k < alen; k++) {
				final int rowOut = rowsLeft.get(aix[k]);
				final double vl = leftVals[k];
				for(int j = 0; j < colsRight.size(); j++) {
					final double vr = right[offRight + j];
					final int colOut = colsRight.get(j);
					addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
				}
			}
		}
	}

	protected static void MMToUpperTriangleDenseDense(double[] left, double[] right, IColIndex rowsLeft,
		IColIndex colsRight, MatrixBlock result) {
		final int loc = location(rowsLeft, colsRight);
		if(loc < 0)
			MMToUpperTriangleDenseDenseAllUpperTriangle(left, right, rowsLeft, colsRight, result);
		else if(loc > 0)
			MMToUpperTriangleDenseDenseAllLowerTriangle(left, right, rowsLeft, colsRight, result);
		else
			MMToUpperTriangleDenseDenseDiagonal(left, right, rowsLeft, colsRight, result);
	}

	protected static void MMToUpperTriangleDenseDenseAllUpperTriangle(double[] left, double[] right, IColIndex rowsLeft,
		IColIndex colsRight, MatrixBlock result) {
		final int lSize = rowsLeft.size();
		final int rSize = colsRight.size();
		final int commonDim = Math.min(left.length / lSize, right.length / rSize);
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int i = 0; i < lSize; i++) {
			MMToUpperTriangleDenseDenseAllUpperTriangleRow(left, right, rowsLeft.get(i), colsRight, commonDim, lSize,
				rSize, i, resV, resCols);
		}
	}

	protected static void MMToUpperTriangleDenseDenseAllUpperTriangleRow(final double[] left, final double[] right,
		final int rowOut, final IColIndex colsRight, final int commonDim, final int lSize, final int rSize, final int i,
		final double[] resV, final int resCols) {
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * lSize;
			final double vl = left[offL + i];
			if(vl != 0) {
				final int offR = k * rSize;
				for(int j = 0; j < rSize; j++)
					resV[colsRight.get(j) * resCols + rowOut] += vl * right[offR + j];
			}
		}
	}

	protected static void MMToUpperTriangleDenseDenseAllLowerTriangle(double[] left, double[] right, IColIndex rowsLeft,
		IColIndex colsRight, MatrixBlock result) {
		final int commonDim = Math.min(left.length / rowsLeft.size(), right.length / colsRight.size());
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * rowsLeft.size();
			final int offR = k * colsRight.size();
			for(int i = 0; i < rowsLeft.size(); i++) {
				final int rowOut = rowsLeft.get(i) * resCols;
				final double vl = left[offL + i];
				for(int j = 0; j < colsRight.size(); j++)
					resV[colsRight.get(j) + rowOut] += vl * right[offR + j];

			}
		}
	}

	protected static void MMToUpperTriangleDenseDenseDiagonal(double[] left, double[] right, IColIndex rowsLeft,
		IColIndex colsRight, MatrixBlock result) {
		final int commonDim = Math.min(left.length / rowsLeft.size(), right.length / colsRight.size());
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * rowsLeft.size();
			final int offR = k * colsRight.size();
			for(int i = 0; i < rowsLeft.size(); i++) {
				final int rowOut = rowsLeft.get(i);
				final double vl = left[offL + i];
				for(int j = 0; j < colsRight.size(); j++) {
					final double vr = right[offR + j];
					final int colOut = colsRight.get(j);
					addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
				}
			}
		}
	}

	protected static void TSMMToUpperTriangleSparseSparseScaling(SparseBlock left, SparseBlock right, IColIndex rowsLeft,
		IColIndex colsRight, int[] scale, MatrixBlock result) {
		final int commonDim = Math.min(left.numRows(), right.numRows());
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i) || right.isEmpty(i))
				continue;

			final int leftAPos = left.pos(i);
			final int leftAlen = left.size(i) + leftAPos;
			final int[] leftAix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int rightAPos = right.pos(i);
			final int rightAlen = right.size(i) + rightAPos;
			final int[] rightAix = right.indexes(i);
			final double[] rightVals = right.values(i);
			final double sv = scale[i];
			for(int k = leftAPos; k < leftAlen; k++) {
				final int rowOut = rowsLeft.get(leftAix[k]);
				final double vl = leftVals[k] * sv;
				for(int j = rightAPos; j < rightAlen; j++) {
					final double vr = rightVals[j];
					final int colOut = colsRight.get(rightAix[j]);
					addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
				}
			}
		}
	}

	protected static void TSMMToUpperTriangleDenseSparseScaling(double[] left, SparseBlock right, IColIndex rowsLeft,
		IColIndex colsRight, int[] scale, MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.length / rowsLeft.size(), right.numRows());
		final int resCols = result.getNumColumns();
		for(int i = 0; i < commonDim; i++) {
			if(right.isEmpty(i))
				continue;
			final int apos = right.pos(i);
			final int alen = right.size(i) + apos;
			final int[] aix = right.indexes(i);
			final double[] rightVals = right.values(i);
			final int offLeft = i * rowsLeft.size();
			final double sv = scale[i];
			for(int j = 0; j < rowsLeft.size(); j++) {
				final int rowOut = rowsLeft.get(j);
				final double vl = left[offLeft + j] * sv;
				if(vl != 0) {
					for(int k = apos; k < alen; k++) {
						final double vr = rightVals[k];
						final int colOut = colsRight.get(aix[k]);
						addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
					}
				}
			}
		}
	}

	protected static void TSMMToUpperTriangleSparseDenseScaling(SparseBlock left, double[] right, IColIndex rowsLeft,
		IColIndex colsRight, int[] scale, MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.numRows(), right.length / colsRight.size());
		final int resCols = result.getNumColumns();
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i))
				continue;
			final int apos = left.pos(i);
			final int alen = left.size(i) + apos;
			final int[] aix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int offRight = i * colsRight.size();
			final double sv = scale[i];
			for(int k = apos; k < alen; k++) {
				final int rowOut = rowsLeft.get(aix[k]);
				final double vl = leftVals[k] * sv;
				for(int j = 0; j < colsRight.size(); j++) {
					final double vr = right[offRight + j];
					final int colOut = colsRight.get(j);
					addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
				}
			}
		}
	}

	protected static void TSMMToUpperTriangleDenseDenseScaling(double[] left, double[] right, IColIndex rowsLeft,
		IColIndex colsRight, int[] scale, MatrixBlock result) {
		final int commonDim = Math.min(left.length / rowsLeft.size(), right.length / colsRight.size());
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * rowsLeft.size();
			final int offR = k * colsRight.size();
			final int sv = scale[k];
			for(int i = 0; i < rowsLeft.size(); i++) {
				final int rowOut = rowsLeft.get(i);
				final double vl = left[offL + i] * sv;
				if(vl != 0) {
					for(int j = 0; j < colsRight.size(); j++) {
						final double vr = right[offR + j];
						final int colOut = colsRight.get(j);
						addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
					}
				}
			}
		}
	}

	private static int location(IColIndex leftRows, IColIndex rightColumns) {
		final int firstRow = leftRows.get(0);
		final int firstCol = rightColumns.get(0);
		final int lastRow = leftRows.get(leftRows.size() - 1);
		final int lastCol = rightColumns.get(rightColumns.size() - 1);
		final int locationLower = location(lastRow, firstCol);
		final int locationHigher = location(firstRow, lastCol);

		if(locationLower > 0)
			return 1;
		else if(locationHigher < 0)
			return -1;
		else
			return 0;
	}

	private static int location(int row, int col) {
		if(row == col)
			return 0;
		else if(row < col)
			return 1;
		else
			return -1;
	}
}
