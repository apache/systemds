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
	 * @param left         Left side dictionary
	 * @param right        Right side dictionary
	 * @param leftRows     Left side row offsets
	 * @param rightColumns Right side column offsets
	 * @param result       The result matrix
	 * @param counts       The scaling factors
	 */
	public static void MMDictsWithScaling(ADictionary left, ADictionary right, int[] leftRows, int[] rightColumns,
		MatrixBlock result, int[] counts) {
		LOG.warn("Inefficient double allocation of dictionary");
		final boolean modifyRight = right.getInMemorySize() > left.getInMemorySize();
		final ADictionary rightM = modifyRight ? right.scaleTuples(counts, rightColumns.length) : right;
		final ADictionary leftM = modifyRight ? left : left.scaleTuples(counts, leftRows.length);
		MMDicts(leftM, rightM, leftRows, rightColumns, result);
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
	public static void TSMMDictionaryWithScaling(ADictionary dict, int[] counts, int[] rows, int[] cols,
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
	public static void MMDicts(ADictionary left, ADictionary right, int[] rowsLeft, int[] colsRight,
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
	public static void TSMMToUpperTriangle(ADictionary left, ADictionary right, int[] rowsLeft, int[] colsRight,
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
	public static void TSMMToUpperTriangleScaling(ADictionary left, ADictionary right, int[] rowsLeft, int[] colsRight,
		int[] scale, MatrixBlock result) {
		left.TSMMToUpperTriangleScaling(left, rowsLeft, colsRight, scale, result);
	}

	protected static void TSMMDictsDenseWithScaling(double[] dv, int[] rowsLeft, int[] colsRight, int[] scaling,
		MatrixBlock result) {
		final int commonDim = Math.min(dv.length / rowsLeft.length, dv.length / colsRight.length);
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * rowsLeft.length;
			final int offR = k * colsRight.length;
			final int scale = scaling[k];
			for(int i = 0; i < rowsLeft.length; i++) {
				final int offOut = rowsLeft[i] * resCols;
				final double vl = dv[offL + i] * scale;
				if(vl != 0) {
					for(int j = 0; j < colsRight.length; j++)
						resV[offOut + colsRight[j]] += vl * dv[offR + j];
				}
			}
		}
	}

	protected static void TSMMDictsSparseWithScaling(SparseBlock sb, int[] rowsLeft, int[] colsRight, int[] scaling,
		MatrixBlock result) {

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
				final int offOut = rowsLeft[aix[k]] * resCols;
				for(int j = apos; j < alen; j++)
					resV[offOut + colsRight[aix[j]]] += v * avals[j];
			}
		}
	}

	protected static void MMDictsDenseDense(double[] left, double[] right, int[] rowsLeft, int[] colsRight,
		MatrixBlock result) {
		final int commonDim = Math.min(left.length / rowsLeft.length, right.length / colsRight.length);
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * rowsLeft.length;
			final int offR = k * colsRight.length;
			for(int i = 0; i < rowsLeft.length; i++) {
				final int offOut = rowsLeft[i] * resCols;
				final double vl = left[offL + i];
				if(vl != 0) {
					for(int j = 0; j < colsRight.length; j++)
						resV[offOut + colsRight[j]] += vl * right[offR + j];
				}
			}
		}
	}

	protected static void MMDictsSparseDense(SparseBlock left, double[] right, int[] rowsLeft, int[] colsRight,
		MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.numRows(), right.length / colsRight.length);
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i))
				continue;
			final int apos = left.pos(i);
			final int alen = left.size(i) + apos;
			final int[] aix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int offRight = i * colsRight.length;
			for(int k = apos; k < alen; k++) {
				final int offOut = rowsLeft[aix[k]] * result.getNumColumns();
				final double v = leftVals[k];
				for(int j = 0; j < colsRight.length; j++)
					resV[offOut + colsRight[j]] += v * right[offRight + j];
			}
		}
	}

	protected static void MMDictsDenseSparse(double[] left, SparseBlock right, int[] rowsLeft, int[] colsRight,
		MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.length / rowsLeft.length, right.numRows());
		for(int i = 0; i < commonDim; i++) {
			if(right.isEmpty(i))
				continue;
			final int apos = right.pos(i);
			final int alen = right.size(i) + apos;
			final int[] aix = right.indexes(i);
			final double[] rightVals = right.values(i);
			final int offLeft = i * rowsLeft.length;
			for(int j = 0; j < rowsLeft.length; j++) {
				final int offOut = rowsLeft[j] * result.getNumColumns();
				final double v = left[offLeft + j];
				if(v != 0) {
					for(int k = apos; k < alen; k++)
						resV[offOut + colsRight[aix[k]]] += v * rightVals[k];
				}
			}
		}
	}

	protected static void MMDictsSparseSparse(SparseBlock left, SparseBlock right, int[] rowsLeft, int[] colsRight,
		MatrixBlock result) {
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
				final int offOut = rowsLeft[leftAix[k]] * resCols;
				final double v = leftVals[k];
				for(int j = rightAPos; j < rightAlen; j++)
					resV[offOut + colsRight[rightAix[j]]] += v * rightVals[j];
			}
		}
	}

	protected static void MMToUpperTriangleSparseSparse(SparseBlock left, SparseBlock right, int[] rowsLeft,
		int[] colsRight, MatrixBlock result) {
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
				final int rowOut = rowsLeft[leftAix[k]];
				final double vl = leftVals[k];
				for(int j = rightAPos; j < rightAlen; j++) {
					final double vr = rightVals[j];
					final int colOut = colsRight[rightAix[j]];
					addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
				}
			}
		}
	}

	protected static void MMToUpperTriangleDenseSparse(double[] left, SparseBlock right, int[] rowsLeft, int[] colsRight,
		MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.length / rowsLeft.length, right.numRows());
		final int resCols = result.getNumColumns();
		for(int i = 0; i < commonDim; i++) {
			if(right.isEmpty(i))
				continue;
			final int apos = right.pos(i);
			final int alen = right.size(i) + apos;
			final int[] aix = right.indexes(i);
			final double[] rightVals = right.values(i);
			final int offLeft = i * rowsLeft.length;
			for(int j = 0; j < rowsLeft.length; j++) {
				final int rowOut = rowsLeft[j];
				final double vl = left[offLeft + j];
				if(vl != 0) {
					for(int k = apos; k < alen; k++) {
						final double vr = rightVals[k];
						final int colOut = colsRight[aix[k]];
						addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
					}
				}
			}
		}
	}

	protected static void MMToUpperTriangleSparseDense(SparseBlock left, double[] right, int[] rowsLeft, int[] colsRight,
		MatrixBlock result) {
		final int loc = location(rowsLeft, colsRight);
		if(loc < 0)
			MMToUpperTriangleSparseDenseAllUpperTriangle(left, right, rowsLeft, colsRight, result);
		else if(loc > 0)
			MMToUpperTriangleSparseDenseAllLowerTriangle(left, right, rowsLeft, colsRight, result);
		else
			MMToUpperTriangleSparseDenseDiagonal(left, right, rowsLeft, colsRight, result);
	}

	protected static void MMToUpperTriangleSparseDenseAllUpperTriangle(SparseBlock left, double[] right, int[] rowsLeft,
		int[] colsRight, MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.numRows(), right.length / colsRight.length);
		final int resCols = result.getNumColumns();
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i))
				continue;
			final int apos = left.pos(i);
			final int alen = left.size(i) + apos;
			final int[] aix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int offRight = i * colsRight.length;
			for(int k = apos; k < alen; k++) {
				final int rowOut = rowsLeft[aix[k]];
				final double vl = leftVals[k];
				for(int j = 0; j < colsRight.length; j++)
					resV[colsRight[j] * resCols + rowOut] += vl * right[offRight + j];
			}
		}
	}

	protected static void MMToUpperTriangleSparseDenseAllLowerTriangle(SparseBlock left, double[] right, int[] rowsLeft,
		int[] colsRight, MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.numRows(), right.length / colsRight.length);
		final int resCols = result.getNumColumns();
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i))
				continue;
			final int apos = left.pos(i);
			final int alen = left.size(i) + apos;
			final int[] aix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int offRight = i * colsRight.length;
			for(int k = apos; k < alen; k++) {
				final int rowOut = rowsLeft[aix[k]] * resCols;
				final double vl = leftVals[k];
				for(int j = 0; j < colsRight.length; j++)
					resV[colsRight[j] + rowOut] += vl * right[offRight + j];
			}
		}
	}

	protected static void MMToUpperTriangleSparseDenseDiagonal(SparseBlock left, double[] right, int[] rowsLeft,
		int[] colsRight, MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.numRows(), right.length / colsRight.length);
		final int resCols = result.getNumColumns();
		// generic
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i))
				continue;
			final int apos = left.pos(i);
			final int alen = left.size(i) + apos;
			final int[] aix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int offRight = i * colsRight.length;
			for(int k = apos; k < alen; k++) {
				final int rowOut = rowsLeft[aix[k]];
				final double vl = leftVals[k];
				for(int j = 0; j < colsRight.length; j++) {
					final double vr = right[offRight + j];
					final int colOut = colsRight[j];
					addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
				}
			}
		}
	}

	protected static void MMToUpperTriangleDenseDense(double[] left, double[] right, int[] rowsLeft, int[] colsRight,
		MatrixBlock result) {
		final int loc = location(rowsLeft, colsRight);
		if(loc < 0)
			MMToUpperTriangleDenseDenseAllUpperTriangle(left, right, rowsLeft, colsRight, result);
		else if(loc > 0)
			MMToUpperTriangleDenseDenseAllLowerTriangle(left, right, rowsLeft, colsRight, result);
		else
			MMToUpperTriangleDenseDenseDiagonal(left, right, rowsLeft, colsRight, result);
	}

	protected static void MMToUpperTriangleDenseDenseAllUpperTriangle(double[] left, double[] right, int[] rowsLeft,
		int[] colsRight, MatrixBlock result) {
		final int commonDim = Math.min(left.length / rowsLeft.length, right.length / colsRight.length);
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * rowsLeft.length;
			final int offR = k * colsRight.length;
			for(int i = 0; i < rowsLeft.length; i++) {
				final int rowOut = rowsLeft[i];
				final double vl = left[offL + i];
				if(vl != 0) {
					for(int j = 0; j < colsRight.length; j++)
						resV[colsRight[j] * resCols + rowOut] += vl * right[offR + j];
				}
			}
		}
	}

	protected static void MMToUpperTriangleDenseDenseAllLowerTriangle(double[] left, double[] right, int[] rowsLeft,
		int[] colsRight, MatrixBlock result) {
		final int commonDim = Math.min(left.length / rowsLeft.length, right.length / colsRight.length);
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * rowsLeft.length;
			final int offR = k * colsRight.length;
			for(int i = 0; i < rowsLeft.length; i++) {
				final int rowOut = rowsLeft[i] * resCols;
				final double vl = left[offL + i];
				for(int j = 0; j < colsRight.length; j++)
					resV[colsRight[j] + rowOut] += vl * right[offR + j];

			}
		}
	}

	protected static void MMToUpperTriangleDenseDenseDiagonal(double[] left, double[] right, int[] rowsLeft,
		int[] colsRight, MatrixBlock result) {
		final int commonDim = Math.min(left.length / rowsLeft.length, right.length / colsRight.length);
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * rowsLeft.length;
			final int offR = k * colsRight.length;
			for(int i = 0; i < rowsLeft.length; i++) {
				final int rowOut = rowsLeft[i];
				final double vl = left[offL + i];
				for(int j = 0; j < colsRight.length; j++) {
					final double vr = right[offR + j];
					final int colOut = colsRight[j];
					addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
				}
			}
		}
	}

	protected static void TSMMToUpperTriangleSparseSparseScaling(SparseBlock left, SparseBlock right, int[] rowsLeft,
		int[] colsRight, int[] scale, MatrixBlock result) {
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
				final int rowOut = rowsLeft[leftAix[k]];
				final double vl = leftVals[k] * sv;
				for(int j = rightAPos; j < rightAlen; j++) {
					final double vr = rightVals[j];
					final int colOut = colsRight[rightAix[j]];
					addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
				}
			}
		}
	}

	protected static void TSMMToUpperTriangleDenseSparseScaling(double[] left, SparseBlock right, int[] rowsLeft,
		int[] colsRight, int[] scale, MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.length / rowsLeft.length, right.numRows());
		final int resCols = result.getNumColumns();
		for(int i = 0; i < commonDim; i++) {
			if(right.isEmpty(i))
				continue;
			final int apos = right.pos(i);
			final int alen = right.size(i) + apos;
			final int[] aix = right.indexes(i);
			final double[] rightVals = right.values(i);
			final int offLeft = i * rowsLeft.length;
			final double sv = scale[i];
			for(int j = 0; j < rowsLeft.length; j++) {
				final int rowOut = rowsLeft[j];
				final double vl = left[offLeft + j] * sv;
				if(vl != 0) {
					for(int k = apos; k < alen; k++) {
						final double vr = rightVals[k];
						final int colOut = colsRight[aix[k]];
						addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
					}
				}
			}
		}
	}

	protected static void TSMMToUpperTriangleSparseDenseScaling(SparseBlock left, double[] right, int[] rowsLeft,
		int[] colsRight, int[] scale, MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.numRows(), right.length / colsRight.length);
		final int resCols = result.getNumColumns();
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i))
				continue;
			final int apos = left.pos(i);
			final int alen = left.size(i) + apos;
			final int[] aix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int offRight = i * colsRight.length;
			final double sv = scale[i];
			for(int k = apos; k < alen; k++) {
				final int rowOut = rowsLeft[aix[k]];
				final double vl = leftVals[k] * sv;
				for(int j = 0; j < colsRight.length; j++) {
					final double vr = right[offRight + j];
					final int colOut = colsRight[j];
					addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
				}
			}
		}
	}

	protected static void TSMMToUpperTriangleDenseDenseScaling(double[] left, double[] right, int[] rowsLeft,
		int[] colsRight, int[] scale, MatrixBlock result) {
		final int commonDim = Math.min(left.length / rowsLeft.length, right.length / colsRight.length);
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * rowsLeft.length;
			final int offR = k * colsRight.length;
			final int sv = scale[k];
			for(int i = 0; i < rowsLeft.length; i++) {
				final int rowOut = rowsLeft[i];
				final double vl = left[offL + i] * sv;
				if(vl != 0) {
					for(int j = 0; j < colsRight.length; j++) {
						final double vr = right[offR + j];
						final int colOut = colsRight[j];
						addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
					}
				}
			}
		}
	}

	private static int location(int[] leftRows, int[] rightColumns) {
		final int firstRow = leftRows[0];
		final int firstCol = rightColumns[0];
		final int lastRow = leftRows[leftRows.length - 1];
		final int lastCol = rightColumns[rightColumns.length - 1];
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
