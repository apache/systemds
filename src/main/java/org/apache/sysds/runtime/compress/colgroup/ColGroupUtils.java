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

import java.util.Arrays;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.utils.DoubleCountHashMap;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public interface ColGroupUtils {
	public static final Log LOG = LogFactory.getLog(ColGroupUtils.class.getName());

	/**
	 * Calculate the result of performing the binary operation on an empty row to the left
	 * 
	 * v op empty
	 * 
	 * @param op         The operator
	 * @param v          The values to use on the left side of the operator
	 * @param colIndexes The column indexes to extract
	 * @return The result as a double array.
	 */
	public static double[] binaryDefRowLeft(BinaryOperator op, double[] v, IColIndex colIndexes) {
		final ValueFunction fn = op.fn;
		final int len = colIndexes.size();
		final double[] ret = new double[len];
		for(int i = 0; i < len; i++)
			ret[i] = fn.execute(v[colIndexes.get(i)], 0);
		return ret;
	}

	/**
	 * Calculate the result of performing the binary operation on an empty row to the right
	 * 
	 * empty op v
	 * 
	 * @param op         The operator
	 * @param v          The values to use on the left side of the operator
	 * @param colIndexes The column indexes to extract
	 * @return The result as a double array.
	 */
	public static double[] binaryDefRowRight(BinaryOperator op, double[] v, IColIndex colIndexes) {
		final ValueFunction fn = op.fn;
		final int len = colIndexes.size();
		final double[] ret = new double[len];
		for(int i = 0; i < len; i++)
			ret[i] = fn.execute(0, v[colIndexes.get(i)]);
		return ret;
	}

	/**
	 * Copy values from tmpResult into correct positions of result (according to colIndexes in lhs and rhs)
	 *
	 * @param lhs       Left ColumnGroup
	 * @param rhs       Right ColumnGroup
	 * @param tmpResult The matrix block to move values from
	 * @param result    The result matrix block to move values to
	 */
	public static void copyValuesColGroupMatrixBlocks(AColGroup lhs, AColGroup rhs, MatrixBlock tmpResult,
		MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		if(tmpResult.isEmpty())
			return;
		else if(tmpResult.isInSparseFormat()) {
			SparseBlock sb = tmpResult.getSparseBlock();
			for(int row = 0; row < lhs._colIndexes.size(); row++) {
				if(sb.isEmpty(row))
					continue;
				final int apos = sb.pos(row);
				final int alen = sb.size(row) + apos;
				final int[] aix = sb.indexes(row);
				final double[] avals = sb.values(row);
				final int offRes = lhs._colIndexes.get(row) * result.getNumColumns();
				for(int col = apos; col < alen; col++)
					resV[offRes + rhs._colIndexes.get(aix[col])] += avals[col];
			}
		}
		else {
			double[] tmpRetV = tmpResult.getDenseBlockValues();
			for(int row = 0; row < lhs.getNumCols(); row++) {
				final int offRes = lhs._colIndexes.get(row) * result.getNumColumns();
				final int offTmp = row * rhs.getNumCols();
				for(int col = 0; col < rhs.getNumCols(); col++) {
					resV[offRes + rhs._colIndexes.get(col)] += tmpRetV[offTmp + col];
				}
			}
		}
	}

	/*
	 * Returns null if all zero
	 * 
	 * @param mb Matrix Block to find most common value in all columns.
	 * 
	 * @return Double array with most common values.
	 */
	public static double[] extractMostCommonValueInColumns(MatrixBlock mb) {
		final int nCol = mb.getNumColumns();
		final int nVal = mb.getNumRows();
		final int[] nnz = LibMatrixReorg.countNnzPerColumn(mb);

		double[] ref = new double[nCol];
		boolean contains = false;
		for(int i = 0; i < nCol; i++) {
			if(nnz[i] > nVal / 2) {
				contains = true;
				ref[i] = 1;
			}
		}

		if(contains)
			getMostCommonValues(mb, ref, nnz);
		else
			return null;

		contains = false;
		for(int i = 0; i < nCol; i++)
			if(ref[i] != 0) {
				contains = true;
				break;
			}
		if(contains == false)
			return null;
		else
			return ref;
	}

	public static double refSum(double[] reference) {
		double ret = 0;
		for(double d : reference)
			ret += d;
		return ret;
	}

	public static double refSumSq(double[] reference) {
		double ret = 0;
		for(double d : reference)
			ret += d * d;
		return ret;
	}

	public static boolean allZero(double[] in) {
		for(double v : in)
			if(v != 0)
				return false;
		return true;
	}

	public static boolean containsInfOrNan(double pattern, double[] reference) {
		if(Double.isNaN(pattern)) {
			for(double d : reference)
				if(Double.isNaN(d))
					return true;
			return false;
		}
		else {
			for(double d : reference)
				if(Double.isInfinite(d))
					return true;
			return false;
		}
	}

	public static double[] createReference(int nCol, double val) {
		double[] reference = new double[nCol];
		for(int i = 0; i < nCol; i++)
			reference[i] = val;
		return reference;
	}

	public static double[] unaryOperator(UnaryOperator op, double[] reference) {
		final double[] newRef = new double[reference.length];
		for(int i = 0; i < reference.length; i++)
			newRef[i] = op.fn.execute(reference[i]);
		return newRef;
	}

	public static void outerProduct(final double[] leftRowSum, final double[] rightColumnSum, final double[] result,
		int rl, int ru) {
		for(int row = rl; row < ru; row++) {
			final int offOut = rightColumnSum.length * row;
			final double vLeft = leftRowSum[row];
			for(int col = 0; col < rightColumnSum.length; col++) {
				result[offOut + col] += vLeft * rightColumnSum[col];
			}
		}
	}

	public static void outerProduct(final double[] leftRowSum, final double[] rightColumnSum,
		final IColIndex colIdxRight, final double[] result, final int nColR, final int rl, final int ru) {
		for(int row = rl; row < ru; row++) {
			final int offOut = nColR * row;
			final double vLeft = leftRowSum[row];
			for(int col = 0; col < rightColumnSum.length; col++)
				result[offOut + colIdxRight.get(col)] += vLeft * rightColumnSum[col];
		}
	}

	public static void outerProduct(final double[] leftRowSum, final SparseBlock rightColSum,
		final IColIndex colIdxRight, final double[] result, final int nColR, final int rl, final int ru) {
		final int alen = rightColSum.size(0);
		final int[] aix = rightColSum.indexes(0);
		final double[] aval = rightColSum.values(0);
		for(int row = rl; row < ru; row++) {
			final int offOut = nColR * row;
			final double vLeft = leftRowSum[row];
			for(int j = 0; j < alen; j++)
				result[offOut + colIdxRight.get(aix[j])] += vLeft * aval[j];
		}
	}

	private static void getMostCommonValues(MatrixBlock mb, double[] ref, int[] nnzCols) {
		// take each column marked by ref and find most common value in that and assign it to ref.
		// if the columns are

		DoubleCountHashMap[] counters = new DoubleCountHashMap[ref.length];

		if(mb.isInSparseFormat()) {
			// initialize the counters with zero count.
			for(int i = 0; i < ref.length; i++) {
				if(ref[i] != 0) {
					counters[i] = new DoubleCountHashMap(8);
					counters[i].increment(0.0, nnzCols[i]);
				}
			}
			final SparseBlock sb = mb.getSparseBlock();
			for(int r = 0; r < mb.getNumRows(); r++) {
				if(sb.isEmpty(r))
					continue;
				final int apos = sb.pos(r);
				final int alen = sb.size(r) + apos;
				final int[] aix = sb.indexes(r);
				final double[] aval = sb.values(r);
				for(int j = apos; j < alen; j++)
					if(ref[aix[j]] != 0)
						counters[aix[j]].increment(aval[j]);
			}
		}
		else {
			for(int i = 0; i < ref.length; i++)
				if(ref[i] != 0)
					counters[i] = new DoubleCountHashMap(8);
			double[] dv = mb.getDenseBlockValues();
			final int nCol = ref.length;
			for(int r = 0; r < mb.getNumRows(); r++) {
				final int rOff = r * nCol;
				for(int c = 0; c < nCol; c++)
					if(ref[c] != 0)
						counters[c].increment(dv[rOff + c]);

			}
		}
		for(int i = 0; i < ref.length; i++)
			if(ref[i] != 0)
				ref[i] = counters[i].getMostFrequent();
	}

	public static void addMatrixToResult(MatrixBlock tmp, MatrixBlock result, IColIndex colIndexes, int rl, int ru) {
		if(tmp.isEmpty())
			return;
		final double[] retV = result.getDenseBlockValues();
		final int nColRet = result.getNumColumns();
		if(tmp.isInSparseFormat()) {
			final SparseBlock sb = tmp.getSparseBlock();
			for(int row = rl, offT = 0; row < ru; row++, offT++) {
				if(sb.isEmpty(offT))
					continue;
				final int apos = sb.pos(offT);
				final int alen = sb.size(offT);
				final int[] aix = sb.indexes(offT);
				final double[] avals = sb.values(offT);
				final int offR = row * nColRet;
				for(int i = apos; i < apos + alen; i++)
					retV[offR + colIndexes.get(aix[i])] += avals[i];
			}
		}
		else {
			final double[] tmpV = tmp.getDenseBlockValues();
			final int nCol = colIndexes.size();
			for(int row = rl, offT = 0; row < ru; row++, offT += nCol) {
				final int offR = row * nColRet;
				for(int col = 0; col < nCol; col++)
					retV[offR + colIndexes.get(col)] += tmpV[offT + col];
			}
		}
	}

	public static double[] reorderDefault(double[] vals, int[] reordering) {
		double[] ret = new double[vals.length];
		for(int i = 0; i < vals.length; i++)
			ret[i] = vals[reordering[i]];
		return ret;
	}

	/**
	 * Get a list of points locations from the SparseBlock.
	 * 
	 * This is used to find 1 indexes in a sparse selection matrix.
	 * 
	 * We assume the input only have one non zero per row, and that non zero is a 1.
	 * 
	 * @param sb Sparse block to extract points from
	 * @param rl row to start from
	 * @param ru row to end at
	 * @return The coordinates that contain values.
	 */
	public static P[] getSortedSelection(SparseBlock sb, int rl, int ru) {

		int c = 0;
		// count loop
		for(int i = rl; i < ru; i++) {
			if(!sb.isEmpty(i))
				c++;
		}

		P[] points = new P[c];
		c = 0; // count from start again
		for(int i = rl; i < ru; i++) {
			if(sb.isEmpty(i))
				continue;
			final int sPos = sb.pos(i);
			points[c++] = new P(i, sb.indexes(i)[sPos]);
		}

		Arrays.sort(points, (a, b) -> {
			return a.o < b.o ? -1 : 1;
		});
		return points;
	}

	public static class P {
		public final int r;
		public final int o;

		private P(int r, int o) {
			this.r = r;
			this.o = o;
		}

		@Override
		public String toString() {
			return "P(" + r + "," + o + ")";
		}
	}

}
