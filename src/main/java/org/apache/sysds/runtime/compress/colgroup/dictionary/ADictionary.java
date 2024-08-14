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

import java.io.Serializable;

import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;

/**
 * This dictionary class aims to encapsulate the storage and operations over unique tuple values of a column group.
 */
public abstract class ADictionary implements IDictionary, Serializable {
	private static final long serialVersionUID = 9118692576356558592L;

	public abstract IDictionary clone();

	public final CM_COV_Object centralMoment(ValueFunction fn, int[] counts, int nRows) {
		return centralMoment(new CM_COV_Object(), fn, counts, nRows);
	}

	public final CM_COV_Object centralMomentWithDefault(ValueFunction fn, int[] counts, double def, int nRows) {
		return centralMomentWithDefault(new CM_COV_Object(), fn, counts, def, nRows);
	}

	public final CM_COV_Object centralMomentWithReference(ValueFunction fn, int[] counts, double reference, int nRows) {
		return centralMomentWithReference(new CM_COV_Object(), fn, counts, reference, nRows);
	}

	@Override
	public final boolean equals(Object o) {
		if(o instanceof IDictionary)
			return equals((IDictionary) o);
		return false;
	}

	@Override
	public final boolean equals(double[] v) {
		return equals(new Dictionary(v));
	}

	/**
	 * Make a double into a string, if the double is a whole number then return it without decimal points
	 * 
	 * @param v The value
	 * @return The string
	 */
	public static String doubleToString(double v) {
		if(v == (long) v)
			return Long.toString(((long) v));
		else
			return Double.toString(v);
	}

	/**
	 * Correct Nan Values in an result. If there are any NaN values in the given Res then they are replaced with 0.
	 * 
	 * @param res        The array to correct
	 * @param colIndexes The column indexes.
	 */
	public static void correctNan(double[] res, IColIndex colIndexes) {
		// since there is no nan values in most dictionaries, we exploit that
		// nan only occur if we multiplied infinity with 0.
		for(int j = 0; j < colIndexes.size(); j++) {
			final int cix = colIndexes.get(j);
			res[cix] = Double.isNaN(res[cix]) ? 0 : res[cix];
		}
	}

	@Override
	public IDictionary rightMMPreAggSparse(int numVals, SparseBlock b, IColIndex thisCols, IColIndex aggregateColumns,
		int nColRight) {
		if(aggregateColumns.size() < nColRight)
			return rightMMPreAggSparseSelectedCols(numVals, b, thisCols, aggregateColumns);
		else
			return rightMMPreAggSparseAllColsRight(numVals, b, thisCols, nColRight);
	}

	protected IDictionary rightMMPreAggSparseSelectedCols(int numVals, SparseBlock b, IColIndex thisCols,
		IColIndex aggregateColumns) {

		final int thisColsSize = thisCols.size();
		final int aggColSize = aggregateColumns.size();
		final double[] ret = new double[numVals * aggColSize];

		for(int h = 0; h < thisColsSize; h++) {
			// chose row in right side matrix via column index of the dictionary
			final int colIdx = thisCols.get(h);
			if(b.isEmpty(colIdx))
				continue;

			// extract the row values on the right side.
			final double[] sValues = b.values(colIdx);
			final int[] sIndexes = b.indexes(colIdx);
			final int sPos = b.pos(colIdx);
			final int sEnd = b.size(colIdx) + sPos;

			for(int j = 0; j < numVals; j++) { // rows left
				final int offOut = j * aggColSize;
				final double v = getValue(j, h, thisColsSize);
				sparseAddSelected(sPos, sEnd, aggColSize, aggregateColumns, sIndexes, sValues, ret, offOut, v);
			}

		}
		return Dictionary.create(ret);
	}

	private final void sparseAddSelected(int sPos, int sEnd, int aggColSize, IColIndex aggregateColumns, int[] sIndexes,
		double[] sValues, double[] ret, int offOut, double v) {

		int retIdx = 0;
		for(int i = sPos; i < sEnd; i++) {
			// skip through the retIdx.
			while(retIdx < aggColSize && aggregateColumns.get(retIdx) < sIndexes[i])
				retIdx++;
			if(retIdx == aggColSize)
				break;
			ret[offOut + retIdx] += v * sValues[i];
		}
		retIdx = 0;
	}

	protected IDictionary rightMMPreAggSparseAllColsRight(int numVals, SparseBlock b, IColIndex thisCols,
		int nColRight) {
		final int thisColsSize = thisCols.size();
		final double[] ret = new double[numVals * nColRight];

		for(int h = 0; h < thisColsSize; h++) { // common dim
			// chose row in right side matrix via column index of the dictionary
			final int colIdx = thisCols.get(h);
			if(b.isEmpty(colIdx))
				continue;

			// extract the row values on the right side.
			final double[] sValues = b.values(colIdx);
			final int[] sIndexes = b.indexes(colIdx);
			final int sPos = b.pos(colIdx);
			final int sEnd = b.size(colIdx) + sPos;

			for(int i = 0; i < numVals; i++) { // rows left
				final int offOut = i * nColRight;
				final double v = getValue(i, h, thisColsSize);
				SparseAdd(sPos, sEnd, ret, offOut, sIndexes, sValues, v);
			}
		}
		return Dictionary.create(ret);
	}

	private final void SparseAdd(int sPos, int sEnd, double[] ret, int offOut, int[] sIdx, double[] sVals, double v) {
		if(v != 0) {
			for(int k = sPos; k < sEnd; k++) { // cols right with value
				ret[offOut + sIdx[k]] += v * sVals[k];
			}
		}
	}

	@Override
	public void put(SparseBlock sb, int idx, int rowOut, int nCol, IColIndex columns) {
		for(int i = 0; i < nCol; i++)
			sb.append(rowOut, columns.get(i), getValue(idx, i, nCol));
	}

	@Override
	public double[] getRow(int i, int nCol) {
		double[] ret = new double[nCol];
		for(int c = 0; c < nCol; c++) {
			ret[c] = getValue(i, c, nCol);
		}
		return ret;
	}
}
