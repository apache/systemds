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
import java.util.Iterator;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Class to encapsulate information about a column group that is encoded with dense dictionary encoding (DDC).
 * 
 */
public abstract class ColGroupDDC extends ColGroupValue {
	private static final long serialVersionUID = -3204391646123465004L;

	protected ColGroupDDC() {
		super();
	}

	protected ColGroupDDC(int[] colIndices, int numRows, ABitmap ubm, CompressionSettings cs) {
		super(colIndices, numRows, ubm, cs);
	}

	protected ColGroupDDC(int[] colIndices, int numRows, ADictionary dict) {
		super(colIndices, numRows, dict);
	}

	public CompressionType getCompType() {
		return CompressionType.DDC;
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int rl, int ru) {
		int ncol = getNumCols();
		double[] values = getValues();
		for(int i = rl; i < ru; i++)
			for(int j = 0; j < ncol; j++)
				target.appendValue(i, _colIndexes[j], getData(i, j, values));
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		int ncol = getNumCols();
		double[] dictionary = getValues();
		for(int i = 0; i < _numRows; i++) {
			for(int colIx = 0; colIx < ncol; colIx++) {
				int origMatrixColIx = getColIndex(colIx);
				int col = colIndexTargets[origMatrixColIx];
				double cellVal = getData(i, colIx, dictionary);
				target.quickSetValue(i, col, cellVal);
			}
		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int colpos) {
		int ncol = getNumCols();
		double[] c = target.getDenseBlockValues();
		double[] values = getValues();
		int nnz = 0;
		for(int i = 0; i < _numRows; i++) {
			int index = getIndex(i);
			if(index != values.length) {
				nnz += ((c[i] = values[(index) * ncol + colpos]) != 0) ? 1 : 0;
			}
			else {
				c[i] = 0.0;
			}
		}
		target.setNonZeros(nnz);
	}

	@Override
	public double get(int r, int c) {
		// find local column index
		int ix = Arrays.binarySearch(_colIndexes, c);
		if(ix < 0)
			throw new RuntimeException("Column index " + c + " not in DDC group.");

		// get value
		int index = getIndex(r, ix);
		if(index != getNumValues()) {

			return _dict.getValue(index);
		}
		else {
			return 0.0;
		}
	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		int ncol = getNumCols();
		final int numVals = getNumValues();
		for(int i = rl; i < ru; i++) {
			int lnnz = 0;
			for(int colIx = 0; colIx < ncol; colIx++) {
				int index = getIndex(i, colIx);
				if(index < numVals) {
					lnnz += (_dict.getValue(getIndex(i, colIx)) != 0) ? 1 : 0;
				}
			}
			rnnz[i - rl] += lnnz;
		}
	}

	@Override
	protected void computeSum(double[] c, KahanFunction kplus) {
		c[0] += _dict.sum(getCounts(), _colIndexes.length, kplus);
	}

	@Override
	protected void computeColSums(double[] c, KahanFunction kplus) {
		_dict.colSum(c, getCounts(), _colIndexes, kplus);
	}

	@Override
	protected void computeRowSums(double[] c, KahanFunction kplus, int rl, int ru, boolean mean) {
		final int numVals = getNumValues();
		KahanObject kbuff = new KahanObject(0, 0);
		KahanPlus kplus2 = KahanPlus.getKahanPlusFnObject();
		// pre-aggregate nnz per value tuple
		double[] vals = _dict.sumAllRowsToDouble(kplus, kbuff, _colIndexes.length);

		for(int rix = rl; rix < ru; rix++) {
			int index = getIndex(rix);
			if(index != numVals) {
				setandExecute(c, kbuff, kplus2, vals[index], rix * (2 + (mean ? 1 : 0)));
			}
		}
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru) {
		final int numVals = getNumValues();
		int ncol = getNumCols();
		double[] dictionary = getValues();

		for(int i = rl; i < ru; i++) {
			int rowIndex = getIndex(i);
			if(rowIndex != numVals) {
				for(int j = 0; j < ncol; j++)
					c[i] = builtin.execute(c[i], dictionary[rowIndex + j]);
			}
			else {
				c[i] = builtin.execute(c[i], 0.0);
			}
		}
	}

	public void postScaling(double[] values, double[] vals, double[] c, int numVals) {
		postScaling(values, vals, c, numVals, 0, 0);
	}

	public void postScaling(double[] values, double[] vals, double[] c, int numVals, int i, int totalCols) {
		final int ncol = getNumCols();

		for(int j = 0; j < ncol; j++) {
			int colIx = _colIndexes[j] + i * totalCols;
			for(int k = 0, valOff = 0; k < numVals; k++, valOff += ncol) {
				double aval = vals[k];
				if(valOff != numVals) {
					c[colIx] += aval * values[valOff + j];
				}
			}
		}
	}

	@Override
	public int[] getCounts(int[] counts) {
		return getCounts(0, _numRows, counts);
	}

	@Override
	public int[] getCounts(int rl, int ru, int[] counts) {
		for(int i = rl; i < ru; i++) {
			int index = getIndex(i);
			counts[index]++;
		}
		return counts;
	}


	@Override
	public void rightMultByMatrix(double[] b, double[] c, int numVals, double[] values, int rl, int ru, int vOff){
		throw new NotImplementedException("Not Implemented");
		// final int numCols = getNumCols();
	}
	

	@Override
	public void leftMultByMatrix(double[] a, double[] c, int numVals, double[] values, int numRows, int numCols, int rl,
		int ru, int voff) {
		numVals = (numVals == -1) ? getNumValues() : numVals;

		for(int i = rl, j = voff; i < ru; i++, j++) {
			if(8 * numVals < _numRows) {
				// iterative over codes and pre-aggregate inputs per code (guaranteed <=255)
				// temporary array also avoids false sharing in multi-threaded environments
				double[] vals = preAggregate(a, numVals, j);
				postScaling(values, vals, c, numVals, i, numCols);
			}
			else {
				for(int k = 0, aOff = j  *_numRows; k < _numRows; k++, aOff++) {
					double aval = a[aOff];
					if(aval != 0) {
						int valOff = getIndex(k) * _colIndexes.length;
						if(valOff != numVals) {
							for(int h = 0; h < _colIndexes.length; h++) {
								int colIx = _colIndexes[h] + i * numCols;
								c[colIx] += aval * values[valOff + h];
							}
						}
					}
				}
			}
		}
	}

	@Override
	public void leftMultByRowVector(double[] a, double[] result, int numVals) {
		numVals = (numVals == -1) ? getNumValues() : numVals;
		double[] values = getValues();

		leftMultByRowVector(a, result, numVals, values);

	}

	public double[] preAggregate(double[] a, int numVals) {
		return preAggregate(a, numVals, 0);
	}

	/**
	 * Pre aggregates a specific row from the input a which can be a row or a matrix.
	 * 
	 * @param a       the input vector or matrix to multiply with
	 * @param numVals the number of values contained in the dictionary
	 * @param aRows   the row index from a
	 * @return the pre-aggregated values.
	 */
	public double[] preAggregate(double[] a, int numVals, int aRows) {
		double[] vals;
		if(aRows > 0) {
			vals = allocDVector(numVals, true);
			// int off = _numRows * aRows;
			for(int i = 0, off = _numRows * aRows; i < _numRows; i++, off++) {
				int index = getIndex(i);
				if(index != numVals) { // Since we know that multiplying with 0 is .. 0 don't begin to aggregate.
					vals[index] += a[off];
				}
			}
		}
		else {
			vals = allocDVector(numVals, true);
			for(int i = 0; i < _numRows; i++) {
				int index = getIndex(i);
				if(index != numVals) { // Since we know that multiplying with 0 is .. 0 don't begin to aggregate.
					vals[index] += a[i];
				}
			}
		}
		return vals;
	}

	@Override
	public void leftMultByRowVector(double[] a, double[] c, int numVals, double[] values) {
		// double[] c = result.getDenseBlockValues();
		numVals = (numVals == -1) ? getNumValues() : numVals;

		if(8 * numVals < _numRows) {
			// iterative over codes and pre-aggregate inputs per code (guaranteed <=255)
			// temporary array also avoids false sharing in multi-threaded environments
			double[] vals = preAggregate(a, numVals);
			postScaling(values, vals, c, numVals);
		}
		else {
			// iterate over codes, compute all, and add to the result
			for(int i = 0; i < _numRows; i++) {
				double aval = a[i];
				if(aval != 0)
					for(int j = 0, valOff = getIndex(i) * _colIndexes.length; j < _colIndexes.length; j++)
						if(valOff != numVals) {
							c[_colIndexes[j]] += aval * values[valOff + j];
						}
			}
		}
	}

	@Override
	public Iterator<IJV> getIterator(int rl, int ru, boolean inclZeros, boolean rowMajor) {
		// DDC iterator is always row major, so no need for custom handling
		return new DDCIterator(rl, ru, inclZeros);
	}

	@Override
	public ColGroupRowIterator getRowIterator(int rl, int ru) {
		return new DDCRowIterator(rl, ru);
	}

	private class DDCIterator implements Iterator<IJV> {
		// iterator configuration
		private final int _ru;
		private final boolean _inclZeros;

		// iterator state
		private final IJV _buff = new IJV();
		private int _rpos = -1;
		private int _cpos = -1;
		private double _value = 0;

		public DDCIterator(int rl, int ru, boolean inclZeros) {
			_ru = ru;
			_inclZeros = inclZeros;
			_rpos = rl;
			_cpos = -1;
			getNextValue();
		}

		@Override
		public boolean hasNext() {
			return(_rpos < _ru);
		}

		@Override
		public IJV next() {
			_buff.set(_rpos, _colIndexes[_cpos], _value);
			getNextValue();
			return _buff;
		}

		private void getNextValue() {
			do {
				boolean nextRow = (_cpos + 1 >= getNumCols());
				_rpos += nextRow ? 1 : 0;
				_cpos = nextRow ? 0 : _cpos + 1;
				if(_rpos >= _ru)
					return; // reached end
				_value = _dict.getValue(getIndex(_rpos, _cpos));
			}
			while(!_inclZeros && _value == 0);
		}
	}

	private class DDCRowIterator extends ColGroupRowIterator {
		public DDCRowIterator(int rl, int ru) {
			// do nothing
		}

		public void next(double[] buff, int rowIx, int segIx, boolean last) {
			// copy entire value tuple to output row
			final int clen = getNumCols();
			final int off = getIndex(rowIx) * clen;
			final double[] values = getValues();
			for(int j = 0; j < clen; j++)
				buff[_colIndexes[j]] = values[off + j];
		}
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		return sb.toString();
	}

	/**
	 * Generic get index in dictionary for value at row position.
	 * 
	 * @param r row position to get dictionary index for.
	 * @return The dictionary index
	 */
	protected abstract int getIndex(int r);

	/**
	 * Generic get index in dictionary for value at row, col position. If used consider changing to getIndex and
	 * precalculate offset to row
	 * 
	 * @param r     The row to find
	 * @param colIx the col index to find
	 * @return the index in the dictionary containing the specified value
	 */
	protected abstract int getIndex(int r, int colIx);

	/**
	 * Generic get value for byte-length-agnostic access to first column.
	 * 
	 * @param r      Global row index
	 * @param values The values contained in the column groups dictionary
	 * @return value
	 */
	protected abstract double getData(int r, double[] values);

	/**
	 * Generic get value for byte-length-agnostic access.
	 * 
	 * @param r      Global row index
	 * @param colIx  Local column index
	 * @param values The values contained in the column groups dictionary
	 * @return value
	 */
	protected abstract double getData(int r, int colIx, double[] values);

	/**
	 * Generic set value for byte-length-agnostic write of encoded value.
	 * 
	 * @param r    global row index
	 * @param code encoded value
	 */
	protected abstract void setData(int r, int code);

}
