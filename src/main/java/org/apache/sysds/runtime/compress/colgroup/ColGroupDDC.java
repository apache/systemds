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
import org.apache.sysds.runtime.compress.utils.AbstractBitmap;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Class to encapsulate information about a column group that is encoded with dense dictionary encoding (DDC).
 * 
 * NOTE: zero values are included at position 0 in the value dictionary, which simplifies various operations such as
 * counting the number of non-zeros.
 */
public abstract class ColGroupDDC extends ColGroupValue {
	private static final long serialVersionUID = -3204391646123465004L;

	@Override
	public CompressionType getCompType() {
		return CompressionType.DDC;
	}

	public ColGroupDDC() {
		super();
	}

	protected ColGroupDDC(int[] colIndices, int numRows, AbstractBitmap ubm, CompressionSettings cs) {
		super(colIndices, numRows, ubm, cs);
	}

	protected ColGroupDDC(int[] colIndices, int numRows, double[] values) {
		super(colIndices, numRows, values);
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int rl, int ru) {
		double[] dictionary = getValues();
		for(int i = rl; i < ru; i++) {
			for(int colIx = 0; colIx < _colIndexes.length; colIx++) {
				int col = _colIndexes[colIx];
				double cellVal = getData(i, colIx, dictionary);
				target.quickSetValue(i, col, cellVal);
			}
		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		int nrow = getNumRows();
		int ncol = getNumCols();
		double[] dictionary = getValues();
		for(int i = 0; i < nrow; i++) {
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
		throw new NotImplementedException("Old Function Not In use");
		// int nrow = getNumRows();
		// for(int i = 0; i < nrow; i++) {
		// double cellVal = getData(i, colpos);
		// target.quickSetValue(i, 0, cellVal);
		// }
	}

	@Override
	public double get(int r, int c) {
		// find local column index
		int ix = Arrays.binarySearch(_colIndexes, c);
		if(ix < 0)
			throw new RuntimeException("Column index " + c + " not in DDC group.");

		// get value
		return _dict.getValue(getIndex(r, ix));
	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		int ncol = getNumCols();
		for(int i = rl; i < ru; i++) {
			int lnnz = 0;
			for(int colIx = 0; colIx < ncol; colIx++)
				lnnz += (_dict.getValue(getIndex(i, colIx)) != 0) ? 1 : 0;
			rnnz[i - rl] += lnnz;
		}
	}

	@Override
	protected void computeSum(MatrixBlock result, KahanFunction kplus) {
		final int ncol = getNumCols();
		final int numVals = getNumValues();

		// if(numVals < MAX_TMP_VALS) {
		// iterative over codes and count per code

		final int[] counts = getCounts();
		if(_dict instanceof QDictionary && !(kplus instanceof KahanPlusSq)) {
			final QDictionary values = ((QDictionary) _dict);
			long sum = 0;
			for(int k = 0, valOff = 0; k < numVals; k++, valOff += ncol) {
				int cntk = counts[k];
				for(int j = 0; j < ncol; j++)
					sum += values.getValueByte(valOff + j) * cntk;
			}
			result.quickSetValue(0, 0, result.quickGetValue(0, 0) + sum * values._scale);
			result.quickSetValue(0, 1, 0);
		}
		else {
			double[] values = getValues();
			// post-scaling of pre-aggregate with distinct values
			KahanObject kbuff = new KahanObject(result.quickGetValue(0, 0), result.quickGetValue(0, 1));
			for(int k = 0, valOff = 0; k < numVals; k++, valOff += ncol) {
				int cntk = counts[k];
				for(int j = 0; j < ncol; j++)
					kplus.execute3(kbuff, values[valOff + j], cntk);
			}
			result.quickSetValue(0, 0, kbuff._sum);
			result.quickSetValue(0, 1, kbuff._correction);
		}
	}

	protected void computeColSums(MatrixBlock result, KahanFunction kplus) {
		int nrow = getNumRows();
		int ncol = getNumCols();
		double[] values = _dict.getValues();

		KahanObject[] kbuff = new KahanObject[getNumCols()];
		for(int j = 0; j < ncol; j++)
			kbuff[j] = new KahanObject(result.quickGetValue(0, _colIndexes[j]),
				result.quickGetValue(1, _colIndexes[j]));

		for(int i = 0; i < nrow; i++) {
			int rowIndex = getIndex(i);
			for(int j = 0; j < ncol; j++)
				kplus.execute2(kbuff[j], values[rowIndex + j]);
		}

		for(int j = 0; j < ncol; j++) {
			result.quickSetValue(0, _colIndexes[j], kbuff[j]._sum);
			result.quickSetValue(1, _colIndexes[j], kbuff[j]._correction);
		}
	}

	// protected void computeRowSums(MatrixBlock result, KahanFunction kplus, int rl, int ru) {
	// int ncol = getNumCols();
	// KahanObject kbuff = new KahanObject(0, 0);
	// double[] values = getValues();
	// for(int i = rl; i < ru; i++) {
	// kbuff.set(result.quickGetValue(i, 0), result.quickGetValue(i, 1));
	// int rowIndex = getIndex(i);
	// for(int j = 0; j < ncol; j++)
	// kplus.execute2(kbuff, values[rowIndex + j]);
	// result.quickSetValue(i, 0, kbuff._sum);
	// result.quickSetValue(i, 1, kbuff._correction);
	// }
	// }

	@Override
	protected void computeRowSums(MatrixBlock result, KahanFunction kplus, int rl, int ru) {
		// note: due to corrections the output might be a large dense block
		DenseBlock c = result.getDenseBlock();

		if(_dict instanceof QDictionary && !(kplus instanceof KahanPlusSq)) {
			final QDictionary qDict = ((QDictionary) _dict);
			if(_colIndexes.length == 1) {
				byte[] vals = qDict._values;
				for(int i = rl; i < ru; i++) {
					double[] cvals = c.values(i);
					int cix = c.pos(i);
					cvals[cix] = cvals[cix] + vals[getIndex(i)] * qDict._scale;
				}
			}
			else {
				short[] vals = qDict.sumAllRowsToShort(_colIndexes.length);
				for(int i = rl; i < ru; i++) {
					double[] cvals = c.values(i);
					int cix = c.pos(i);
					cvals[cix] = cvals[cix] + vals[getIndex(i)] * qDict._scale;
				}
			}
		}
		else {
			KahanObject kbuff = new KahanObject(0, 0);
			KahanPlus kplus2 = KahanPlus.getKahanPlusFnObject();
			// pre-aggregate nnz per value tuple
			double[] vals = _dict.sumAllRowsToDouble(kplus, kbuff, _colIndexes.length, false);

			// scan data and add to result (use kahan plus not general KahanFunction
			// for correctness in case of sqk+)
			for(int i = rl; i < ru; i++) {
				double[] cvals = c.values(i);
				int cix = c.pos(i);
				kbuff.set(cvals[cix], cvals[cix + 1]);
				kplus2.execute2(kbuff, vals[getIndex(i)]);
				cvals[cix] = kbuff._sum;
				cvals[cix + 1] = kbuff._correction;
			}

		}
	}

	protected void computeRowMxx(MatrixBlock result, Builtin builtin, int rl, int ru) {
		double[] c = result.getDenseBlockValues();
		int ncol = getNumCols();
		double[] dictionary = getValues();

		for(int i = rl; i < ru; i++) {
			int rowIndex = getIndex(i);
			for(int j = 0; j < ncol; j++)
				c[i] = builtin.execute(c[i], dictionary[rowIndex + j]);
		}
	}

	protected final void postScaling(double[] vals, double[] c) {
		final int ncol = getNumCols();
		final int numVals = getNumValues();
		double[] values = getValues();

		for(int k = 0, valOff = 0; k < numVals; k++, valOff += ncol) {
			double aval = vals[k];
			for(int j = 0; j < ncol; j++) {
				int colIx = _colIndexes[j];
				c[colIx] += aval * values[valOff + j];
			}
		}
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
	 * @param r global row index
	 * @return value
	 */
	protected abstract double getData(int r, double[] dictionary);

	/**
	 * Generic get value for byte-length-agnostic access.
	 * 
	 * @param r          global row index
	 * @param colIx      local column index
	 * @param dictionary The values contained in the column groups dictionary
	 * @return value
	 */
	protected abstract double getData(int r, int colIx, double[] dictionary);

	/**
	 * Generic set value for byte-length-agnostic write of encoded value.
	 * 
	 * @param r    global row index
	 * @param code encoded value
	 */
	protected abstract void setData(int r, int code);

	protected abstract int getCode(int r);

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

		@Override
		public void next(double[] buff, int rowIx, int segIx, boolean last) {
			// copy entire value tuple to output row
			final int clen = getNumCols();
			final int off = getCode(rowIx) * clen;
			final double[] values = getValues();
			for(int j = 0; j < clen; j++)
				buff[_colIndexes[j]] = values[off + j];
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		return sb.toString();
	}

}
