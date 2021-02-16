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

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.pre.IPreAggregate;
import org.apache.sysds.runtime.compress.colgroup.pre.PreAggregateFactory;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
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

	protected ColGroupDDC(int[] colIndices, int numRows, ADictionary dict, int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
	}

	public CompressionType getCompType() {
		return CompressionType.DDC;
	}

	@Override
	public void decompressToBlockSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		decompressToBlockUnSafe(target, rl, ru, offT, values);
		target.setNonZeros(target.getNonZeros() + _numRows * _colIndexes.length);
	}

	@Override
	public void decompressToBlockUnSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		final int nCol = _colIndexes.length;
		final int tCol = target.getNumColumns();
		double[] c = target.getDenseBlockValues();
		offT = offT * tCol;

		if(_zeros)
			for(int i = rl; i < ru; i++, offT += tCol) {
				int rowIndex = getIndex(i) * nCol;
				if(rowIndex < values.length)
					for(int j = 0; j < nCol; j++)
						c[offT + _colIndexes[j]] += values[rowIndex + j];

			}
		else
			for(int i = rl; i < ru; i++, offT += tCol) {
				int rowIndex = getIndex(i) * nCol;
				for(int j = 0; j < nCol; j++)
					c[offT + _colIndexes[j]] += values[rowIndex + j];
			}

	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		int ncol = getNumCols();
		double[] dictionary = getValues();
		for(int i = 0; i < _numRows; i++) {
			int rowIndex = getIndex(i) * ncol;
			if(rowIndex < dictionary.length) {
				for(int colIx = 0; colIx < ncol; colIx++) {
					int origMatrixColIx = getColIndex(colIx);
					int col = colIndexTargets[origMatrixColIx];
					double cellVal = dictionary[rowIndex + colIx];
					target.quickSetValue(i, col, target.quickGetValue(i, col) + cellVal);
				}
			}
		}
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colpos) {
		int ncol = getNumCols();
		double[] c = target.getDenseBlockValues();
		double[] values = getValues();
		int nnz = 0;
		for(int i = 0; i < _numRows; i++) {
			int index = getIndex(i);
			if(index < getNumValues())
				nnz += ((c[i] += values[(index) * ncol + colpos]) != 0) ? 1 : 0;
			else
				nnz++;

		}
		target.setNonZeros(nnz);
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colpos, int rl, int ru) {
		int ncol = getNumCols();
		double[] c = target.getDenseBlockValues();
		double[] values = getValues();
		final int numValues = getNumValues();
		int nnz = 0;
		for(int i = 0, r = rl; i < ru - rl; i++, r++) {
			int index = getIndex(r);
			if(index < numValues)
				nnz += ((c[i] += values[(index) * ncol + colpos]) != 0) ? 1 : 0;
			else
				nnz++;
		}
		target.setNonZeros(nnz);
	}

	@Override
	public void decompressColumnToBlock(double[] c, int colpos, int rl, int ru) {
		int ncol = getNumCols();
		double[] values = getValues();
		final int numValues = getNumValues();
		for(int i = 0, r = rl; i < ru - rl; i++, r++) {
			int index = getIndex(r);
			if(index < numValues)
				c[i] += values[(index) * ncol + colpos];
		}
	}

	@Override
	public double get(int r, int c) {
		// find local column index
		int ix = Arrays.binarySearch(_colIndexes, c);
		if(ix < 0)
			throw new RuntimeException("Column index " + c + " not in DDC group.");

		// get value
		int index = getIndex(r);
		if(index < getNumValues())
			return _dict.getValue(index * _colIndexes.length + ix);
		else
			return 0.0;

	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		int ncol = _colIndexes.length;
		final int numVals = getNumValues();
		double[] values = _dict.getValues();
		for(int i = rl; i < ru; i++) {
			int lnnz = 0;
			int index = getIndex(i);
			if(index < numVals) {
				for(int colIx = index; colIx < ncol + index; colIx++) {
					lnnz += (values[colIx]) != 0 ? 1 : 0;
				}
			}
			rnnz[i - rl] += lnnz;
		}
	}

	@Override
	protected void computeRowSums(double[] c, boolean square, int rl, int ru, boolean mean) {
		final int numVals = getNumValues();
		// pre-aggregate nnz per value tuple
		double[] vals = _dict.sumAllRowsToDouble(square, _colIndexes.length);

		final int mult = (2 + (mean ? 1 : 0));
		for(int rix = rl; rix < ru; rix++) {
			int index = getIndex(rix);
			if(index < numVals) {
				setandExecute(c, false, vals[index], rix * mult);
			}
		}
	}

	@Override
	protected void computeRowMxx(MatrixBlock c, Builtin builtin, int rl, int ru) {
		int ncol = getNumCols();
		double[] dictionary = getValues();

		for(int i = rl; i < ru; i++) {
			int index = getIndex(i) * ncol;
			for(int j = 0; j < ncol; j++) {
				if(index < dictionary.length) {
					c.quickSetValue(i, 0, builtin.execute(c.quickGetValue(i, 0), dictionary[index + j]));
				}
				else {
					c.quickSetValue(i, 0, builtin.execute(c.quickGetValue(i, 0), 0.0));
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
	public void leftMultByMatrix(double[] a, double[] c, double[] values, int numRows, int numCols, int rl, int ru,
		int voff) {

		int numVals = getNumValues();
		// if(8 * numVals < _numRows) {
		for(int i = rl, j = voff; i < ru; i++, j++) {
			double[] vals = preAggregate(a, j);
			postScaling(values, vals, c, numVals, i, numCols);
		}
		// }
		// else {
		// for(int i = rl, j = voff; i < ru; i++, j++) {
		// int offC = i * numCols;
		// numVals = numVals * _colIndexes.length;
		// for(int k = 0, aOff = j * _numRows; k < _numRows; k++, aOff++) {
		// double aval = a[aOff];
		// if(aval != 0) {
		// int valOff = getIndex(k) * _colIndexes.length;
		// if(valOff < numVals) {
		// for(int h = 0; h < _colIndexes.length; h++) {
		// int colIx = _colIndexes[h] + offC;
		// c[colIx] += aval * values[valOff + h];
		// }
		// }
		// }
		// }

		// }
		// }
	}

	@Override
	public void leftMultBySparseMatrix(SparseBlock sb, double[] c, double[] values, int numRows, int numCols, int row,
		double[] MaterializedRow) {
		final int numVals = getNumValues();
		double[] vals = preAggregateSparse(sb, row);
		postScaling(values, vals, c, numVals, row, numCols);
	}

	@Override
	public double[] preAggregate(double[] a, int row) {
		double[] vals = allocDVector(getNumValues() + 1, true);
		if(row > 0)
			for(int i = 0, off = _numRows * row; i < _numRows; i++, off++)
				vals[getIndex(i)] += a[off];
		else
			for(int i = 0; i < _numRows; i++)
				vals[getIndex(i)] += a[i];

		return vals;
	}

	@Override
	public double[] preAggregateSparse(SparseBlock sb, int row) {
		double[] vals = allocDVector(getNumValues() + 1, true);
		int[] indexes = sb.indexes(row);
		double[] sparseV = sb.values(row);
		for(int i = sb.pos(row); i < sb.size(row) + sb.pos(row); i++)
			vals[getIndex(indexes[i])] += sparseV[i];

		return vals;
	}

	// @Override
	// public void leftMultByRowVector(double[] a, double[] c, int numVals, double[] values) {
	// // if(8 * numVals < _numRows) {
	// double[] vals = preAggregate(a);
	// postScaling(values, vals, c, numVals);
	// // }
	// // else {
	// // numVals = numVals * _colIndexes.length;
	// // // iterate over codes, compute all, and add to the result
	// // for(int i = 0; i < _numRows; i++) {
	// // double aval = a[i];
	// // if(aval != 0)
	// // for(int j = 0, valOff = getIndex(i) * _colIndexes.length; j < _colIndexes.length; j++)
	// // if(valOff < numVals) {
	// // c[_colIndexes[j]] += aval * values[valOff + j];
	// // }
	// // }
	// // }
	// }

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

	@Override
	public IPreAggregate preAggregateDDC(ColGroupDDC lhs) {
		final int nCol = lhs.getNumValues();
		final int rhsNV = this.getNumValues();
		final int retSize = nCol * rhsNV;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		if(!this._zeros && !lhs._zeros)
			DDCNZDDCNZ(lhs, this, ag, nCol);
		else if(!this._zeros)
			for(int i = 0; i < this._numRows; i++) {
				int col = lhs.getIndex(i);
				if(col < nCol)
					ag.increment(col + this.getIndex(i) * nCol);
			}
		else if(!lhs._zeros)
			for(int i = 0; i < this._numRows; i++) {
				int row = this.getIndex(i);
				if(row < rhsNV)
					ag.increment(lhs.getIndex(i) + row * nCol);
			}
		else
			DDCZDDCZ(lhs, this, nCol, rhsNV, ag, nCol);

		return ag;
	}

	private static void DDCNZDDCNZ(ColGroupDDC cgl, ColGroupDDC cgr, IPreAggregate ag, final int nCol) {
		for(int i = 0; i < cgr._numRows; i++)
			ag.increment(cgl.getIndex(i) + cgr.getIndex(i) * nCol);
	}

	private static void DDCZDDCZ(ColGroupDDC cgl, ColGroupDDC cgr, final int lhsNV, final int rhsNV, IPreAggregate ag,
		final int nCol) {
		for(int i = 0; i < cgr._numRows; i++) {
			int row = cgr.getIndex(i);
			if(row < rhsNV) {
				int col = cgl.getIndex(i);
				if(col < lhsNV)
					ag.increment(col + row * nCol);
			}
		}
	}

	@Override
	public IPreAggregate preAggregateSDC(ColGroupSDC lhs) {
		final int nCol = lhs.getNumValues();
		final int rhsNV = this.getNumValues();
		final int retSize = nCol * rhsNV;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		final int[] sdcIndexes = lhs._indexes;
		final char[] sdcData = lhs._data;
		final int offsetToDefault = nCol - 1;

		int off = 0;
		int i = 0;

		int col;
		for(; i < this._numRows && off < sdcIndexes.length; i++) {
			int row = this.getIndex(i);
			if(sdcIndexes[off] == i)
				col = sdcData[off++];
			else
				col = offsetToDefault;
			if(row < this.getNumValues())
				ag.increment(col + row * nCol);
		}
		col = offsetToDefault;
		for(; i < this._numRows; i++) {
			int row = this.getIndex(i);
			if(row < this.getNumValues())
				ag.increment(col + row * nCol);
		}

		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDCSingle(ColGroupSDCSingle lhs) {
		final int nCol = lhs.getNumValues();
		final int rhsNV = this.getNumValues();
		final int retSize = nCol * rhsNV;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		final int[] sdcIndexes = lhs._indexes;
		final int offsetToDefault = lhs.getNumValues() - 1;

		int off = 0;
		int i = 0;

		int col;
		for(; i < this._numRows && off < sdcIndexes.length; i++) {
			int row = this.getIndex(i);
			if(sdcIndexes[off] == i) {
				col = offsetToDefault;
				off++;
			}
			else
				col = 0;
			if(row < this.getNumValues())
				ag.increment(col + row * nCol);
		}
		col = 0;
		for(; i < this._numRows; i++) {
			int row = this.getIndex(i);

			if(row < this.getNumValues())
				ag.increment(col + row * nCol);
		}

		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDCZeros(ColGroupSDCZeros lhs) {
		final int nCol = lhs.getNumValues();
		final int rhsNV = this.getNumValues();
		final int retSize = nCol * rhsNV;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		final int[] sdcIndexes = lhs._indexes;
		final char[] sdcData = lhs._data;

		for(int i = 0; i < sdcIndexes.length; i++) {
			int row = this.getIndex(sdcIndexes[i]);
			int col = sdcData[i];
			if(row < this.getNumValues())
				ag.increment(col + row * nCol);
		}

		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDCSingleZeros(ColGroupSDCSingleZeros lhs) {
		final int nCol = lhs.getNumValues();
		final int rhsNV = this.getNumValues();
		final int retSize = nCol * rhsNV;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		final int[] sdcIndexes = lhs._indexes;

		for(int i = 0; i < sdcIndexes.length; i++) {
			int row = this.getIndex(sdcIndexes[i]);
			if(row < this.getNumValues())
				ag.increment(row * nCol);
		}

		return ag;
	}

	@Override
	public IPreAggregate preAggregateOLE(ColGroupOLE lhs) {
		final int NVR = this.getNumValues();
		final int NVL = lhs.getNumValues();
		final int retSize = NVR * NVL;
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		for(int kl = 0; kl < NVL; kl++) {
			final int bOffL = lhs._ptr[kl];
			final int bLenL = lhs.len(kl);
			for(int bixL = 0, offL = 0, sLenL = 0; bixL < bLenL; bixL += sLenL + 1, offL += blksz) {
				sLenL = lhs._data[bOffL + bixL];
				for(int i = 1; i <= sLenL; i++) {
					int idx = this.getIndex(offL + lhs._data[bOffL + bixL + i]);
					ag.increment(kl + idx * NVL);
				}
			}
		}

		return ag;
	}

	@Override
	public IPreAggregate preAggregateRLE(ColGroupRLE lhs) {
		final int NVR = this.getNumValues();
		final int NVL = lhs.getNumValues();
		final int retSize = NVR * NVL;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		for(int kl = 0; kl < NVL; kl++) {
			final int boffL = lhs._ptr[kl];
			final int blenL = lhs.len(kl);
			for(int bixL = 0, startL = 0, lenL = 0; bixL < blenL && startL < _numRows; startL += lenL, bixL += 2) {
				startL += lhs._data[boffL + bixL];
				lenL = lhs._data[boffL + bixL + 1];
				final int endL = startL + lenL;
				for(int i = startL; i < endL; i++){
					int kr = getIndex(i) * NVL;
					ag.increment(kl + kr);
				}
			}
		}
		return ag;
	}
}
