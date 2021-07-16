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

import java.io.DataInput;
import java.io.IOException;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.Bitmap;
import org.apache.sysds.runtime.compress.utils.BitmapLossy;
import org.apache.sysds.runtime.compress.utils.MultiColBitmap;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class DictionaryFactory {

	protected static final Log LOG = LogFactory.getLog(DictionaryFactory.class.getName());

	public enum Type {
		FP64_DICT, MATRIX_BLOCK_DICT, INT8_DICT
	}

	public static ADictionary read(DataInput in) throws IOException {
		Type type = Type.values()[in.readByte()];
		switch(type) {
			case FP64_DICT:
				return Dictionary.read(in);
			case MATRIX_BLOCK_DICT:
				return MatrixBlockDictionary.read(in);
			case INT8_DICT:
				return QDictionary.read(in);
			default:
				throw new DMLCompressionException("Unsupported type of dictionary : " + type);
		}

	}

	public static long getInMemorySize(int nrValues, int nrColumns, double tupleSparsity, boolean lossy) {
		if(lossy)
			return QDictionary.getInMemorySize(nrValues * nrColumns);
		else if(nrColumns > 1 && tupleSparsity < 0.4)
			return MatrixBlockDictionary.getInMemorySize(nrValues, nrColumns, tupleSparsity);
		else
			return Dictionary.getInMemorySize(nrValues * nrColumns);
	}

	public static ADictionary create(ABitmap ubm) {
		return create(ubm, 1.0);
	}

	public static ADictionary create(ABitmap ubm, double sparsity, boolean withZeroTuple) {
		return (withZeroTuple) ? createWithAppendedZeroTuple(ubm, sparsity) : create(ubm, sparsity);
	}

	public static ADictionary create(ABitmap ubm, double sparsity) {
		if(ubm instanceof BitmapLossy)
			return new QDictionary((BitmapLossy) ubm);
		else if(ubm instanceof Bitmap)
			return new Dictionary(((Bitmap) ubm).getValues());
		else if(sparsity < 0.4 && ubm instanceof MultiColBitmap) {
			final int nCols = ubm.getNumColumns();
			final int nRows = ubm.getNumValues();
			final MultiColBitmap mcbm = (MultiColBitmap) ubm;

			final MatrixBlock m = new MatrixBlock(nRows, nCols, true);
			m.allocateSparseRowsBlock();
			final SparseBlock sb = m.getSparseBlock();

			final int nVals = ubm.getNumValues();
			for(int i = 0; i < nVals; i++) {
				final double[] tuple = mcbm.getValues(i);
				for(int col = 0; col < nCols; col++)
					sb.append(i, col, tuple[col]);
			}
			m.recomputeNonZeros();
			return new MatrixBlockDictionary(m);
		}
		else if(ubm instanceof MultiColBitmap) {
			MultiColBitmap mcbm = (MultiColBitmap) ubm;
			final int nCol = ubm.getNumColumns();
			final int nVals = ubm.getNumValues();
			double[] resValues = new double[nVals * nCol];
			for(int i = 0; i < nVals; i++)
				System.arraycopy(mcbm.getValues(i), 0, resValues, i * nCol, nCol);

			return new Dictionary(resValues);
		}
		throw new NotImplementedException(
			"Not implemented creation of bitmap type : " + ubm.getClass().getSimpleName());
	}

	public static ADictionary createWithAppendedZeroTuple(ABitmap ubm) {
		return createWithAppendedZeroTuple(ubm, 1.0);
	}

	public static ADictionary createWithAppendedZeroTuple(ABitmap ubm, double sparsity) {
		final int nRows = ubm.getNumValues() + 1;
		final int nCols = ubm.getNumColumns();
		if(ubm instanceof Bitmap) {
			Bitmap bm = (Bitmap) ubm;
			double[] resValues = new double[ubm.getNumValues() + 1];
			double[] from = bm.getValues();
			System.arraycopy(from, 0, resValues, 0, from.length);
			return new Dictionary(resValues);
		}
		else if(sparsity < 0.4 && ubm instanceof MultiColBitmap) {
			final MultiColBitmap mcbm = (MultiColBitmap) ubm;
			final MatrixBlock m = new MatrixBlock(nRows, nCols, true);
			m.allocateSparseRowsBlock();
			final SparseBlock sb = m.getSparseBlock();

			final int nVals = ubm.getNumValues();
			for(int i = 0; i < nVals; i++) {
				final double[] tuple = mcbm.getValues(i);
				for(int col = 0; col < nCols; col++)
					sb.append(i, col, tuple[col]);
			}
			m.recomputeNonZeros();
			return new MatrixBlockDictionary(m);
		}
		else if(ubm instanceof MultiColBitmap) {
			MultiColBitmap mcbm = (MultiColBitmap) ubm;
			final int nVals = ubm.getNumValues();
			double[] resValues = new double[nRows * nCols];
			for(int i = 0; i < nVals; i++)
				System.arraycopy(mcbm.getValues(i), 0, resValues, i * nCols, nCols);

			return new Dictionary(resValues);
		}
		else {
			throw new NotImplementedException(
				"Not implemented creation of bitmap type : " + ubm.getClass().getSimpleName());
		}
	}

	public static ADictionary moveFrequentToLastDictionaryEntry(ADictionary dict, ABitmap ubm, int nRow,
		int largestIndex) {
		final int zeros = nRow - (int) ubm.getNumOffsets();
		final int nCol = ubm.getNumColumns();
		final int largestIndexSize = ubm.getOffsetsList(largestIndex).size();
		if(dict instanceof MatrixBlockDictionary) {
			MatrixBlockDictionary mbd = (MatrixBlockDictionary) dict;
			MatrixBlock mb = mbd.getMatrixBlock();
			if(mb.isEmpty()) {
				if(zeros == 0)
					return dict;
				else
					return new MatrixBlockDictionary(new MatrixBlock(mb.getNumRows() + 1, mb.getNumColumns(), true));
			}
			else if(mb.isInSparseFormat()) {
				MatrixBlockDictionary mbdn = moveToLastDictionaryEntrySparse(mb.getSparseBlock(), largestIndex, zeros,
					nCol, largestIndexSize);
				MatrixBlock mbn = mbdn.getMatrixBlock();
				mbn.setNonZeros(mb.getNonZeros());
				if(mbn.getNonZeros() == 0)
					mbn.recomputeNonZeros();
				return mbdn;
			}
			else
				return moveToLastDictionaryEntryDense(mb.getDenseBlockValues(), largestIndex, zeros, nCol,
					largestIndexSize);
		}
		else
			return moveToLastDictionaryEntryDense(dict.getValues(), largestIndex, zeros, nCol, largestIndexSize);

	}

	private static MatrixBlockDictionary moveToLastDictionaryEntrySparse(SparseBlock sb, int indexToMove, int zeros,
		int nCol, int largestIndexSize) {

		if(zeros == 0) {
			MatrixBlock ret = new MatrixBlock(sb.numRows(), nCol, true);
			ret.setSparseBlock(sb);
			final SparseRow swap = sb.get(indexToMove);
			for(int i = indexToMove + 1; i < sb.numRows(); i++)
				sb.set(i - 1, sb.get(i), false);
			sb.set(sb.numRows() - 1, swap, false);
			return new MatrixBlockDictionary(ret);
		}

		MatrixBlock ret = new MatrixBlock(sb.numRows() + 1, nCol, true);
		ret.allocateSparseRowsBlock();
		final SparseBlock retB = ret.getSparseBlock();
		if(zeros > largestIndexSize) {
			for(int i = 0; i < sb.numRows(); i++)
				retB.set(i, sb.get(i), false);
		}
		else {
			for(int i = 0; i < indexToMove; i++)
				retB.set(i, sb.get(i), false);

			retB.set(sb.numRows(), sb.get(indexToMove), false);
			for(int i = indexToMove + 1; i < sb.numRows(); i++)
				retB.set(i - 1, sb.get(i), false);
		}
		return new MatrixBlockDictionary(ret);
	}

	private static ADictionary moveToLastDictionaryEntryDense(double[] values, int indexToMove, int zeros, int nCol,
		int largestIndexSize) {
		final int offsetToLargest = indexToMove * nCol;

		if(zeros == 0) {
			final double[] swap = new double[nCol];
			System.arraycopy(values, offsetToLargest, swap, 0, nCol);
			for(int i = offsetToLargest; i < values.length - nCol; i++)
				values[i] = values[i + nCol];

			System.arraycopy(swap, 0, values, values.length - nCol, nCol);
			return new Dictionary(values);
		}

		final double[] newDict = new double[values.length + nCol];

		if(zeros > largestIndexSize)
			System.arraycopy(values, 0, newDict, 0, values.length);
		else {
			System.arraycopy(values, 0, newDict, 0, offsetToLargest);
			System.arraycopy(values, offsetToLargest + nCol, newDict, offsetToLargest,
				values.length - offsetToLargest - nCol);
			System.arraycopy(values, offsetToLargest, newDict, newDict.length - nCol, nCol);
		}
		return new Dictionary(newDict);
	}
}
