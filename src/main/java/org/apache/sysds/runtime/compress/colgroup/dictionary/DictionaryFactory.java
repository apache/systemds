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
import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.Bitmap;
import org.apache.sysds.runtime.compress.utils.BitmapLossy;
import org.apache.sysds.runtime.compress.utils.MultiColBitmap;
import org.apache.sysds.runtime.data.SparseBlock;
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
		// Log.warn("Inefficient creation of dictionary, to then allocate again.");
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

	public static ADictionary moveFrequentToLastDictionaryEntry(ADictionary dict, ABitmap ubm, int numRows,
		int largestIndex) {
		LOG.warn("Inefficient creation of dictionary, to then allocate again to move one entry to end.");
		final double[] dictValues = dict.getValues();
		final int zeros = numRows - (int) ubm.getNumOffsets();
		final int nCol = ubm.getNumColumns();
		final int offsetToLargest = largestIndex * nCol;

		if(zeros == 0) {
			final double[] swap = new double[nCol];
			System.arraycopy(dictValues, offsetToLargest, swap, 0, nCol);
			for(int i = offsetToLargest; i < dictValues.length - nCol; i++) {
				dictValues[i] = dictValues[i + nCol];
			}
			System.arraycopy(swap, 0, dictValues, dictValues.length - nCol, nCol);
			return dict;
		}

		final int largestIndexSize = ubm.getOffsetsList(largestIndex).size();
		final double[] newDict = new double[dictValues.length + nCol];

		if(zeros > largestIndexSize)
			System.arraycopy(dictValues, 0, newDict, 0, dictValues.length);
		else {
			System.arraycopy(dictValues, 0, newDict, 0, offsetToLargest);
			System.arraycopy(dictValues, offsetToLargest + nCol, newDict, offsetToLargest,
				dictValues.length - offsetToLargest - nCol);
			System.arraycopy(dictValues, offsetToLargest, newDict, newDict.length - nCol, nCol);
		}
		return new Dictionary(newDict);
	}
}
