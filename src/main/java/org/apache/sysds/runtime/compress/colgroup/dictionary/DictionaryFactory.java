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
import java.util.ArrayList;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.bitmap.Bitmap;
import org.apache.sysds.runtime.compress.bitmap.MultiColBitmap;
import org.apache.sysds.runtime.compress.utils.DArrCounts;
import org.apache.sysds.runtime.compress.utils.DblArrayCountHashMap;
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

	public static ADictionary create(DblArrayCountHashMap map, int nCols, boolean addZeroTuple) {
		try {

			final ArrayList<DArrCounts> vals = map.extractValues();
			final int nVals = vals.size();
			int nnz = 0;
			for(int i = 0; i < nVals; i++) {
				final DArrCounts dac = vals.get(i);
				double[] dv = dac.key.getData();

				for(double v : dv)
					nnz += v != 0 ? 1 : 0;
			}
			final int nTuplesOut = nVals + (addZeroTuple ? 1 : 0);
			boolean sparse = MatrixBlock.evalSparseFormatInMemory(nTuplesOut, nCols, nnz);
			if(sparse) {
				final MatrixBlock retB = new MatrixBlock(nTuplesOut, nCols, true);
				retB.allocateSparseRowsBlock();
				final SparseBlock sb = retB.getSparseBlock();
				for(int i = 0; i < nVals; i++) {
					final DArrCounts dac = vals.get(i);
					double[] dv = dac.key.getData();
					for(int k = 0; k < dv.length; k++)
						sb.append(i, k, dv[k]);
				}
				retB.recomputeNonZeros();
				return new MatrixBlockDictionary(retB, nCols);
			}
			else {

				final double[] resValues = new double[(nTuplesOut) * nCols];
				for(int i = 0; i < nVals; i++) {
					final DArrCounts dac = vals.get(i);
					System.arraycopy(dac.key.getData(), 0, resValues, dac.id * nCols, nCols);
				}
				return new Dictionary(resValues);
			}
		}
		catch(Exception e) {
			LOG.error("Failed to create dictionary: ", e);
			return null;
		}

	}

	public static ADictionary createDelta(DblArrayCountHashMap map, int nCols, boolean addZeroTuple) {
		final ArrayList<DArrCounts> vals = map.extractValues();
		final int nVals = vals.size();
		final double[] resValues = new double[(nVals + (addZeroTuple ? 1 : 0)) * nCols];
		for(int i = 0; i < nVals; i++) {
			final DArrCounts dac = vals.get(i);
			System.arraycopy(dac.key.getData(), 0, resValues, dac.id * nCols, nCols);
		}
		return new DeltaDictionary(resValues, nCols);
	}

	public static ADictionary create(ABitmap ubm) {
		return create(ubm, 1.0);
	}

	public static ADictionary create(ABitmap ubm, double sparsity, boolean withZeroTuple) {
		return (withZeroTuple) ? createWithAppendedZeroTuple(ubm, sparsity) : create(ubm, sparsity);
	}

	public static ADictionary create(ABitmap ubm, double sparsity) {
		final int nCol = ubm.getNumColumns();
		if(ubm instanceof Bitmap)
			return new Dictionary(((Bitmap) ubm).getValues());
		else if(sparsity < 0.4 && nCol > 4 && ubm instanceof MultiColBitmap) {
			final MultiColBitmap mcbm = (MultiColBitmap) ubm;

			final MatrixBlock m = new MatrixBlock(ubm.getNumValues(), nCol, true);
			m.allocateSparseRowsBlock();
			final SparseBlock sb = m.getSparseBlock();

			final int nVals = ubm.getNumValues();
			for(int i = 0; i < nVals; i++) {
				final double[] tuple = mcbm.getValues(i);
				for(int col = 0; col < nCol; col++)
					sb.append(i, col, tuple[col]);
			}
			m.recomputeNonZeros();
			return new MatrixBlockDictionary(m, nCol);
		}
		else if(ubm instanceof MultiColBitmap) {
			MultiColBitmap mcbm = (MultiColBitmap) ubm;
			final int nVals = ubm.getNumValues();
			double[] resValues = new double[nVals * nCol];
			for(int i = 0; i < nVals; i++)
				System.arraycopy(mcbm.getValues(i), 0, resValues, i * nCol, nCol);

			return new Dictionary(resValues);
		}
		throw new NotImplementedException("Not implemented creation of bitmap type : " + ubm.getClass().getSimpleName());
	}

	public static ADictionary create(ABitmap ubm, int defaultIndex, double[] defaultTuple, double sparsity,
		boolean addZero) {
		final int nCol = ubm.getNumColumns();
		final int nVal = ubm.getNumValues() - (addZero ? 0 : 1);
		if(nCol > 4 && sparsity < 0.4) {
			// return sparse
			throw new NotImplementedException("Not supported sparse allocation yet");
		}
		else {
			double[] dict = new double[nCol * nVal];
			if(ubm instanceof Bitmap) {
				final double[] bmv = ((Bitmap) ubm).getValues();
				System.arraycopy(bmv, 0, dict, 0, defaultIndex);
				defaultTuple[0] = bmv[defaultIndex];
				System.arraycopy(bmv, defaultIndex + 1, dict, defaultIndex, bmv.length - defaultIndex - 1);
			}
			else if(ubm instanceof MultiColBitmap) {
				MultiColBitmap mcbm = (MultiColBitmap) ubm;
				for(int i = 0; i < defaultIndex; i++)
					System.arraycopy(mcbm.getValues(i), 0, dict, i * nCol, nCol);
				System.arraycopy(mcbm.getValues(defaultIndex), 0, defaultTuple, 0, nCol);
				for(int i = defaultIndex; i < ubm.getNumValues() - 1; i++)
					System.arraycopy(mcbm.getValues(i + 1), 0, dict, i * nCol, nCol);
			}
			else
				throw new NotImplementedException("not supported ABitmap of type:" + ubm.getClass().getSimpleName());

			return new Dictionary(dict);
		}
	}

	public static ADictionary createWithAppendedZeroTuple(ABitmap ubm, double sparsity) {
		final int nVals = ubm.getNumValues();
		final int nRows = nVals + 1;
		final int nCols = ubm.getNumColumns();

		if(ubm instanceof Bitmap) {
			final double[] resValues = new double[nRows];
			final double[] from = ((Bitmap) ubm).getValues();
			System.arraycopy(from, 0, resValues, 0, from.length);
			return new Dictionary(resValues);
		}

		final MultiColBitmap mcbm = (MultiColBitmap) ubm;
		if(sparsity < 0.4 && nCols > 4) {
			final MatrixBlock m = new MatrixBlock(nRows, nCols, true);
			m.allocateSparseRowsBlock();
			final SparseBlock sb = m.getSparseBlock();

			for(int i = 0; i < nVals; i++) {
				final double[] tuple = mcbm.getValues(i);
				for(int col = 0; col < nCols; col++)
					sb.append(i, col, tuple[col]);
			}
			m.recomputeNonZeros();
			return new MatrixBlockDictionary(m, nCols);
		}

		final double[] resValues = new double[nRows * nCols];
		for(int i = 0; i < nVals; i++)
			System.arraycopy(mcbm.getValues(i), 0, resValues, i * nCols, nCols);

		return new Dictionary(resValues);
	}
}
