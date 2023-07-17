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
import java.util.Map;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.bitmap.Bitmap;
import org.apache.sysds.runtime.compress.bitmap.MultiColBitmap;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.AColGroupCompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.IContainADictionary;
import org.apache.sysds.runtime.compress.colgroup.IContainDefaultTuple;
import org.apache.sysds.runtime.compress.lib.CLALibCombineGroups;
import org.apache.sysds.runtime.compress.utils.ACount.DArrCounts;
import org.apache.sysds.runtime.compress.utils.DblArrayCountHashMap;
import org.apache.sysds.runtime.compress.utils.DoubleCountHashMap;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public interface DictionaryFactory {
	static final Log LOG = LogFactory.getLog(DictionaryFactory.class.getName());

	public enum Type {
		FP64_DICT, MATRIX_BLOCK_DICT, INT8_DICT, IDENTITY, IDENTITY_SLICE,
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

	public static ADictionary create(DblArrayCountHashMap map, int nCols, boolean addZeroTuple, double sparsity) {
		try {
			final ArrayList<DArrCounts> vals = map.extractValues();
			final int nVals = vals.size();
			final int nTuplesOut = nVals + (addZeroTuple ? 1 : 0);
			if(sparsity < 0.4) {
				final MatrixBlock retB = new MatrixBlock(nTuplesOut, nCols, true);
				retB.allocateSparseRowsBlock();
				final SparseBlock sb = retB.getSparseBlock();
				for(int i = 0; i < nVals; i++) {
					final DArrCounts dac = vals.get(i);
					final double[] dv = dac.key.getData();
					for(int k = 0; k < dv.length; k++)
						sb.append(dac.id, k, dv[k]);
				}
				retB.recomputeNonZeros();
				retB.examSparsity(true);
				return MatrixBlockDictionary.create(retB);
			}
			else {

				final double[] resValues = new double[(nTuplesOut) * nCols];
				for(int i = 0; i < nVals; i++) {
					final DArrCounts dac = vals.get(i);
					System.arraycopy(dac.key.getData(), 0, resValues, dac.id * nCols, nCols);
				}
				return Dictionary.create(resValues);
			}
		}
		catch(Exception e) {
			LOG.error("Failed to create dictionary: ", e);
			return null;
		}
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
			return Dictionary.create(((Bitmap) ubm).getValues());
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
			m.examSparsity(true);
			return MatrixBlockDictionary.create(m);
		}
		else if(ubm instanceof MultiColBitmap) {
			MultiColBitmap mcbm = (MultiColBitmap) ubm;
			final int nVals = ubm.getNumValues();
			double[] resValues = new double[nVals * nCol];
			for(int i = 0; i < nVals; i++)
				System.arraycopy(mcbm.getValues(i), 0, resValues, i * nCol, nCol);

			return Dictionary.create(resValues);
		}
		throw new NotImplementedException("Not implemented creation of bitmap type : " + ubm.getClass().getSimpleName());
	}

	public static ADictionary create(ABitmap ubm, int defaultIndex, double[] defaultTuple, double sparsity,
		boolean addZero) {
		final int nCol = ubm.getNumColumns();
		final int nVal = ubm.getNumValues() - (addZero ? 0 : 1);
		if(nCol > 4 && sparsity < 0.4) {
			final MultiColBitmap mcbm = (MultiColBitmap) ubm; // always multi column

			final MatrixBlock m = new MatrixBlock(nVal, nCol, true);
			m.allocateSparseRowsBlock();
			final SparseBlock sb = m.getSparseBlock();

			for(int i = 0; i < defaultIndex; i++)
				sb.set(i, new SparseRowVector(mcbm.getValues(i)), false);

			// copy default
			System.arraycopy(mcbm.getValues(defaultIndex), 0, defaultTuple, 0, nCol);

			for(int i = defaultIndex; i < ubm.getNumValues() - 1; i++)
				sb.set(i, new SparseRowVector(mcbm.getValues(i + 1)), false);

			m.recomputeNonZeros();
			m.examSparsity(true);
			return MatrixBlockDictionary.create(m);
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
				final MultiColBitmap mcbm = (MultiColBitmap) ubm;
				for(int i = 0; i < defaultIndex; i++)
					System.arraycopy(mcbm.getValues(i), 0, dict, i * nCol, nCol);
				System.arraycopy(mcbm.getValues(defaultIndex), 0, defaultTuple, 0, nCol);
				for(int i = defaultIndex; i < ubm.getNumValues() - 1; i++)
					System.arraycopy(mcbm.getValues(i + 1), 0, dict, i * nCol, nCol);
			}
			else
				throw new NotImplementedException("not supported ABitmap of type:" + ubm.getClass().getSimpleName());

			return Dictionary.create(dict);
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
			return Dictionary.create(resValues);
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
			m.examSparsity(true);
			return MatrixBlockDictionary.create(m);
		}

		final double[] resValues = new double[nRows * nCols];
		for(int i = 0; i < nVals; i++)
			System.arraycopy(mcbm.getValues(i), 0, resValues, i * nCols, nCols);

		return Dictionary.create(resValues);
	}

	public static ADictionary create(DoubleCountHashMap map) {
		final double[] resValues = map.getDictionary();
		return Dictionary.create(resValues);
	}

	public static ADictionary combineDictionaries(AColGroupCompressed a, AColGroupCompressed b) {
		return combineDictionaries(a, b, null);
	}

	public static ADictionary combineDictionaries(AColGroupCompressed a, AColGroupCompressed b,
		Map<Integer, Integer> filter) {
		if(a instanceof ColGroupEmpty && b instanceof ColGroupEmpty)
			return null; // null return is handled elsewhere.

		CompressionType ac = a.getCompType();
		CompressionType bc = b.getCompType();

		boolean ae = a instanceof IContainADictionary;
		boolean be = b instanceof IContainADictionary;

		if(ae && be) {

			ADictionary ad = ((IContainADictionary) a).getDictionary();
			ADictionary bd = ((IContainADictionary) b).getDictionary();
			if(ac.isConst()) {
				if(bc.isConst()) {
					return Dictionary.create(CLALibCombineGroups.constructDefaultTuple(a, b));
				}
				else if(bc.isDense()) {
					final double[] at = ((IContainDefaultTuple) a).getDefaultTuple();
					return combineConstSparseSparseRet(at, bd, b.getNumCols(), filter);
				}
			}
			else if(ac.isDense()) {
				if(bc.isConst()) {
					final double[] bt = ((IContainDefaultTuple) b).getDefaultTuple();
					return combineSparseConstSparseRet(ad, a.getNumCols(), bt, filter);
				}
				else if(bc.isDense())
					return combineFullDictionaries(ad, a.getNumCols(), bd, b.getNumCols(), filter);
				else if(bc.isSDC()) {
					double[] tuple = ((IContainDefaultTuple) b).getDefaultTuple();
					return combineSDCRight(ad, a.getNumCols(), bd, tuple, filter);
				}
			}
			else if(ac.isSDC()) {
				if(bc.isSDC()) {
					final double[] at = ((IContainDefaultTuple) a).getDefaultTuple();
					final double[] bt = ((IContainDefaultTuple) b).getDefaultTuple();
					return combineSDC(ad, at, bd, bt, filter);
				}
			}
		}
		throw new NotImplementedException("Not supporting combining: " + a + " " + b);
	}

	/**
	 * Combine the dictionaries assuming a sparse combination where each dictionary can be a SDC containing a default
	 * element that have to be introduced into the combined dictionary.
	 * 
	 * @param a A Dictionary can be SDC or const
	 * @param b A Dictionary can be Const or SDC.
	 * @return The combined dictionary
	 */
	public static ADictionary combineDictionariesSparse(AColGroupCompressed a, AColGroupCompressed b) {
		CompressionType ac = a.getCompType();
		CompressionType bc = b.getCompType();

		if(ac.isSDC()) {
			ADictionary ad = ((IContainADictionary) a).getDictionary();
			if(bc.isConst()) {
				double[] bt = ((IContainDefaultTuple) b).getDefaultTuple();
				return combineSparseConstSparseRet(ad, a.getNumCols(), bt);
			}
			else if(bc.isSDC()) {
				ADictionary bd = ((IContainADictionary) b).getDictionary();
				if(a.sameIndexStructure(b)) {
					return ad.cbind(bd, b.getNumCols());
				}
				// real combine extract default and combine like dense but with default before.
			}
		}
		else if(ac.isConst()) {
			double[] at = ((IContainDefaultTuple) a).getDefaultTuple();
			if(bc.isSDC()) {
				ADictionary bd = ((IContainADictionary) b).getDictionary();
				return combineConstSparseSparseRet(at, bd, b.getNumCols());
			}
		}

		throw new NotImplementedException("Not supporting combining dense: " + a + " " + b);
	}

	/**
	 * Combine the dictionaries as if the dictionaries contain the full spectrum of the combined data.
	 * 
	 * @param a   Left side dictionary
	 * @param nca Number of columns left dictionary
	 * @param b   Right side dictionary
	 * @param ncb Number of columns right dictionary
	 * @return A combined dictionary
	 */
	public static ADictionary combineFullDictionaries(ADictionary a, int nca, ADictionary b, int ncb) {
		return combineFullDictionaries(a, nca, b, ncb, null);
	}

	/**
	 * Combine the dictionaries as if the dictionaries only contain the values in the specified filter.
	 * 
	 * @param a      Left side dictionary
	 * @param nca    Number of columns left dictionary
	 * @param b      Right side dictionary
	 * @param ncb    Number of columns right dictionary
	 * @param filter The mapping filter to not include all possible combinations in the output, this filter is allowed to
	 *               be null, that means the output is defaulting back to a full combine
	 * @return A combined dictionary
	 */
	public static ADictionary combineFullDictionaries(ADictionary a, int nca, ADictionary b, int ncb,
		Map<Integer, Integer> filter) {
		final int ra = a.getNumberOfValues(nca);
		final int rb = b.getNumberOfValues(ncb);

		MatrixBlock ma = a.getMBDict(nca).getMatrixBlock();
		MatrixBlock mb = b.getMBDict(ncb).getMatrixBlock();

		if(ra == 1 && rb == 1)
			return new MatrixBlockDictionary(ma.append(mb));

		MatrixBlock out = new MatrixBlock(filter != null ? filter.size() : ra * rb, nca + ncb, false);

		out.allocateBlock();

		if(filter != null) {
			for(int r : filter.keySet()) {
				int o = filter.get(r);
				int ia = r % ra;
				int ib = r / ra;
				for(int c = 0; c < nca; c++)
					out.quickSetValue(o, c, ma.quickGetValue(ia, c));

				for(int c = 0; c < ncb; c++)
					out.quickSetValue(o, c + nca, mb.quickGetValue(ib, c));

			}
		}
		else {

			for(int r = 0; r < out.getNumRows(); r++) {
				int ia = r % ra;
				int ib = r / ra;
				for(int c = 0; c < nca; c++)
					out.quickSetValue(r, c, ma.quickGetValue(ia, c));

				for(int c = 0; c < ncb; c++)
					out.quickSetValue(r, c + nca, mb.quickGetValue(ib, c));

			}
		}
		return new MatrixBlockDictionary(out);
	}

	public static ADictionary combineSDCRight(ADictionary a, int nca, ADictionary b, double[] tub) {

		final int ncb = tub.length;
		final int ra = a.getNumberOfValues(nca);
		final int rb = b.getNumberOfValues(ncb);

		MatrixBlock ma = a.getMBDict(nca).getMatrixBlock();
		MatrixBlock mb = b.getMBDict(ncb).getMatrixBlock();

		MatrixBlock out = new MatrixBlock(ra * (rb + 1), nca + ncb, false);

		out.allocateBlock();

		for(int r = 0; r < ra; r++) {

			for(int c = 0; c < nca; c++)
				out.quickSetValue(r, c, ma.quickGetValue(r, c));
			for(int c = 0; c < ncb; c++)
				out.quickSetValue(r, c + nca, tub[c]);
		}

		for(int r = ra; r < out.getNumRows(); r++) {
			int ia = r % ra;
			int ib = r / ra - 1;
			for(int c = 0; c < nca; c++) // all good.
				out.quickSetValue(r, c, ma.quickGetValue(ia, c));

			for(int c = 0; c < ncb; c++)
				out.quickSetValue(r, c + nca, mb.quickGetValue(ib, c));

		}
		return new MatrixBlockDictionary(out);
	}

	public static ADictionary combineSDCRight(ADictionary a, int nca, ADictionary b, double[] tub,
		Map<Integer, Integer> filter) {
		if(filter == null)
			return combineSDCRight(a, nca, b, tub);
		final int ncb = tub.length;
		final int ra = a.getNumberOfValues(nca);
		final int rb = b.getNumberOfValues(ncb);

		MatrixBlock ma = a.getMBDict(nca).getMatrixBlock();
		MatrixBlock mb = b.getMBDict(ncb).getMatrixBlock();

		MatrixBlock out = new MatrixBlock(filter.size(), nca + ncb, false);

		out.allocateBlock();

		for(int r = 0; r < ra; r++) {
			if(filter.containsKey(r)) {

				int o = filter.get(r);
				for(int c = 0; c < nca; c++)
					out.quickSetValue(o, c, ma.quickGetValue(r, c));
				for(int c = 0; c < ncb; c++)
					out.quickSetValue(o, c + nca, tub[c]);
			}

		}

		for(int r = ra; r <  ra * rb; r++) {
			if(filter.containsKey(r)) {
				int o = filter.get(r);

				int ia = r % ra;
				int ib = r / ra - 1;
				for(int c = 0; c < nca; c++) // all good.
					out.quickSetValue(o, c, ma.quickGetValue(ia, c));

				for(int c = 0; c < ncb; c++)
					out.quickSetValue(o, c + nca, mb.quickGetValue(ib, c));

			}
		}
		return new MatrixBlockDictionary(out);
	}

	public static ADictionary combineSDC(ADictionary a, double[] tua, ADictionary b, double[] tub) {
		final int nca = tua.length;
		final int ncb = tub.length;
		final int ra = a.getNumberOfValues(nca);
		final int rb = b.getNumberOfValues(ncb);

		MatrixBlock ma = a.getMBDict(nca).getMatrixBlock();
		MatrixBlock mb = b.getMBDict(ncb).getMatrixBlock();

		MatrixBlock out = new MatrixBlock((ra + 1) * (rb + 1), nca + ncb, false);

		out.allocateBlock();

		// 0 row both default tuples

		for(int c = 0; c < nca; c++)
			out.quickSetValue(0, c, tua[c]);

		for(int c = 0; c < ncb; c++)
			out.quickSetValue(0, c + nca, tub[c]);

		// default case for b and all cases for a.
		for(int r = 1; r < ra + 1; r++) {
			for(int c = 0; c < nca; c++)
				out.quickSetValue(r, c, ma.quickGetValue(r - 1, c));
			for(int c = 0; c < ncb; c++)
				out.quickSetValue(r, c + nca, tub[c]);
		}

		for(int r = ra + 1; r < out.getNumRows(); r++) {
			int ia = r % (ra + 1) - 1;
			int ib = r / (ra + 1) - 1;

			if(ia == -1)
				for(int c = 0; c < nca; c++)
					out.quickSetValue(r, c, tua[c]);
			else
				for(int c = 0; c < nca; c++)
					out.quickSetValue(r, c, ma.quickGetValue(ia, c));

			for(int c = 0; c < ncb; c++) // all good here.
				out.quickSetValue(r, c + nca, mb.quickGetValue(ib, c));

		}

		return new MatrixBlockDictionary(out);
	}

	public static ADictionary combineSDC(ADictionary a, double[] tua, ADictionary b, double[] tub,
		Map<Integer, Integer> filter) {
		if(filter == null)
			return combineSDC(a, tua, b, tub);
		final int nca = tua.length;
		final int ncb = tub.length;
		final int ra = a.getNumberOfValues(nca);
		final int rb = b.getNumberOfValues(nca);

		MatrixBlock ma = a.getMBDict(nca).getMatrixBlock();
		MatrixBlock mb = b.getMBDict(ncb).getMatrixBlock();

		MatrixBlock out = new MatrixBlock(filter.size(), nca + ncb, false);

		out.allocateBlock();

		// 0 row both default tuples
		if(filter.containsKey(0)) {
			int o = filter.get(0);
			for(int c = 0; c < nca; c++)
				out.quickSetValue(o, c, tua[c]);

			for(int c = 0; c < ncb; c++)
				out.quickSetValue(o, c + nca, tub[c]);
		}

		// default case for b and all cases for a.
		for(int r = 1; r < ra + 1; r++) {
			if(filter.containsKey(r)) {
				int o = filter.get(r);
				for(int c = 0; c < nca; c++)
					out.quickSetValue(o, c, ma.quickGetValue(r - 1, c));
				for(int c = 0; c < ncb; c++)
					out.quickSetValue(o, c + nca, tub[c]);
			}
		}

		for(int r = ra + 1; r < ra * rb; r++) {

			if(filter.containsKey(r)) {
				int o = filter.get(r);

				int ia = r % (ra + 1) - 1;
				int ib = r / (ra + 1) - 1;

				if(ia == -1)
					for(int c = 0; c < nca; c++)
						out.quickSetValue(o, c, tua[c]);
				else
					for(int c = 0; c < nca; c++)
						out.quickSetValue(o, c, ma.quickGetValue(ia, c));

				for(int c = 0; c < ncb; c++) // all good here.
					out.quickSetValue(o, c + nca, mb.quickGetValue(ib, c));
			}
		}

		return new MatrixBlockDictionary(out);

	}

	public static ADictionary combineSparseConstSparseRet(ADictionary a, int nca, double[] tub) {
		final int ncb = tub.length;
		final int ra = a.getNumberOfValues(nca);

		MatrixBlock ma = a.getMBDict(nca).getMatrixBlock();

		MatrixBlock out = new MatrixBlock(ra, nca + ncb, false);

		out.allocateBlock();

		// default case for b and all cases for a.
		for(int r = 0; r < ra; r++) {
			for(int c = 0; c < nca; c++)
				out.quickSetValue(r, c, ma.quickGetValue(r, c));
			for(int c = 0; c < ncb; c++)
				out.quickSetValue(r, c + nca, tub[c]);
		}

		return new MatrixBlockDictionary(out);

	}

	private static ADictionary combineSparseConstSparseRet(ADictionary a, int nca, double[] tub,
		Map<Integer, Integer> filter) {
		if(filter == null)
			return combineSparseConstSparseRet(a, nca, tub);
		else
			throw new NotImplementedException();
		// final int ncb = tub.length;
		// final int ra = a.getNumberOfValues(nca);

		// MatrixBlock ma = a.getMBDict(nca).getMatrixBlock();

		// MatrixBlock out = new MatrixBlock(ra, nca + ncb, false);

		// out.allocateBlock();

		// // default case for b and all cases for a.
		// for(int r = 0; r < ra; r++) {
		// for(int c = 0; c < nca; c++)
		// out.quickSetValue(r, c, ma.quickGetValue(r, c));
		// for(int c = 0; c < ncb; c++)
		// out.quickSetValue(r, c + nca, tub[c]);
		// }

		// return new MatrixBlockDictionary(out);

	}

	public static ADictionary combineConstSparseSparseRet(double[] tua, ADictionary b, int ncb) {
		final int nca = tua.length;
		final int rb = b.getNumberOfValues(ncb);

		MatrixBlock mb = b.getMBDict(ncb).getMatrixBlock();

		MatrixBlock out = new MatrixBlock(rb, nca + ncb, false);

		out.allocateBlock();

		// default case for b and all cases for a.
		for(int r = 0; r < rb; r++) {
			for(int c = 0; c < nca; c++)
				out.quickSetValue(r, c, tua[c]);
			for(int c = 0; c < ncb; c++)
				out.quickSetValue(r, c + nca, mb.quickGetValue(r, c));
		}

		return new MatrixBlockDictionary(out);

	}

	private static ADictionary combineConstSparseSparseRet(double[] tua, ADictionary b, int ncb,
		Map<Integer, Integer> filter) {
		if(filter == null)
			return combineConstSparseSparseRet(tua, b, ncb);
		else
			throw new NotImplementedException();
		// final int nca = tua.length;
		// final int rb = b.getNumberOfValues(ncb);

		// MatrixBlock mb = b.getMBDict(ncb).getMatrixBlock();

		// MatrixBlock out = new MatrixBlock(rb, nca + ncb, false);

		// out.allocateBlock();

		// // default case for b and all cases for a.
		// for(int r = 0; r < rb; r++) {
		// for(int c = 0; c < nca; c++)
		// out.quickSetValue(r, c, tua[c]);
		// for(int c = 0; c < ncb; c++)
		// out.quickSetValue(r, c + nca, mb.quickGetValue(r, c));
		// }

		// return new MatrixBlockDictionary(out);

	}
}
