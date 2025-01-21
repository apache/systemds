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
import java.util.List;

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
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.lib.CLALibCombineGroups;
import org.apache.sysds.runtime.compress.utils.ACount;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayCountHashMap;
import org.apache.sysds.runtime.compress.utils.DoubleCountHashMap;
import org.apache.sysds.runtime.compress.utils.HashMapLongInt;
import org.apache.sysds.runtime.compress.utils.HashMapLongInt.KV;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;

public interface DictionaryFactory {
	static final Log LOG = LogFactory.getLog(DictionaryFactory.class.getName());

	public enum Type {
		FP64_DICT, MATRIX_BLOCK_DICT, INT8_DICT, IDENTITY, IDENTITY_SLICE, PLACE_HOLDER
	}

	public static IDictionary read(DataInput in) throws IOException {
		final Type type = Type.values()[in.readByte()];
		switch(type) {
			case FP64_DICT:
				return Dictionary.read(in);
			case INT8_DICT:
				return QDictionary.read(in);
			case PLACE_HOLDER:
				return PlaceHolderDict.read(in);
			case IDENTITY:
				return IdentityDictionary.read(in);
			case IDENTITY_SLICE:
				return IdentityDictionarySlice.read(in);
			case MATRIX_BLOCK_DICT:
			default:
				return MatrixBlockDictionary.read(in);
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

	public static IDictionary create(DblArrayCountHashMap map, int nCols, boolean addZeroTuple, double sparsity) {

		final ACount<DblArray>[] vals = map.extractValues();
		final int nVals = vals.length;
		final int nTuplesOut = nVals + (addZeroTuple ? 1 : 0);
		if(sparsity < 0.4) {
			final MatrixBlock retB = new MatrixBlock(nTuplesOut, nCols, true);
			retB.allocateSparseRowsBlock();
			final SparseBlock sb = retB.getSparseBlock();
			for(int i = 0; i < nVals; i++) {
				final ACount<DblArray> dac = vals[i];
				final double[] dv = dac.key().getData();
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
				final ACount<DblArray> dac = vals[i];
				System.arraycopy(dac.key().getData(), 0, resValues, dac.id * nCols, nCols);
			}
			return Dictionary.create(resValues);
		}

	}

	public static IDictionary create(ABitmap ubm) {
		return create(ubm, 1.0);
	}

	public static IDictionary create(ABitmap ubm, double sparsity) {
		final int nCol = ubm.getNumColumns();
		if(ubm instanceof Bitmap)
			return Dictionary.create(((Bitmap) ubm).getValues());
		else if(sparsity < 0.4 && nCol > 4) { // && ubm instanceof MultiColBitmap
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
		else {// if(ubm instanceof MultiColBitmap) {
			MultiColBitmap mcbm = (MultiColBitmap) ubm;
			final int nVals = ubm.getNumValues();
			double[] resValues = new double[nVals * nCol];
			for(int i = 0; i < nVals; i++)
				System.arraycopy(mcbm.getValues(i), 0, resValues, i * nCol, nCol);

			return Dictionary.create(resValues);
		}
	}

	public static IDictionary create(ABitmap ubm, int defaultIndex, double[] defaultTuple, double sparsity,
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
			else { // if(ubm instanceof MultiColBitmap) {
				final MultiColBitmap mcbm = (MultiColBitmap) ubm;
				for(int i = 0; i < defaultIndex; i++)
					System.arraycopy(mcbm.getValues(i), 0, dict, i * nCol, nCol);
				System.arraycopy(mcbm.getValues(defaultIndex), 0, defaultTuple, 0, nCol);
				for(int i = defaultIndex; i < ubm.getNumValues() - 1; i++)
					System.arraycopy(mcbm.getValues(i + 1), 0, dict, i * nCol, nCol);
			}

			return Dictionary.create(dict);
		}
	}

	// public static IDictionary createWithAppendedZeroTuple(ABitmap ubm, double sparsity) {
	// final int nVals = ubm.getNumValues();
	// final int nRows = nVals + 1;
	// final int nCols = ubm.getNumColumns();

	// if(ubm instanceof Bitmap) {
	// final double[] resValues = new double[nRows];
	// final double[] from = ((Bitmap) ubm).getValues();
	// System.arraycopy(from, 0, resValues, 0, from.length);
	// return Dictionary.create(resValues);
	// }

	// final MultiColBitmap mcbm = (MultiColBitmap) ubm;
	// if(sparsity < 0.4 && nCols > 4) {
	// final MatrixBlock m = new MatrixBlock(nRows, nCols, true);
	// m.allocateSparseRowsBlock();
	// final SparseBlock sb = m.getSparseBlock();

	// for(int i = 0; i < nVals; i++) {
	// final double[] tuple = mcbm.getValues(i);
	// for(int col = 0; col < nCols; col++)
	// sb.append(i, col, tuple[col]);
	// }
	// m.recomputeNonZeros();
	// m.examSparsity(true);
	// return MatrixBlockDictionary.create(m);
	// }

	// final double[] resValues = new double[nRows * nCols];
	// for(int i = 0; i < nVals; i++)
	// System.arraycopy(mcbm.getValues(i), 0, resValues, i * nCols, nCols);

	// return Dictionary.create(resValues);
	// }

	public static IDictionary create(DoubleCountHashMap map) {
		final double[] resValues = map.getDictionary();
		return Dictionary.create(resValues);
	}

	public static IDictionary combineDictionaries(AColGroupCompressed a, AColGroupCompressed b) {
		return combineDictionaries(a, b, null);
	}

	public static IDictionary combineDictionaries(AColGroupCompressed a, AColGroupCompressed b, HashMapLongInt filter) {
		if(a instanceof ColGroupEmpty && b instanceof ColGroupEmpty)
			return null; // null return is handled elsewhere.

		CompressionType ac = a.getCompType();
		CompressionType bc = b.getCompType();

		boolean ae = a instanceof IContainADictionary;
		boolean be = b instanceof IContainADictionary;

		if(ae && be) {

			final IDictionary ad = ((IContainADictionary) a).getDictionary();
			final IDictionary bd = ((IContainADictionary) b).getDictionary();
			if(ac.isConst()) {
				if(bc.isConst()) {
					return Dictionary.create(CLALibCombineGroups.constructDefaultTuple(a, b));
				}
				else if(bc.isDense()) {
					final double[] at = ((IContainDefaultTuple) a).getDefaultTuple();
					Pair<int[], int[]> r = IColIndex.reorderingIndexes(a.getColIndices(), b.getColIndices());
					return combineConstLeft(at, bd, b.getNumCols(), r.getKey(), r.getValue(), filter);
				}
			}
			else if(ac.isDense()) {
				if(bc.isConst()) {
					final Pair<int[], int[]> r = IColIndex.reorderingIndexes(a.getColIndices(), b.getColIndices());
					final double[] bt = ((IContainDefaultTuple) b).getDefaultTuple();
					return combineSparseConstSparseRet(ad, a.getNumCols(), bt, r.getKey(), r.getValue(), filter);
				}
				else if(bc.isDense()) {
					return combineFullDictionaries(ad, a.getColIndices(), bd, b.getColIndices(), filter);
				}
				else if(bc.isSDC()) {
					double[] tuple = ((IContainDefaultTuple) b).getDefaultTuple();
					return combineSDCRight(ad, a.getColIndices(), bd, tuple, b.getColIndices(), filter);
				}
			}
			else if(ac.isSDC()) {
				if(bc.isSDC()) {
					final double[] at = ((IContainDefaultTuple) a).getDefaultTuple();
					final double[] bt = ((IContainDefaultTuple) b).getDefaultTuple();
					return combineSDCFilter(ad, at, a.getColIndices(), bd, bt, b.getColIndices(), filter);
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
	public static IDictionary combineDictionariesSparse(AColGroupCompressed a, AColGroupCompressed b) {
		return combineDictionariesSparse(a, b, null);
	}

	/**
	 * Combine the dictionaries assuming a sparse combination where each dictionary can be a SDC containing a default
	 * element that have to be introduced into the combined dictionary.
	 * 
	 * @param a      A Dictionary can be SDC or const
	 * @param b      A Dictionary can be Const or SDC
	 * @param filter A filter to remove elements in the combined dictionary
	 * @return The combined dictionary
	 */
	public static IDictionary combineDictionariesSparse(AColGroupCompressed a, AColGroupCompressed b,
		HashMapLongInt filter) {
		CompressionType ac = a.getCompType();
		CompressionType bc = b.getCompType();

		if(filter != null)
			throw new NotImplementedException("Not supported filter for sparse join yet!");

		if(ac.isSDC()) {
			final IDictionary ad = ((IContainADictionary) a).getDictionary();
			if(bc.isConst()) {
				final Pair<int[], int[]> r = IColIndex.reorderingIndexes(a.getColIndices(), b.getColIndices());
				double[] bt = ((IContainDefaultTuple) b).getDefaultTuple();
				return combineSparseConstSparseRet(ad, a.getNumCols(), bt, r.getKey(), r.getValue());
			}
			else if(bc.isSDC()) {
				final IDictionary bd = ((IContainADictionary) b).getDictionary();
				if(a.sameIndexStructure(b)) {
					// in order or other order..
					if(IColIndex.inOrder(a.getColIndices(), b.getColIndices()))
						return ad.cbind(bd, b.getNumCols());
					else if(IColIndex.inOrder(b.getColIndices(), a.getColIndices()))
						return bd.cbind(ad, b.getNumCols());
					else {
						final Pair<int[], int[]> r = IColIndex.reorderingIndexes(a.getColIndices(), b.getColIndices());
						return cbindReorder(ad, bd, r.getKey(), r.getValue());
					}
				}
			}
		}
		else if(ac.isConst()) {
			final double[] at = ((IContainDefaultTuple) a).getDefaultTuple();
			if(bc.isSDC()) {
				final IDictionary bd = ((IContainADictionary) b).getDictionary();
				final Pair<int[], int[]> r = IColIndex.reorderingIndexes(a.getColIndices(), b.getColIndices());
				return combineConstLeftAll(at, bd, b.getNumCols(), r.getKey(), r.getValue());
			}
		}

		throw new NotImplementedException("Not supporting combining dense: " + a + " " + b);
	}

	private static IDictionary cbindReorder(IDictionary a, IDictionary b, int[] ai, int[] bi) {
		final int nca = ai.length;
		final int ncb = bi.length;
		final int ra = a.getNumberOfValues(nca);
		final int rb = b.getNumberOfValues(ncb);
		final MatrixBlock ma = a.getMBDict(nca).getMatrixBlock();
		final MatrixBlock mb = b.getMBDict(ncb).getMatrixBlock();
		if(ra != rb)
			throw new DMLCompressionException("Invalid cbind reorder, different sizes of dictionaries");
		final MatrixBlock out = new MatrixBlock(ra, nca + ncb, false);

		for(int r = 0; r < ra; r++) {// each row
			//
			for(int c = 0; c < nca; c++)
				out.set(r, ai[c], ma.get(r, c));

			for(int c = 0; c < ncb; c++)
				out.set(r, bi[c], mb.get(r, c));
		}

		return new MatrixBlockDictionary(out);
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
	public static IDictionary combineFullDictionaries(IDictionary a, int nca, IDictionary b, int ncb) {
		return combineFullDictionaries(a, nca, b, ncb, null);
	}

	public static IDictionary combineFullDictionaries(IDictionary a, IColIndex ai, IDictionary b, IColIndex bi,
		HashMapLongInt filter) {
		final int nca = ai.size();
		final int ncb = bi.size();
		return combineFullDictionaries(a, ai, nca, b, bi, ncb, filter);
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
	public static IDictionary combineFullDictionaries(IDictionary a, int nca, IDictionary b, int ncb,
		HashMapLongInt filter) {
		return combineFullDictionaries(a, null, nca, b, null, ncb, filter);
	}

	public static IDictionary combineFullDictionaries(IDictionary a, IColIndex ai, int nca, IDictionary b, IColIndex bi,
		int ncb, HashMapLongInt filter) {
		final int ra = a.getNumberOfValues(nca);
		final int rb = b.getNumberOfValues(ncb);

		final MatrixBlock ma = a.getMBDict(nca).getMatrixBlock();
		final MatrixBlock mb = b.getMBDict(ncb).getMatrixBlock();
		final int filterSize = filter != null ? filter.size() : ra * rb;
		if(filterSize == 0)
			return null;
		final MatrixBlock out = new MatrixBlock(filterSize, nca + ncb, false);
		out.allocateBlock();

		if(ai != null && bi != null && !IColIndex.inOrder(ai, bi)) {

			Pair<int[], int[]> reordering = IColIndex.reorderingIndexes(ai, bi);
			if(filter != null)
				// throw new NotImplementedException();
				combineFullDictionariesOOOFilter(out, filter, ra, rb, nca, ncb, reordering.getKey(), reordering.getValue(),
					ma, mb);
			else
				combineFullDictionariesOOONoFilter(out, ra, rb, nca, ncb, reordering.getKey(), reordering.getValue(), ma,
					mb);

		}
		else {
			if(filter != null)
				combineFullDictionariesFilter(out, filter, ra, rb, nca, ncb, ma, mb);
			else
				combineFullDictionariesNoFilter(out, ra, rb, nca, ncb, ma, mb);
		}

		out.examSparsity(true);

		return new MatrixBlockDictionary(out);
	}

	private static void combineFullDictionariesFilter(MatrixBlock out, HashMapLongInt filter, int ra, int rb, int nca,
		int ncb, MatrixBlock ma, MatrixBlock mb) {
		for(KV k : filter) {
			final int r = (int) (k.k);
			final int o = k.v;
			int ia = r % ra;
			int ib = r / ra;
			for(int c = 0; c < nca; c++)
				out.set(o, c, ma.get(ia, c));

			for(int c = 0; c < ncb; c++)
				out.set(o, c + nca, mb.get(ib, c));
		}
	}

	private static void combineFullDictionariesOOOFilter(MatrixBlock out, HashMapLongInt filter, int ra, int rb, int nca,
		int ncb, int[] ai, int[] bi, MatrixBlock ma, MatrixBlock mb) {
		for(KV k : filter) {
			final int r = (int) (k.k);
			final int o = k.v;
			int ia = r % ra;
			int ib = r / ra;
			for(int c = 0; c < nca; c++)
				out.set(o, ai[c], ma.get(ia, c));
			for(int c = 0; c < ncb; c++)
				out.set(o, bi[c], mb.get(ib, c));
		}
	}

	private static void combineFullDictionariesOOONoFilter(MatrixBlock out, int ra, int rb, int nca, int ncb, int[] ai,
		int[] bi, MatrixBlock ma, MatrixBlock mb) {
		for(int r = 0; r < out.getNumRows(); r++) {
			int ia = r % ra;
			int ib = r / ra;
			for(int c = 0; c < nca; c++)
				out.set(r, ai[c], ma.get(ia, c));
			for(int c = 0; c < ncb; c++)
				out.set(r, bi[c], mb.get(ib, c));
		}
	}

	private static void combineFullDictionariesNoFilter(MatrixBlock out, int ra, int rb, int nca, int ncb,
		MatrixBlock ma, MatrixBlock mb) {
		for(int r = 0; r < out.getNumRows(); r++) {
			int ia = r % ra;
			int ib = r / ra;
			for(int c = 0; c < nca; c++)
				out.set(r, c, ma.get(ia, c));
			for(int c = 0; c < ncb; c++)
				out.set(r, c + nca, mb.get(ib, c));
		}
	}

	public static IDictionary combineSDCRightNoFilter(IDictionary a, int nca, IDictionary b, double[] tub) {
		return combineSDCRightNoFilter(a, null, nca, b, tub, null);
	}

	public static IDictionary combineSDCRightNoFilter(IDictionary a, IColIndex ai, int nca, IDictionary b, double[] tub,
		IColIndex bi) {
		if(ai != null || bi != null)
			throw new NotImplementedException();
		final int ncb = tub.length;
		final int ra = a.getNumberOfValues(nca);
		final int rb = b.getNumberOfValues(ncb);
		final MatrixBlock ma = a.getMBDict(nca).getMatrixBlock();
		final MatrixBlock mb = b.getMBDict(ncb).getMatrixBlock();
		final MatrixBlock out = new MatrixBlock(ra * (rb + 1), nca + ncb, false);

		out.allocateBlock();

		for(int r = 0; r < ra; r++) {

			for(int c = 0; c < nca; c++)
				out.set(r, c, ma.get(r, c));
			for(int c = 0; c < ncb; c++)
				out.set(r, c + nca, tub[c]);
		}

		for(int r = ra; r < out.getNumRows(); r++) {
			int ia = r % ra;
			int ib = r / ra - 1;
			for(int c = 0; c < nca; c++) // all good.
				out.set(r, c, ma.get(ia, c));

			for(int c = 0; c < ncb; c++)
				out.set(r, c + nca, mb.get(ib, c));

		}
		return new MatrixBlockDictionary(out);
	}

	public static IDictionary combineSDCRight(IDictionary a, IColIndex ai, IDictionary b, double[] tub, IColIndex bi,
		HashMapLongInt filter) {
		return combineSDCRight(a, ai, ai.size(), b, tub, bi, filter);
	}

	public static IDictionary combineSDCRight(IDictionary a, int nca, IDictionary b, double[] tub,
		HashMapLongInt filter) {
		return combineSDCRight(a, null, nca, b, tub, null, filter);
	}

	public static IDictionary combineSDCRight(IDictionary a, IColIndex ai, int nca, IDictionary b, double[] tub,
		IColIndex bi, HashMapLongInt filter) {
		if(filter == null)
			return combineSDCRightNoFilter(a, ai, nca, b, tub, bi);

		final int ncb = tub.length;
		final int ra = a.getNumberOfValues(nca);
		final int rb = b.getNumberOfValues(ncb);
		final MatrixBlock ma = a.getMBDict(nca).getMatrixBlock();
		final MatrixBlock mb = b.getMBDict(ncb).getMatrixBlock();
		final MatrixBlock out = new MatrixBlock(filter.size(), nca + ncb, false);
		out.allocateBlock();

		if(ai != null && bi != null) {
			Pair<int[], int[]> re = IColIndex.reorderingIndexes(ai, bi);
			combineSDCRightOOOFilter(out, nca, ncb, tub, ra, rb, ma, mb, re.getKey(), re.getValue(), filter);
		}
		else {
			combineSDCRightFilter(out, nca, ncb, tub, ra, rb, ma, mb, filter);
		}
		return new MatrixBlockDictionary(out);
	}

	private static void combineSDCRightFilter(MatrixBlock out, int nca, int ncb, double[] tub, int ra, int rb,
		MatrixBlock ma, MatrixBlock mb, HashMapLongInt filter) {
		for(int r = 0; r < ra; r++) {
			int o = filter.get(r);
			if(o != -1) {
				for(int c = 0; c < nca; c++)
					out.set(o, c, ma.get(r, c));
				for(int c = 0; c < ncb; c++)
					out.set(o, c + nca, tub[c]);
			}
		}
		for(int r = ra; r < ra * rb + ra; r++) {
			int o = filter.get(r);
			if(o != -1) {
				int ia = r % ra;
				int ib = r / ra - 1;
				for(int c = 0; c < nca; c++) // all good.
					out.set(o, c, ma.get(ia, c));
				for(int c = 0; c < ncb; c++)
					out.set(o, c + nca, mb.get(ib, c));
			}
		}
	}

	private static void combineSDCRightOOOFilter(MatrixBlock out, int nca, int ncb, double[] tub, int ra, int rb,
		MatrixBlock ma, MatrixBlock mb, int[] ai, int[] bi, HashMapLongInt filter) {
		for(int r = 0; r < ra; r++) {
			int o = filter.get(r);
			if(o != -1) {
				for(int c = 0; c < nca; c++)
					out.set(o, ai[c], ma.get(r, c));
				for(int c = 0; c < ncb; c++)
					out.set(o, bi[c], tub[c]);
			}
		}
		for(int r = ra; r < ra * rb + ra; r++) {
			int o = filter.get(r);
			if(o != -1) {
				int ia = r % ra;
				int ib = r / ra - 1;
				for(int c = 0; c < nca; c++) // all good.
					out.set(o, ai[c], ma.get(ia, c));
				for(int c = 0; c < ncb; c++)
					out.set(o, bi[c], mb.get(ib, c));
			}
		}
	}

	public static IDictionary combineSDCNoFilter(IDictionary a, double[] tua, IDictionary b, double[] tub) {
		return combineSDCNoFilter(a, tua, null, b, tub, null);
	}

	public static IDictionary combineSDCNoFilter(IDictionary a, double[] tua, IColIndex ai, IDictionary b, double[] tub,
		IColIndex bi) {
		final int nca = tua.length;
		final int ncb = tub.length;
		final int ra = a.getNumberOfValues(nca);
		final int rb = b.getNumberOfValues(ncb);
		final MatrixBlock ma = a.getMBDict(nca).getMatrixBlock();
		final MatrixBlock mb = b.getMBDict(ncb).getMatrixBlock();
		final MatrixBlock out = new MatrixBlock((ra + 1) * (rb + 1), nca + ncb, false);

		out.allocateBlock();

		if(ai != null || bi != null) {
			final Pair<int[], int[]> re = IColIndex.reorderingIndexes(ai, bi);
			combineSDCNoFilterOOO(nca, ncb, tua, tub, out, ma, mb, ra, rb, re.getKey(), re.getValue());
		}
		else
			combineSDCNoFilter(nca, ncb, tua, tub, out, ma, mb, ra, rb);
		return new MatrixBlockDictionary(out);
	}

	private static void combineSDCNoFilter(int nca, int ncb, double[] tua, double[] tub, MatrixBlock out, MatrixBlock ma,
		MatrixBlock mb, int ra, int rb) {

		// 0 row both default tuples
		for(int c = 0; c < nca; c++)
			out.set(0, c, tua[c]);

		for(int c = 0; c < ncb; c++)
			out.set(0, c + nca, tub[c]);

		// default case for b and all cases for a.
		for(int r = 1; r < ra + 1; r++) {
			for(int c = 0; c < nca; c++)
				out.set(r, c, ma.get(r - 1, c));
			for(int c = 0; c < ncb; c++)
				out.set(r, c + nca, tub[c]);
		}

		for(int r = ra + 1; r < out.getNumRows(); r++) {
			final int ia = r % (ra + 1) - 1;
			final int ib = r / (ra + 1) - 1;

			if(ia == -1)
				for(int c = 0; c < nca; c++)
					out.set(r, c, tua[c]);
			else
				for(int c = 0; c < nca; c++)
					out.set(r, c, ma.get(ia, c));

			for(int c = 0; c < ncb; c++) // all good here.
				out.set(r, c + nca, mb.get(ib, c));
		}
	}

	private static void combineSDCNoFilterOOO(int nca, int ncb, double[] tua, double[] tub, MatrixBlock out,
		MatrixBlock ma, MatrixBlock mb, int ra, int rb, int[] ai, int[] bi) {

		// 0 row both default tuples
		for(int c = 0; c < nca; c++)
			out.set(0, ai[c], tua[c]);

		for(int c = 0; c < ncb; c++)
			out.set(0, bi[c], tub[c]);

		// default case for b and all cases for a.
		for(int r = 1; r < ra + 1; r++) {
			for(int c = 0; c < nca; c++)
				out.set(r, ai[c], ma.get(r - 1, c));
			for(int c = 0; c < ncb; c++)
				out.set(r, bi[c], tub[c]);
		}

		for(int r = ra + 1; r < out.getNumRows(); r++) {
			final int ia = r % (ra + 1) - 1;
			final int ib = r / (ra + 1) - 1;

			if(ia == -1)
				for(int c = 0; c < nca; c++)
					out.set(r, ai[c], tua[c]);
			else
				for(int c = 0; c < nca; c++)
					out.set(r, ai[c], ma.get(ia, c));

			for(int c = 0; c < ncb; c++) // all good here.
				out.set(r, bi[c], mb.get(ib, c));
		}
	}

	public static IDictionary combineSDCFilter(IDictionary a, double[] tua, IDictionary b, double[] tub,
		HashMapLongInt filter) {
		return combineSDCFilter(a, tua, null, b, tub, null, filter);
	}

	public static IDictionary combineSDCFilter(IDictionary a, double[] tua, IColIndex ai, IDictionary b, double[] tub,
		IColIndex bi, HashMapLongInt filter) {
		if(filter == null)
			return combineSDCNoFilter(a, tua, ai, b, tub, bi);

		final int nca = tua.length;
		final int ncb = tub.length;
		final int ra = a.getNumberOfValues(nca);
		final int rb = b.getNumberOfValues(ncb);
		final MatrixBlock ma = a.getMBDict(nca).getMatrixBlock();
		final MatrixBlock mb = b.getMBDict(ncb).getMatrixBlock();
		final MatrixBlock out = new MatrixBlock(filter.size(), nca + ncb, false);
		out.allocateBlock();

		if(ai != null && bi != null) {
			Pair<int[], int[]> re = IColIndex.reorderingIndexes(ai, bi);
			combineSDCFilterOOO(filter, nca, ncb, tua, tub, out, ma, mb, ra, rb, re.getKey(), re.getValue());
		}
		else
			combineSDCFilter(filter, nca, ncb, tua, tub, out, ma, mb, ra, rb);

		return new MatrixBlockDictionary(out);
	}

	private static void combineSDCFilter(HashMapLongInt filter, int nca, int ncb, double[] tua, double[] tub,
		MatrixBlock out, MatrixBlock ma, MatrixBlock mb, int ra, int rb) {

		// 0 row both default tuples
		final int o0 = filter.get(0);
		if(o0 != -1) {
			for(int c = 0; c < nca; c++)
				out.set(o0, c, tua[c]);

			for(int c = 0; c < ncb; c++)
				out.set(o0, c + nca, tub[c]);
		}

		// default case for b and all cases for a.
		for(int r = 1; r < ra + 1; r++) {
			final int o = filter.get(r);
			if(o != -1) {
				for(int c = 0; c < nca; c++)
					out.set(o, c, ma.get(r - 1, c));
				for(int c = 0; c < ncb; c++)
					out.set(o, c + nca, tub[c]);
			}
		}

		for(int r = ra + 1; r < ra * rb + ra + rb + 1; r++) {
			final int o = filter.get(r);
			if(o != -1) {
				final int ia = r % (ra + 1) - 1;
				final int ib = r / (ra + 1) - 1;

				if(ia == -1)
					for(int c = 0; c < nca; c++)
						out.set(o, c, tua[c]);
				else
					for(int c = 0; c < nca; c++)
						out.set(o, c, ma.get(ia, c));

				for(int c = 0; c < ncb; c++) // all good here.
					out.set(o, c + nca, mb.get(ib, c));
			}
		}
	}

	private static void combineSDCFilterOOO(HashMapLongInt filter, int nca, int ncb, double[] tua, double[] tub,
		MatrixBlock out, MatrixBlock ma, MatrixBlock mb, int ra, int rb, int[] ai, int[] bi) {
		// 0 row both default tuples
		final int o0 = filter.get(0);
		if(o0 != -1) {
			for(int c = 0; c < nca; c++)
				out.set(o0, ai[c], tua[c]);
			for(int c = 0; c < ncb; c++)
				out.set(o0, bi[c], tub[c]);
		}

		// default case for b and all cases for a.
		for(int r = 1; r < ra + 1; r++) {
			final int o = filter.get(r);
			if(o != -1) {
				for(int c = 0; c < nca; c++)
					out.set(o, ai[c], ma.get(r - 1, c));
				for(int c = 0; c < ncb; c++)
					out.set(o, bi[c], tub[c]);
			}
		}

		for(int r = ra + 1; r < ra * rb + ra + rb + 1; r++) {
			final int o = filter.get(r);
			if(o != -1) {
				final int ia = r % (ra + 1) - 1;
				final int ib = r / (ra + 1) - 1;

				if(ia == -1)
					for(int c = 0; c < nca; c++)
						out.set(o, ai[c], tua[c]);
				else
					for(int c = 0; c < nca; c++)
						out.set(o, ai[c], ma.get(ia, c));

				for(int c = 0; c < ncb; c++) // all good here.
					out.set(o, bi[c], mb.get(ib, c));
			}
		}
	}

	private static IDictionary combineSparseConstSparseRet(IDictionary a, int nca, double[] tub, int[] ai, int[] bi) {
		final int ncb = tub.length;
		final int ra = a.getNumberOfValues(nca);

		MatrixBlock ma = a.getMBDict(nca).getMatrixBlock();

		MatrixBlock out = new MatrixBlock(ra, nca + ncb, false);

		out.allocateBlock();

		// default case for b and all cases for a.
		for(int r = 0; r < ra; r++) {
			for(int c = 0; c < nca; c++)
				out.set(r, ai[c], ma.get(r, c));
			for(int c = 0; c < ncb; c++)
				out.set(r, bi[c], tub[c]);
		}

		return new MatrixBlockDictionary(out);

	}

	private static IDictionary combineSparseConstSparseRet(IDictionary a, int nca, double[] tub, int[] ai, int[] bi,
		HashMapLongInt filter) {
		if(filter == null)
			return combineSparseConstSparseRet(a, nca, tub, ai, bi);
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
		// out.set(r, c, ma.get(r, c));
		// for(int c = 0; c < ncb; c++)
		// out.set(r, c + nca, tub[c]);
		// }

		// return new MatrixBlockDictionary(out);

	}

	private static IDictionary combineConstLeftAll(double[] tua, IDictionary b, int ncb, int[] ai, int[] bi) {

		final int nca = tua.length;
		final int rb = b.getNumberOfValues(ncb);

		MatrixBlock mb = b.getMBDict(ncb).getMatrixBlock();

		MatrixBlock out = new MatrixBlock(rb, nca + ncb, false);

		out.allocateBlock();

		// default case for b and all cases for a.
		for(int r = 0; r < rb; r++) {
			for(int c = 0; c < nca; c++)
				out.set(r, ai[c], tua[c]);
			for(int c = 0; c < ncb; c++)
				out.set(r, bi[c], mb.get(r, c));
		}

		return new MatrixBlockDictionary(out);

	}

	private static IDictionary combineConstLeft(double[] tua, IDictionary b, int ncb, int[] ai, int[] bi,
		HashMapLongInt filter) {
		if(filter == null)
			return combineConstLeftAll(tua, b, ncb, ai, bi);
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
		// out.set(r, c, tua[c]);
		// for(int c = 0; c < ncb; c++)
		// out.set(r, c + nca, mb.get(r, c));
		// }

		// return new MatrixBlockDictionary(out);

	}

	public static IDictionary cBindDictionaries(int nCol, List<IDictionary> dicts) {
		MatrixBlockDictionary baseDict = dicts.get(0).getMBDict(nCol);
		MatrixBlock base = baseDict == null ? new MatrixBlock(1, nCol, true) : baseDict.getMatrixBlock();
		MatrixBlock[] others = new MatrixBlock[dicts.size() - 1];
		for(int i = 1; i < dicts.size(); i++) {
			MatrixBlockDictionary otherDict = dicts.get(i).getMBDict(nCol);
			MatrixBlock otherBase = otherDict == null ? new MatrixBlock(1, nCol, true) : otherDict.getMatrixBlock();
			others[i - 1] = otherBase;
		}
		MatrixBlock ret = base.append(others, null, true);
		return MatrixBlockDictionary.create(ret, true);
	}

	// public static IDictionary cBindDictionaries(List<Pair<Integer, IDictionary>> dicts) {
	// MatrixBlock base = dicts.get(0).getValue().getMBDict(dicts.get(0).getKey()).getMatrixBlock();
	// MatrixBlock[] others = new MatrixBlock[dicts.size() - 1];
	// for(int i = 1; i < dicts.size(); i++) {
	// Pair<Integer, IDictionary> p = dicts.get(i);
	// others[i - 1] = p.getValue().getMBDict(p.getKey()).getMatrixBlock();
	// }
	// MatrixBlock ret = base.append(others, null, true);
	// return new MatrixBlockDictionary(ret);
	// }

	public static IDictionary cBindDictionaries(IDictionary left, IDictionary right, int nColLeft, int nColRight) {
		MatrixBlockDictionary base = left.getMBDict(nColLeft);
		MatrixBlockDictionary add = right.getMBDict(nColRight);

		MatrixBlock a = base == null ? (add != null ? new MatrixBlock(add.getNumberOfValues(nColRight), nColLeft,
			true) : new MatrixBlock(1, nColLeft, true)) : base.getMatrixBlock();
		MatrixBlock b = add == null ? new MatrixBlock(a.getNumRows(), nColRight, true) : add.getMatrixBlock();
		MatrixBlock ret = a.append(b, null, true);
		return MatrixBlockDictionary.create(ret, true);
	}
}
