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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.bitmap.BitmapEncoder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.functional.LinearRegression;
import org.apache.sysds.runtime.compress.colgroup.insertionsort.AInsertionSorter;
import org.apache.sysds.runtime.compress.colgroup.insertionsort.InsertionSorterFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.cost.ACostEstimate;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.utils.DCounts;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayCountHashMap;
import org.apache.sysds.runtime.compress.utils.DoubleCountHashMap;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * Factory class for constructing ColGroups.
 */
public class ColGroupFactory {
	static final Log LOG = LogFactory.getLog(ColGroupFactory.class.getName());

	private final MatrixBlock in;
	private final CompressedSizeInfo csi;
	private final CompressionSettings cs;
	private final ACostEstimate ce;
	private final int k;

	private final int nRow;
	private final int nCol;

	private final ExecutorService pool;

	private ColGroupFactory(MatrixBlock in, CompressedSizeInfo csi, CompressionSettings cs, ACostEstimate ce, int k) {
		this.in = in;
		this.csi = csi;
		this.cs = cs;
		this.k = k;
		this.ce = ce;

		this.nRow = cs.transposed ? in.getNumColumns() : in.getNumRows();
		this.nCol = cs.transposed ? in.getNumRows() : in.getNumColumns();

		this.pool = (k > 1) ? CommonThreadPool.get(k) : null;
	}

	/**
	 * The actual compression method, that handles the logic of compressing multiple columns together.
	 * 
	 * @param in  The input matrix, that could have been transposed. If it is transposed the compSettings should specify
	 *            this.
	 * @param csi The compression information extracted from the estimation, this contains which groups of columns to
	 *            compress together.
	 * @param cs  The compression settings to specify how to compress.
	 * @param k   The degree of parallelism to be used in the compression of the column groups.
	 * @return A resulting array of ColGroups, containing the compressed information from the input matrix block.
	 */
	public static List<AColGroup> compressColGroups(MatrixBlock in, CompressedSizeInfo csi, CompressionSettings cs,
		int k) {
		return new ColGroupFactory(in, csi, cs, null, k).compress();
	}

	/**
	 * 
	 * @param in  The input matrix, that could have been transposed. If it is transposed the compSettings should specify
	 *            this.
	 * @param csi The compression information extracted from the estimation, this contains which groups of columns to
	 *            compress together.
	 * @param cs  The compression settings to specify how to compress.
	 * @param ce  The cost estimator used for the compression
	 * @param k   The degree of parallelism to be used in the compression of the column groups.
	 * @return A resulting array of ColGroups, containing the compressed information from the input matrix block.
	 */
	public static List<AColGroup> compressColGroups(MatrixBlock in, CompressedSizeInfo csi, CompressionSettings cs,
		ACostEstimate ce, int k) {
		return new ColGroupFactory(in, csi, cs, ce, k).compress();
	}

	private List<AColGroup> compress() {
		try {
			List<AColGroup> ret = compressExecute();
			if(pool != null)
				pool.shutdown();
			return ret;
		}
		catch(Exception e) {
			if(pool != null)
				pool.shutdown();
			throw new DMLCompressionException("Compression Failed", e);
		}
	}

	private List<AColGroup> compressExecute() {
		for(CompressedSizeInfoColGroup g : csi.getInfo())
			g.clearMap();
		if(in.isEmpty()) {
			AColGroup empty = ColGroupEmpty.create(cs.transposed ? in.getNumRows() : in.getNumColumns());
			return Collections.singletonList(empty);
		}
		else if(k <= 1)
			return compressColGroupsSingleThreaded();
		else
			return compressColGroupsParallel();
	}

	private List<AColGroup> compressColGroupsSingleThreaded() {
		List<AColGroup> ret = new ArrayList<>(csi.getNumberColGroups());
		List<CompressedSizeInfoColGroup> groups = csi.getInfo();

		for(CompressedSizeInfoColGroup g : groups)
			ret.add(compressColGroup(g));

		return ret;
	}

	private List<AColGroup> compressColGroupsParallel() {
		try {
			final List<CompressedSizeInfoColGroup> groups = csi.getInfo();
			final int nGroups = groups.size();
			// final int blkz = nGroups * 10 / k;
			final int skip = Math.min(k * 10, nGroups);
			final List<CompressTask> tasks = new ArrayList<>(skip);

			// sort to make the "assumed" big jobs first.
			Collections.sort(groups, Comparator.comparing(g -> -g.getNumVals()));

			final AColGroup[] ret = new AColGroup[nGroups];

			for(int i = 0; i < skip; i++)
				tasks.add(new CompressTask(groups, ret, i, skip));

			for(Future<Object> t : pool.invokeAll(tasks))
				t.get();

			return Arrays.asList(ret);
		}
		catch(InterruptedException | ExecutionException e) {
			throw new DMLRuntimeException("Failed compression ", e);
		}
	}

	protected AColGroup compressColGroup(CompressedSizeInfoColGroup cg) {
		if(LOG.isDebugEnabled() && nCol < 1000 && ce != null) {
			final Timing time = new Timing(true);
			final AColGroup ret = compressColGroupAllSteps(cg);
			logEstVsActual(time.stop(), ret, cg);
			return ret;
		}
		return compressColGroupAllSteps(cg);
	}

	private void logEstVsActual(double time, AColGroup act, CompressedSizeInfoColGroup est) {
		final double estC = ce.getCost(est);
		final double actC = ce.getCost(act, nRow);
		final String retType = act.getClass().getSimpleName().toString();
		final String cols = Arrays.toString(est.getColumns());
		final String wanted = est.getBestCompressionType().toString();
		if(estC < actC * 0.75) {
			StringBuilder sb = new StringBuilder();
			sb.append("The estimate cost is significantly off : distinct: ");
			sb.append(est.getNumVals());
			sb.append(" ");
			sb.append(act.getNumValues());
			sb.append(" estimate offsets:");
			sb.append(est.getNumOffs());
			if(act instanceof ColGroupSDCZeros)
				sb.append("  act:" + ((ColGroupSDCZeros) act).getIndexesSize());
			String warning = sb.toString();

			LOG.debug(String.format("time[ms]: %10.2f %25s est %10.0f -- act %10.0f cols:%s wanted:%s\n%s", time, retType,
				estC, actC, cols, wanted, warning));
		}
		else {
			LOG.debug(String.format("time[ms]: %10.2f %25s est %10.0f -- act %10.0f cols:%s wanted:%s", time, retType,
				estC, actC, cols, wanted));
		}

	}

	private AColGroup compressColGroupAllSteps(CompressedSizeInfoColGroup cg) {
		AColGroup g = compressColGroupInitial(cg);
		final int nCol = g.getColIndices().length;
		if(ce != null && ce.shouldSparsify() && nCol >= 4 && isSparsifyingColGroup(g)) {

			double[] constV = null;
			if(g instanceof ColGroupSDC) {
				constV = ((ColGroupSDC) g)._defaultTuple;
				g = ((ColGroupSDC) g).subtractDefaultTuple();
			}

			final AColGroupValue clg = (AColGroupValue) g;
			final int nVal = g.getNumValues();
			final MatrixBlockDictionary mbd = clg._dict.getMBDict(nCol);
			final MatrixBlock mb = mbd.getMatrixBlock();

			if(mb == null || mb.isEmpty())
				return g;

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
			contains = false;
			for(int i = 0; i < nCol; i++)
				if(ref[i] != 0) {
					contains = true;
					break;
				}

			if(contains) {
				// minus overlap on dictionary
				MatrixBlockDictionary mDict = mbd.binOpRight(new BinaryOperator(Minus.getMinusFnObject()), ref);
				if(constV != null)
					for(int i = 0; i < nCol; i++)
						ref[i] += constV[i]; // plus reference on overlap

				LOG.debug(
					String.format("Sparsifying colgroup before %1.4f now %1.4f", mb.getSparsity(), mDict.getSparsity()));
				if(g instanceof ColGroupDDC)
					g = ColGroupDDCFOR.create(g.getColIndices(), nRow, mDict, ((ColGroupDDC) clg)._data,
						clg.getCachedCounts(), ref);
				else if(g instanceof ColGroupSDCZeros) {
					g = ColGroupSDCFOR.create(g.getColIndices(), nRow, mDict, ((ColGroupSDCZeros) clg)._indexes,
						((ColGroupSDCZeros) clg)._data, clg.getCachedCounts(), ref);

				}
			}
			else {
				if(g instanceof ColGroupSDCZeros)
					g = ColGroupSDCFOR.create(g.getColIndices(), nRow, mbd, ((ColGroupSDCZeros) clg)._indexes,
						((ColGroupSDCZeros) clg)._data, clg.getCachedCounts(), ref);
			}

		}
		return g;
	}

	private void getMostCommonValues(MatrixBlock mb, double[] ref, int[] nnzCols) {
		// take each column marked by ref and find most common value in that and assign it to ref.
		// if the columns are

		DoubleCountHashMap[] counters = new DoubleCountHashMap[ref.length];

		if(mb.isInSparseFormat()) {
			// initialize the counters with zero count.
			for(int i = 0; i < ref.length; i++) {
				if(ref[i] != 0) {
					counters[i] = new DoubleCountHashMap(8);
					counters[i].increment(0, nnzCols[i]);
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

	private boolean isSparsifyingColGroup(AColGroup g) {
		return g instanceof ColGroupDDC || g instanceof ColGroupSDC;
	}

	private AColGroup compressColGroupInitial(CompressedSizeInfoColGroup cg) {
		final int[] colIndexes = cg.getColumns();
		final int nrUniqueEstimate = cg.getNumVals();
		CompressionType ct = cg.getBestCompressionType();

		if(ct == CompressionType.EMPTY && !cs.transposed)
			return new ColGroupEmpty(colIndexes);
		else if(ct == CompressionType.UNCOMPRESSED) // don't construct mapping if uncompressed
			return ColGroupUncompressed.create(colIndexes, in, cs.transposed);
		else if((ct == CompressionType.SDC || ct == CompressionType.CONST) && in.isInSparseFormat() && cs.transposed &&
			((colIndexes.length > 1 && cg.getNumOffs() < 0.3 * nRow) || colIndexes.length == 1))
			return compressSDCFromSparseTransposedBlock(colIndexes, nrUniqueEstimate, cg.getTupleSparsity());
		else if(ct == CompressionType.DDC)
			return directCompressDDC(colIndexes, cg);
		else if(ct == CompressionType.LinearFunctional)
			return compressLinearFunctional(colIndexes, in, cs);
		else {
			LOG.debug("Default slow path: " + ct + "  " + cs.transposed + " " + Arrays.toString(colIndexes));
			final int numRows = cs.transposed ? in.getNumColumns() : in.getNumRows();
			final ABitmap ubm = BitmapEncoder.extractBitmap(colIndexes, in, cs.transposed, nrUniqueEstimate,
				cs.sortTuplesByFrequency);
			return compress(colIndexes, numRows, ubm, ct, cs, cg.getTupleSparsity());
		}
	}

	private static AColGroup compress(int[] colIndexes, int rlen, ABitmap ubm, CompressionType compType,
		CompressionSettings cs, double tupleSparsity) {

		if(ubm == null)
			// If ubm is null then there was no values to extract
			// Therefore compress to empty column group
			return new ColGroupEmpty(colIndexes);

		final IntArrayList[] of = ubm.getOffsetList();
		if(of.length == 1 && of[0].size() == rlen) // If this always constant
			return ColGroupConst.create(colIndexes, DictionaryFactory.create(ubm));

		// only consider sparse dictionaries if cocoded more than 4 columns.
		tupleSparsity = colIndexes.length > 4 ? tupleSparsity : 1.0;
		switch(compType) {
			case DDC:
				return compressDDC(colIndexes, rlen, ubm, cs, tupleSparsity);
			case RLE:
				return compressRLE(colIndexes, rlen, ubm, cs, tupleSparsity);
			case OLE:
				return compressOLE(colIndexes, rlen, ubm, cs, tupleSparsity);
			case CONST: // in case somehow one requested const, but it was not const fall back to SDC.
				LOG.warn("Requested const on non constant column, fallback to SDC");
			case EMPTY:
			case SDC:
				return compressSDC(colIndexes, rlen, ubm, cs, tupleSparsity);
			default:
				throw new DMLCompressionException("Not implemented compression of " + compType + " in factory.");
		}
	}

	private AColGroup directCompressDDC(int[] colIndexes, CompressedSizeInfoColGroup cg) {
		if(colIndexes.length > 1)
			return directCompressDDCMultiCol(colIndexes, cg);
		else
			return directCompressDDCSingleCol(colIndexes, cg);
	}

	private AColGroup directCompressDDCSingleCol(int[] colIndexes, CompressedSizeInfoColGroup cg) {
		final int col = colIndexes[0];
		final AMapToData d = MapToFactory.create(nRow, Math.max(Math.min(cg.getNumOffs() + 1, nRow), 126));
		final DoubleCountHashMap map = new DoubleCountHashMap(cg.getNumVals());

		// unlike multi-col no special handling of zero entries are needed.
		if(cs.transposed)
			readToMapDDCTransposed(col, map, d);
		else
			readToMapDDC(col, map, d);

		ADictionary dict = DictionaryFactory.create(map);
		final int nUnique = map.size();
		final AMapToData resData = MapToFactory.resize(d, nUnique);
		return ColGroupDDC.create(colIndexes, nRow, dict, resData, null);
	}

	private AColGroup directCompressDDCMultiCol(int[] colIndexes, CompressedSizeInfoColGroup cg) {
		final AMapToData d = MapToFactory.create(nRow, Math.max(Math.min(cg.getNumOffs() + 1, nRow), 126));
		final int fill = d.getUpperBoundValue();
		d.fill(fill);

		final DblArrayCountHashMap map = new DblArrayCountHashMap(cg.getNumVals(), colIndexes.length);
		boolean extra;
		if(nRow < CompressionSettings.PAR_DDC_THRESHOLD || k == 1)
			extra = readToMapDDC(colIndexes, map, d, 0, nRow, fill);
		else
			extra = parallelReadToMapDDC(colIndexes, map, d, nRow, fill, k);

		if(map.size() == 0)
			// If the column was empty.
			// This is highly unlikely but could happen if forced compression of
			// not transposed column and the estimator says use DDC.
			return new ColGroupEmpty(colIndexes);

		ADictionary dict = DictionaryFactory.create(map, colIndexes.length, extra, cg.getTupleSparsity());
		if(dict == null)
			// Again highly unlikely but possible.
			return new ColGroupEmpty(colIndexes);
		try {
			if(extra)
				d.replace(fill, map.size());

			final int nUnique = map.size() + (extra ? 1 : 0);

			final AMapToData resData = MapToFactory.resize(d, nUnique);
			return ColGroupDDC.create(colIndexes, nRow, dict, resData, null);

		}
		catch(Exception e) {
			ReaderColumnSelection reader = ReaderColumnSelection.createReader(in, colIndexes, cs.transposed, 0, nRow);
			throw new DMLCompressionException("direct compress DDC Multi col failed extra:" + extra + " with reader type:"
				+ reader.getClass().getSimpleName(), e);
		}
	}

	private boolean readToMapDDC(int[] colIndexes, DblArrayCountHashMap map, AMapToData data, int rl, int ru, int fill) {
		ReaderColumnSelection reader = ReaderColumnSelection.createReader(in, colIndexes, cs.transposed, rl, ru);
		DblArray cellVals = reader.nextRow();
		boolean extra = false;
		int r = rl;
		while(r < ru && cellVals != null) {
			final int row = reader.getCurrentRowIndex();
			if(row == r) {
				final int id = map.increment(cellVals);
				data.set(row, id);
				cellVals = reader.nextRow();
				r++;
			}
			else {
				r = row;
				extra = true;
			}
		}

		if(r < ru)
			extra = true;

		return extra;
	}

	private void readToMapDDC(int col, DoubleCountHashMap map, AMapToData data) {
		if(in.isInSparseFormat()) {
			// not good but could happen
			final SparseBlock sb = in.getSparseBlock();
			for(int r = 0; r < nRow; r++) {
				if(sb.isEmpty(r))
					data.set(r, map.increment(0));
				else {
					final int apos = sb.pos(r);
					final int alen = sb.size(r) + apos;
					final int[] aix = sb.indexes(r);
					final int idx = Arrays.binarySearch(aix, apos, alen, col);
					if(idx < 0)
						data.set(r, map.increment(0));
					else
						data.set(r, map.increment(sb.values(r)[idx]));
				}
			}
		}
		else if(in.getDenseBlock().isContiguous()) {
			final double[] dv = in.getDenseBlockValues();
			int off = col;
			for(int r = 0; r < nRow; r++, off += nCol) {
				final int id = map.increment(dv[off]);
				data.set(r, id);
			}
		}
		else {
			throw new NotImplementedException("");
		}
	}

	private void readToMapDDCTransposed(int col, DoubleCountHashMap map, AMapToData data) {
		if(in.isInSparseFormat()) {
			// good
			SparseBlock sb = in.getSparseBlock();
			if(sb.isEmpty(col))
				return;

			final int apos = sb.pos(col);
			final int alen = sb.size(col) + apos;
			final int[] aix = sb.indexes(col);
			final double[] aval = sb.values(col);
			// count zeros
			map.increment(0, nRow - apos - alen);
			// insert all other counts
			for(int j = apos; j < alen; j++) {
				final int id = map.increment(aval[j]);
				data.set(aix[j], id);
			}
		}
		else if(in.getDenseBlock().isContiguous()) {
			double[] dv = in.getDenseBlockValues();
			int off = col * nRow;
			for(int r = 0; r < nRow; r++, off++) {
				final int id = map.increment(dv[off]);
				data.set(r, id);
			}
		}
		else {
			throw new NotImplementedException("");
		}
	}

	private boolean parallelReadToMapDDC(int[] colIndexes, DblArrayCountHashMap map, AMapToData data, int rlen, int fill,
		int k) {

		try {
			final int blk = Math.max(rlen / colIndexes.length / k, 64000 / colIndexes.length);

			List<readToMapDDCTask> tasks = new ArrayList<>();
			for(int i = 0; i < rlen; i += blk) {
				int end = Math.min(rlen, i + blk);
				tasks.add(new readToMapDDCTask(colIndexes, map, data, i, end, fill));
			}
			boolean extra = false;
			for(Future<Boolean> t : pool.invokeAll(tasks))
				extra |= t.get();

			return extra;
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Failed to parallelize DDC compression");
		}
	}

	private static AColGroup compressSDC(int[] colIndexes, int rlen, ABitmap ubm, CompressionSettings cs,
		double tupleSparsity) {

		final int numZeros = ubm.getNumZeros();
		IntArrayList[] offs = ubm.getOffsetList();
		int largestOffset = offs[0].size();
		int largestIndex = 0;
		if(!cs.sortTuplesByFrequency) {
			int index = 0;
			for(IntArrayList a : ubm.getOffsetList()) {
				if(a.size() > largestOffset) {
					largestOffset = a.size();
					largestIndex = index;
				}
				index++;
			}
		}

		// Currently not effecient allocation of the dictionary.
		if(ubm.getNumValues() == 1 && numZeros >= largestOffset) {
			ADictionary dict = DictionaryFactory.create(ubm, tupleSparsity);
			final AOffset off = OffsetFactory.createOffset(ubm.getOffsetList()[0].extractValues(true));
			return ColGroupSDCSingleZeros.create(colIndexes, rlen, dict, off, null);
		}
		else if((ubm.getNumValues() == 2 && numZeros == 0) || (ubm.getNumValues() == 1 && numZeros < largestOffset)) {
			double[] defaultTuple = new double[colIndexes.length];
			ADictionary dict = DictionaryFactory.create(ubm, largestIndex, defaultTuple, tupleSparsity, numZeros > 0);
			return compressSDCSingle(colIndexes, rlen, ubm, dict, defaultTuple);
		}
		else if(numZeros >= largestOffset) {
			ADictionary dict = DictionaryFactory.create(ubm, tupleSparsity);
			return compressSDCZero(colIndexes, rlen, ubm, dict, cs);
		}
		else
			return compressSDCNormal(colIndexes, numZeros, rlen, ubm, largestIndex, tupleSparsity, cs);

	}

	private static AColGroup compressSDCZero(int[] colIndexes, int rlen, ABitmap ubm, ADictionary dict,
		CompressionSettings cs) {
		IntArrayList[] offsets = ubm.getOffsetList();
		AInsertionSorter s = InsertionSorterFactory.create(rlen, offsets, cs.sdcSortType);
		AOffset indexes = OffsetFactory.createOffset(s.getIndexes());
		AMapToData data = s.getData();
		data = MapToFactory.resize(data, dict.getNumberOfValues(colIndexes.length));
		return ColGroupSDCZeros.create(colIndexes, rlen, dict, indexes, data, null);
	}

	private static AColGroup compressSDCNormal(int[] colIndexes, int numZeros, int rlen, ABitmap ubm, int largestIndex,
		double tupleSparsity, CompressionSettings cs) {
		final double[] defaultTuple = new double[colIndexes.length];
		final ADictionary dict = DictionaryFactory.create(ubm, largestIndex, defaultTuple, tupleSparsity, numZeros > 0);
		AInsertionSorter s = InsertionSorterFactory.createNegative(rlen, ubm.getOffsetList(), largestIndex,
			cs.sdcSortType);
		AOffset indexes = OffsetFactory.createOffset(s.getIndexes());
		AMapToData _data = s.getData();
		_data = MapToFactory.resize(_data, dict.getNumberOfValues(colIndexes.length));
		return ColGroupSDC.create(colIndexes, rlen, dict, defaultTuple, indexes, _data, null);
	}

	private static AColGroup compressSDCSingle(int[] colIndexes, int rlen, ABitmap ubm, ADictionary dict,
		double[] defaultTuple) {
		IntArrayList inv = ubm.getOffsetsList(0);
		int[] indexes = new int[rlen - inv.size()];
		int p = 0;
		int v = 0;
		for(int i = 0; i < inv.size(); i++) {
			int j = inv.get(i);
			while(v < j)
				indexes[p++] = v++;
			if(v == j)
				v++;
		}

		while(v < rlen)
			indexes[p++] = v++;
		AOffset off = OffsetFactory.createOffset(indexes);

		return ColGroupSDCSingle.create(colIndexes, rlen, dict, defaultTuple, off, null);
	}

	private static AColGroup compressLinearFunctional(int[] colIndexes, MatrixBlock in, CompressionSettings cs) {
		double[] coefficients = LinearRegression.regressMatrixBlock(in, colIndexes, cs.transposed);
		int numRows = cs.transposed ? in.getNumColumns() : in.getNumRows();
		return ColGroupLinearFunctional.create(colIndexes, coefficients, numRows);
	}

	private static AColGroup compressDDC(int[] colIndexes, int rlen, ABitmap ubm, CompressionSettings cs,
		double tupleSparsity) {
		boolean zeros = ubm.getNumOffsets() < rlen;
		ADictionary dict = DictionaryFactory.create(ubm, tupleSparsity, zeros);
		AMapToData data = MapToFactory.create(rlen, zeros, ubm.getOffsetList());
		return ColGroupDDC.create(colIndexes, rlen, dict, data, null);
	}

	private static AColGroup compressOLE(int[] colIndexes, int rlen, ABitmap ubm, CompressionSettings cs,
		double tupleSparsity) {

		ADictionary dict = DictionaryFactory.create(ubm, tupleSparsity);
		ColGroupOLE ole = new ColGroupOLE(rlen);

		final int numVals = ubm.getNumValues();
		char[][] lBitMaps = new char[numVals][];
		int totalLen = 0;
		for(int i = 0; i < numVals; i++) {
			lBitMaps[i] = ColGroupOLE.genOffsetBitmap(ubm.getOffsetsList(i).extractValues(), ubm.getNumOffsets(i));
			totalLen += lBitMaps[i].length;
		}

		// compact bitmaps to linearized representation
		ole.createCompressedBitmaps(numVals, totalLen, lBitMaps);
		ole._dict = dict;
		ole._zeros = ubm.getNumOffsets() < (long) rlen;
		ole._colIndexes = colIndexes;
		return ole;
	}

	private static AColGroup compressRLE(int[] colIndexes, int rlen, ABitmap ubm, CompressionSettings cs,
		double tupleSparsity) {

		ADictionary dict = DictionaryFactory.create(ubm, tupleSparsity);
		ColGroupRLE rle = new ColGroupRLE(rlen);
		// compress the bitmaps
		final int numVals = ubm.getNumValues();
		char[][] lBitMaps = new char[numVals][];
		int totalLen = 0;

		for(int k = 0; k < numVals; k++) {
			lBitMaps[k] = ColGroupRLE.genRLEBitmap(ubm.getOffsetsList(k).extractValues(), ubm.getNumOffsets(k));
			totalLen += lBitMaps[k].length;
		}
		// compact bitmaps to linearized representation
		rle.createCompressedBitmaps(numVals, totalLen, lBitMaps);
		rle._dict = dict;
		rle._zeros = ubm.getNumOffsets() < (long) rlen;
		rle._colIndexes = colIndexes;
		return rle;
	}

	private AColGroup compressSDCFromSparseTransposedBlock(int[] cols, int nrUniqueEstimate, double tupleSparsity) {
		if(cols.length > 1)
			return compressMultiColSDCFromSparseTransposedBlock(cols, nrUniqueEstimate, tupleSparsity);
		else
			return compressSingleColSDCFromSparseTransposedBlock(cols, nrUniqueEstimate);

	}

	private AColGroup compressMultiColSDCFromSparseTransposedBlock(int[] cols, int nrUniqueEstimate,
		double tupleSparsity) {

		HashSet<Integer> offsetsSet = new HashSet<>();

		SparseBlock sb = in.getSparseBlock();

		for(int i = 0; i < cols.length; i++) {
			if(sb.isEmpty(cols[i]))
				throw new DMLCompressionException("Empty columns should not be entering here");

			int apos = sb.pos(cols[i]);
			int alen = sb.size(cols[i]) + apos;
			int[] aix = sb.indexes(cols[i]);
			for(int j = apos; j < alen; j++)
				offsetsSet.add(aix[j]);
		}

		int[] offsetsInt = offsetsSet.stream().mapToInt(Number::intValue).toArray();
		Arrays.sort(offsetsInt);

		MatrixBlock sub = new MatrixBlock(offsetsInt.length, cols.length, false);
		sub.allocateDenseBlock();
		sub.setNonZeros(offsetsInt.length * cols.length);
		double[] subV = sub.getDenseBlockValues();

		for(int i = 0; i < cols.length; i++) {
			int apos = sb.pos(cols[i]);
			int alen = sb.size(cols[i]) + apos;
			int[] aix = sb.indexes(cols[i]);
			double[] aval = sb.values(cols[i]);
			int offsetsPos = 0;
			for(int j = apos; j < alen; j++) {
				while(offsetsInt[offsetsPos] < aix[j])
					offsetsPos++;
				if(offsetsInt[offsetsPos] == aix[j])
					subV[offsetsPos * cols.length + i] = aval[j];
			}
		}

		int[] subCols = new int[cols.length];
		for(int i = 1; i < cols.length; i++)
			subCols[i] = i;
		ReaderColumnSelection reader = ReaderColumnSelection.createReader(sub, subCols, false);
		final int mapStartSize = Math.min(nrUniqueEstimate, offsetsInt.length / 2);
		DblArrayCountHashMap map = new DblArrayCountHashMap(mapStartSize, subCols.length);

		DblArray cellVals = null;
		AMapToData data = MapToFactory.create(offsetsInt.length, 257);

		while((cellVals = reader.nextRow()) != null) {
			final int row = reader.getCurrentRowIndex();
			data.set(row, map.increment(cellVals));
		}

		ADictionary dict = DictionaryFactory.create(map, cols.length, false, tupleSparsity);
		data = MapToFactory.resize(data, map.size());

		AOffset offs = OffsetFactory.createOffset(offsetsInt);
		return ColGroupSDCZeros.create(cols, in.getNumColumns(), dict, offs, data, null);
	}

	private AColGroup compressSingleColSDCFromSparseTransposedBlock(int[] cols, int nrUniqueEstimate) {

		// This method should only be called if the cols argument is length 1.
		final SparseBlock sb = in.getSparseBlock();
		final int sbRow = cols[0];
		final int apos = sb.pos(sbRow);
		final int alen = sb.size(sbRow) + apos;
		final double[] vals = sb.values(sbRow);
		final DoubleCountHashMap map = new DoubleCountHashMap(nrUniqueEstimate);

		// count distinct items frequencies
		for(int j = apos; j < alen; j++)
			map.increment(vals[j]);

		DCounts[] entries = map.extractValues();
		Arrays.sort(entries, Comparator.comparing(x -> -x.count));

		if(entries[0].count < nRow - sb.size(sbRow)) {
			// If the zero is the default value.
			final int[] counts = new int[entries.length];
			final double[] dict = new double[entries.length];
			for(int i = 0; i < entries.length; i++) {
				final DCounts x = entries[i];
				counts[i] = x.count;
				dict[i] = x.key;
				x.count = i;
			}

			final AOffset offsets = OffsetFactory.createOffset(sb.indexes(sbRow), apos, alen);
			if(entries.length <= 1)
				return ColGroupSDCSingleZeros.create(cols, nRow, new Dictionary(dict), offsets, counts);
			else {
				final AMapToData mapToData = MapToFactory.create((alen - apos), entries.length);
				for(int j = apos; j < alen; j++)
					mapToData.set(j - apos, map.get(vals[j]));
				return ColGroupSDCZeros.create(cols, nRow, new Dictionary(dict), offsets, mapToData, counts);
			}
		}
		else if(entries.length == 1) {
			// SDCSingle and we know all values are x or 0
			final int nonZeros = nRow - entries[0].count;
			final double x = entries[0].key;
			final double[] defaultTuple = new double[] {x};
			final ADictionary zeroDict = new Dictionary(new double[] {0});
			final int[] counts = new int[] {nonZeros};
			final int[] notZeroOffsets = new int[nonZeros];
			final int[] aix = sb.indexes(sbRow);
			int i = 0;
			int r = 0;
			for(int j = apos; r < aix[alen - 1]; r++) {
				if(r == aix[j])
					j++;
				else
					notZeroOffsets[i++] = r;
			}
			r++;

			for(; r < nRow; r++, i++)
				notZeroOffsets[i] = r;

			final AOffset offsets = OffsetFactory.createOffset(notZeroOffsets);

			return ColGroupSDCSingle.create(cols, nRow, zeroDict, defaultTuple, offsets, counts);
		}
		else {
			final ABitmap ubm = BitmapEncoder.extractBitmap(cols, in, true, entries.length, true);
			// zero is not the default value fall back to the standard compression path.
			return compressSDC(cols, nRow, ubm, cs, 1.0);
		}
	}

	private class CompressTask implements Callable<Object> {

		private final List<CompressedSizeInfoColGroup> _groups;
		private final AColGroup[] _ret;
		private final int _off;
		private final int _step;

		protected CompressTask(List<CompressedSizeInfoColGroup> groups, AColGroup[] ret, int off, int step) {
			_groups = groups;
			_ret = ret;
			_off = off;
			_step = step;
		}

		@Override
		public Object call() {
			try {
				for(int i = _off; i < _groups.size(); i += _step)
					_ret[i] = compressColGroup(_groups.get(i));
				return null;
			}
			catch(Exception e) {
				e.printStackTrace();
				throw e;
			}
		}
	}

	private class readToMapDDCTask implements Callable<Boolean> {
		private final int[] _colIndexes;
		private final DblArrayCountHashMap _map;
		private final AMapToData _data;
		private final int _rl;
		private final int _ru;
		private final int _fill;

		protected readToMapDDCTask(int[] colIndexes, DblArrayCountHashMap map, AMapToData data, int rl, int ru,
			int fill) {
			_colIndexes = colIndexes;
			_map = map;
			_data = data;
			_rl = rl;
			_ru = ru;
			_fill = fill;
		}

		@Override
		public Boolean call() {
			return Boolean.valueOf(readToMapDDC(_colIndexes, _map, _data, _rl, _ru, _fill));
		}
	}
}
