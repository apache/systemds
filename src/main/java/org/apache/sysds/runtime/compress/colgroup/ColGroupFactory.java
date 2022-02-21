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
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.bitmap.BitmapEncoder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.insertionsort.AInsertionSorter;
import org.apache.sysds.runtime.compress.colgroup.insertionsort.InsertionSorterFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
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
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.DataConverter;

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

	private ColGroupFactory(MatrixBlock in, CompressedSizeInfo csi, CompressionSettings cs, ACostEstimate ce, int k) {
		this.in = in;
		this.csi = csi;
		this.cs = cs;
		this.k = k;
		// this.k = 1;
		this.ce = ce;

		this.nRow = cs.transposed ? in.getNumColumns() : in.getNumRows();
		this.nCol = cs.transposed ? in.getNumRows() : in.getNumColumns();

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
	public static List<AColGroup> compressColGroups(MatrixBlock in, CompressedSizeInfo csi, CompressionSettings ce,
		int k) {
		return new ColGroupFactory(in, csi, ce, null, k).compress();
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
	 * @return
	 */
	public static List<AColGroup> compressColGroups(MatrixBlock in, CompressedSizeInfo csi, CompressionSettings cs,
		ACostEstimate ce, int k) {
		return new ColGroupFactory(in, csi, cs, ce, k).compress();
	}

	private List<AColGroup> compress() {
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
			ExecutorService pool = CommonThreadPool.get(k);
			List<CompressTask> tasks = new ArrayList<>();

			List<List<CompressedSizeInfoColGroup>> threadGroups = makeGroups();
			for(List<CompressedSizeInfoColGroup> tg : threadGroups)
				if(!tg.isEmpty())
					tasks.add(new CompressTask(tg));

			List<AColGroup> ret = new ArrayList<>();
			for(Future<Collection<AColGroup>> t : pool.invokeAll(tasks))
				ret.addAll(t.get());
			pool.shutdown();
			return ret;
		}
		catch(InterruptedException | ExecutionException e) {
			throw new DMLRuntimeException("Failed compression ", e);
		}
	}

	private List<List<CompressedSizeInfoColGroup>> makeGroups() {
		// sort by number of distinct items
		final List<CompressedSizeInfoColGroup> groups = csi.getInfo();
		Collections.sort(groups, Comparator.comparing(g -> -g.getNumVals()));
		List<List<CompressedSizeInfoColGroup>> ret = new ArrayList<>();
		for(int i = 0; i < k; i++)
			ret.add(new ArrayList<>());

		for(int i = 0; i < groups.size(); i++)
			ret.get(i % k).add(groups.get(i));

		return ret;
	}

	protected AColGroup compressColGroup(CompressedSizeInfoColGroup cg) {
		if(LOG.isDebugEnabled() && nCol < 1000 && ce != null) {
			final Timing time = new Timing(true);
			final AColGroup ret = compressColGroupForced(cg);
			synchronized(this) {
				LOG.debug(
					String.format("time[ms]: %10.2f %20s %s cols:%s wanted:%s", time.stop(), getColumnTypesString(ret),
						getEstimateVsActualSize(ret, cg), Arrays.toString(cg.getColumns()), cg.getBestCompressionType()));
			}
			return ret;
		}
		return compressColGroupForced(cg);
	}

	private String getColumnTypesString(AColGroup ret) {
		return ret.getClass().getSimpleName().toString();
	}

	private String getEstimateVsActualSize(AColGroup ret, CompressedSizeInfoColGroup cg) {
		Level before = Logger.getLogger(ACostEstimate.class.getName()).getLevel();
		Logger.getLogger(ACostEstimate.class.getName()).setLevel(Level.TRACE);
		final double est = ce.getCost(cg);
		final double act = ce.getCost(ret, nRow);
		Logger.getLogger(ACostEstimate.class.getName()).setLevel(before);
		return String.format("[B] %10.0f -- %10.0f", est, act);
	}

	private AColGroup compressColGroupForced(CompressedSizeInfoColGroup cg) {
		final int[] colIndexes = cg.getColumns();
		final int nrUniqueEstimate = cg.getNumVals();
		CompressionType ct = cg.getBestCompressionType();

		if(ct == CompressionType.EMPTY && !cs.transposed)
			return new ColGroupEmpty(colIndexes);
		else if(ct == CompressionType.UNCOMPRESSED) // don't construct mapping if uncompressed
			return ColGroupUncompressed.create(colIndexes, in, cs.transposed);
		else if((ct == CompressionType.SDC || ct == CompressionType.CONST) && in.isInSparseFormat() && cs.transposed &&
			((colIndexes.length > 1 && in.getSparsity() < 0.001) || colIndexes.length == 1))
			// Leverage the Sparse matrix, to construct SDC group
			return compressSDCFromSparseTransposedBlock(colIndexes, nrUniqueEstimate, cg.getTupleSparsity());
		else if(colIndexes.length > 1 && ct == CompressionType.DDC)
			return directCompressDDC(colIndexes, cg);
		else if(ct == CompressionType.DeltaDDC) {
			if(colIndexes.length > 1)
				return directCompressDeltaDDC(colIndexes, in, cs, cg, k);
			else
				return compressDeltaDDC(colIndexes, in, cs, cg);
		}
		else {
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

	private static AColGroup directCompressDeltaDDC(int[] colIndexes, MatrixBlock raw, CompressionSettings cs,
		CompressedSizeInfoColGroup cg, int k) {
		final int rlen = cs.transposed ? raw.getNumColumns() : raw.getNumRows();
		// use a Map that is at least char size.
		final AMapToData d = MapToFactory.create(rlen, MAP_TYPE.INT);
		if(cs.transposed) {
			LOG.warn("In-effecient transpose back of the input matrix to do delta encoding");
			raw = LibMatrixReorg.transposeInPlace(raw, k);
			cs.transposed = false;
		}
		// Delta encode the raw data
		raw = deltaEncodeMatrixBlock(raw);
		return directCompressDDCDeltaColGroup(colIndexes, raw, cs, cg, d, rlen, k);
	}

	private AColGroup directCompressDDC(int[] colIndexes, CompressedSizeInfoColGroup cg) {
		final AMapToData d = MapToFactory.create(nRow, MAP_TYPE.INT);
		final int fill = d.getUpperBoundValue();
		d.fill(fill);

		final DblArrayCountHashMap map = new DblArrayCountHashMap(cg.getNumVals(), colIndexes.length);
		boolean extra;
		if(nRow < CompressionSettings.PAR_DDC_THRESHOLD || k == 1)
			extra = readToMapDDC(colIndexes, in, map, cs, d, 0, nRow, fill);
		else
			extra = parallelReadToMapDDC(colIndexes, in, map, cs, d, nRow, fill, k);

		if(map.size() == 0)
			// If the column was empty.
			// This is highly unlikely but could happen if forced compression of
			// not transposed column and the estimator says use DDC.
			return new ColGroupEmpty(colIndexes);

		ADictionary dict = DictionaryFactory.create(map, colIndexes.length, extra, cg.getTupleSparsity());
		if(dict == null)
			// Again highly unlikely but possible.
			return new ColGroupEmpty(colIndexes);

		if(extra)
			d.replace(fill, map.size());

		final int nUnique = map.size() + (extra ? 1 : 0);
		final AMapToData resData = MapToFactory.resize(d, nUnique);
		return ColGroupDDC.create(colIndexes, nRow, dict, resData, null);
	}

	private static AColGroup directCompressDDCDeltaColGroup(int[] colIndexes, MatrixBlock raw, CompressionSettings cs,
		CompressedSizeInfoColGroup cg, AMapToData data, int rlen, int k) {
		final int fill = data.getUpperBoundValue();
		data.fill(fill);

		DblArrayCountHashMap map = new DblArrayCountHashMap(cg.getNumVals(), colIndexes.length);
		boolean extra;
		if(rlen < CompressionSettings.PAR_DDC_THRESHOLD || k == 1)
			extra = readToMapDDC(colIndexes, raw, map, cs, data, 0, rlen, fill);
		else
			extra = parallelReadToMapDDC(colIndexes, raw, map, cs, data, rlen, fill, k);

		if(map.size() == 0)
			// If the column was empty.
			// This is highly unlikely but could happen if forced compression of
			// not transposed column and the estimator says use DDC.
			return new ColGroupEmpty(colIndexes);
		ADictionary dict = DictionaryFactory.createDelta(map, colIndexes.length, extra);
		if(extra) {
			data.replace(fill, map.size());
			data.setUnique(map.size() + 1);
		}
		else
			data.setUnique(map.size());

		AMapToData resData = MapToFactory.resize(data, map.size() + (extra ? 1 : 0));
		return ColGroupDeltaDDC.create(colIndexes, rlen, dict, resData, null);
	}

	private static boolean readToMapDDC(final int[] colIndexes, final MatrixBlock raw, final DblArrayCountHashMap map,
		final CompressionSettings cs, final AMapToData data, final int rl, final int ru, final int fill) {
		ReaderColumnSelection reader = ReaderColumnSelection.createReader(raw, colIndexes, cs.transposed, rl, ru);
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

	private static boolean parallelReadToMapDDC(final int[] colIndexes, final MatrixBlock raw,
		final DblArrayCountHashMap map, final CompressionSettings cs, final AMapToData data, final int rlen,
		final int fill, final int k) {

		try {
			final int blk = Math.max(rlen / colIndexes.length / k, 64000 / colIndexes.length);
			ExecutorService pool = CommonThreadPool.get(Math.min(Math.max(rlen / blk, 1), k));
			List<readToMapDDCTask> tasks = new ArrayList<>();

			for(int i = 0; i < rlen; i += blk) {
				int end = Math.min(rlen, i + blk);
				tasks.add(new readToMapDDCTask(colIndexes, raw, map, cs, data, i, end, fill));
			}
			boolean extra = false;
			for(Future<Boolean> t : pool.invokeAll(tasks))
				extra |= t.get();

			pool.shutdown();
			return extra;
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Failed to parallelize DDC compression");
		}
	}

	private static MatrixBlock deltaEncodeMatrixBlock(MatrixBlock mb) {
		LOG.warn("Delta encoding entire matrix input!!");
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		double[][] ret = new double[rows][cols];
		double[] a = mb.getDenseBlockValues();
		for(int i = 0, ix = 0; i < rows; i++) {
			int prevRowOff = i > 0 ? ix - cols : 0;
			for(int j = 0; j < cols; j++, ix++) {
				double currentValue = a[ix];
				ret[i][j] = i > 0 ? currentValue - a[prevRowOff + j] : currentValue;
			}
		}
		return DataConverter.convertToMatrixBlock(ret);
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
		else {
			double[] defaultTuple = new double[colIndexes.length];
			ADictionary dict = DictionaryFactory.create(ubm, largestIndex, defaultTuple, tupleSparsity, numZeros > 0);
			return compressSDCNormal(colIndexes, numZeros, rlen, ubm, largestIndex, dict, defaultTuple, cs);
		}
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
		ADictionary dict, double[] defaultTuple, CompressionSettings cs) {
		IntArrayList[] offsets = ubm.getOffsetList();
		AInsertionSorter s = InsertionSorterFactory.createNegative(rlen, offsets, largestIndex, cs.sdcSortType);
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

	private static AColGroup compressDDC(int[] colIndexes, int rlen, ABitmap ubm, CompressionSettings cs,
		double tupleSparsity) {
		boolean zeros = ubm.getNumOffsets() < rlen;
		ADictionary dict = DictionaryFactory.create(ubm, tupleSparsity, zeros);
		AMapToData data = MapToFactory.create(rlen, zeros, ubm.getOffsetList());
		return ColGroupDDC.create(colIndexes, rlen, dict, data, null);
	}

	private static AColGroup compressDeltaDDC(int[] colIndexes, MatrixBlock in, CompressionSettings cs,
		CompressedSizeInfoColGroup cg) {

		LOG.warn("Multi column Delta encoding only supported if delta encoding is only compression");
		if(cs.transposed) {
			LibMatrixReorg.transposeInPlace(in, 1);
			cs.transposed = false;
		}
		// Delta encode the raw data
		in = deltaEncodeMatrixBlock(in);

		final int rlen = in.getNumRows();
		// TODO Add extractBitMap that is delta to not require delta encoding entire input matrix.
		final ABitmap ubm = BitmapEncoder.extractBitmap(colIndexes, in, cs.transposed, cg.getNumVals(),
			cs.sortTuplesByFrequency);
		boolean zeros = ubm.getNumOffsets() < rlen;
		ADictionary dict = DictionaryFactory.create(ubm, cg.getTupleSparsity(), zeros);
		AMapToData data = MapToFactory.create(rlen, zeros, ubm.getOffsetList());
		return ColGroupDeltaDDC.create(colIndexes, rlen, dict, data, null);
	}

	private static AColGroup compressOLE(int[] colIndexes, int rlen, ABitmap ubm, CompressionSettings cs,
		double tupleSparsity) {

		ADictionary dict = DictionaryFactory.create(ubm, tupleSparsity);
		ColGroupOLE ole = new ColGroupOLE(rlen);

		final int numVals = ubm.getNumValues();
		char[][] lbitmaps = new char[numVals][];
		int totalLen = 0;
		for(int i = 0; i < numVals; i++) {
			lbitmaps[i] = ColGroupOLE.genOffsetBitmap(ubm.getOffsetsList(i).extractValues(), ubm.getNumOffsets(i));
			totalLen += lbitmaps[i].length;
		}

		// compact bitmaps to linearized representation
		ole.createCompressedBitmaps(numVals, totalLen, lbitmaps);
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
		char[][] lbitmaps = new char[numVals][];
		int totalLen = 0;

		for(int k = 0; k < numVals; k++) {
			lbitmaps[k] = ColGroupRLE.genRLEBitmap(ubm.getOffsetsList(k).extractValues(), ubm.getNumOffsets(k));
			totalLen += lbitmaps[k].length;
		}
		// compact bitmaps to linearized representation
		rle.createCompressedBitmaps(numVals, totalLen, lbitmaps);
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
		else {
			final ABitmap ubm = BitmapEncoder.extractBitmap(cols, in, true, entries.length, true);
			// zero is not the default value fall back to the standard compression path.
			return compressSDC(cols, nRow, ubm, cs, 1.0);
		}
	}

	class CompressTask implements Callable<Collection<AColGroup>> {

		private final List<CompressedSizeInfoColGroup> _groups;

		protected CompressTask(List<CompressedSizeInfoColGroup> groups) {
			_groups = groups;
		}

		@Override
		public Collection<AColGroup> call() {
			try {
				ArrayList<AColGroup> res = new ArrayList<>(_groups.size());
				for(CompressedSizeInfoColGroup g : _groups)
					res.add(compressColGroup(g));
				return res;
			}
			catch(Exception e) {
				e.printStackTrace();
				throw e;
			}
		}
	}

	static class readToMapDDCTask implements Callable<Boolean> {
		private final int[] _colIndexes;
		private final MatrixBlock _raw;
		private final DblArrayCountHashMap _map;
		private final CompressionSettings _cs;
		private final AMapToData _data;
		private final int _rl;
		private final int _ru;
		private final int _fill;

		protected readToMapDDCTask(int[] colIndexes, MatrixBlock raw, DblArrayCountHashMap map, CompressionSettings cs,
			AMapToData data, int rl, int ru, int fill) {
			_colIndexes = colIndexes;
			_raw = raw;
			_map = map;
			_cs = cs;
			_data = data;
			_rl = rl;
			_ru = ru;
			_fill = fill;
		}

		@Override
		public Boolean call() {
			return Boolean.valueOf(readToMapDDC(_colIndexes, _raw, _map, _cs, _data, _rl, _ru, _fill));
		}
	}

	/**
	 * Temp reuse object, to contain intermediates for compressing column groups that can be used by the same thread
	 * again for subsequent compressions.
	 */
	static class Tmp {
		private DoubleCountHashMap dblCountMap;

		protected Tmp() {
			dblCountMap = null;
		}

		protected DoubleCountHashMap getDblCountMap(int size) {
			if(dblCountMap != null)
				dblCountMap.reset(size);
			else
				dblCountMap = new DoubleCountHashMap(size);
			return dblCountMap;
		}
	}
}
