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
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

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
import org.apache.sysds.runtime.compress.colgroup.insertionsort.AInsertionSorter;
import org.apache.sysds.runtime.compress.colgroup.insertionsort.InsertionSorterFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.utils.DCounts;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayCountHashMap;
import org.apache.sysds.runtime.compress.utils.DoubleCountHashMap;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * Factory class for constructing ColGroups.
 */
public class ColGroupFactory {
	static final Log LOG = LogFactory.getLog(ColGroupFactory.class.getName());

	/**
	 * The actual compression method, that handles the logic of compressing multiple columns together.
	 * 
	 * @param in           The input matrix, that could have been transposed. If it is transposed the compSettings should
	 *                     specify this.
	 * @param csi          The compression information extracted from the estimation, this contains which groups of
	 *                     columns to compress together.
	 * @param compSettings The compression settings to specify how to compress.
	 * @param k            The degree of parallelism to be used in the compression of the column groups.
	 * @return A resulting array of ColGroups, containing the compressed information from the input matrix block.
	 */
	public static List<AColGroup> compressColGroups(MatrixBlock in, CompressedSizeInfo csi,
		CompressionSettings compSettings, int k) {
		for(CompressedSizeInfoColGroup g : csi.getInfo())
			g.clearMap();

		if(in.isEmpty())
			return genEmpty(in, compSettings);
		else if(k <= 1)
			return compressColGroupsSingleThreaded(in, csi, compSettings);
		else
			return compressColGroupsParallel(in, csi, compSettings, k);
	}

	/**
	 * Generate a constant column group.
	 * 
	 * @param numCols The number of columns
	 * @param value   The value contained in all cells.
	 * @return A Constant column group.
	 */
	public static AColGroup genColGroupConst(int numCols, double value) {
		if(numCols <= 0)
			throw new DMLCompressionException("Invalid construction of constant column group with cols: " + numCols);
		final int[] colIndices = Util.genColsIndices(numCols);

		if(value == 0)
			return new ColGroupEmpty(colIndices);
		return genColGroupConst(colIndices, value);
	}

	/**
	 * Generate a constant column group.
	 * 
	 * @param values The value vector that contains all the unique values for each column in the matrix.
	 * @return A Constant column group.
	 */
	public static AColGroup genColGroupConst(double[] values) {
		final int[] colIndices = Util.genColsIndices(values.length);
		return genColGroupConst(colIndices, values);
	}

	/**
	 * Generate a constant column group.
	 * 
	 * It is assumed that the column group is intended for use, therefore zero value is allowed.
	 * 
	 * @param cols  The specific column indexes that is contained in this constant group.
	 * @param value The value contained in all cells.
	 * @return A Constant column group.
	 */
	public static AColGroup genColGroupConst(int[] cols, double value) {
		final int numCols = cols.length;
		double[] values = new double[numCols];
		for(int i = 0; i < numCols; i++)
			values[i] = value;
		return genColGroupConst(cols, values);
	}

	/**
	 * Generate a constant column group.
	 * 
	 * @param cols   The specific column indexes that is contained in this constant group.
	 * @param values The value vector that contains all the unique values for each column in the matrix.
	 * @return A Constant column group.
	 */
	public static AColGroup genColGroupConst(int[] cols, double[] values) {
		if(cols.length != values.length)
			throw new DMLCompressionException("Invalid size of values compared to columns");
		ADictionary dict = new Dictionary(values);
		return ColGroupConst.create(cols, dict);
	}

	/**
	 * Generate a constant column group.
	 * 
	 * @param numCols The number of columns.
	 * @param dict    The dictionary to contain int the Constant group.
	 * @return A Constant column group.
	 */
	public static AColGroup genColGroupConst(int numCols, ADictionary dict) {
		if(numCols != dict.getValues().length)
			throw new DMLCompressionException(
				"Invalid construction of const column group with different number of columns in arguments");
		final int[] colIndices = Util.genColsIndices(numCols);
		return ColGroupConst.create(colIndices, dict);
	}

	private static List<AColGroup> genEmpty(MatrixBlock in, CompressionSettings compSettings) {
		List<AColGroup> ret = new ArrayList<>(1);
		ret.add(genColGroupConst(compSettings.transposed ? in.getNumRows() : in.getNumColumns(), 0));
		return ret;
	}

	private static List<AColGroup> compressColGroupsSingleThreaded(MatrixBlock in, CompressedSizeInfo csi,
		CompressionSettings compSettings) {
		List<AColGroup> ret = new ArrayList<>(csi.getNumberColGroups());
		List<CompressedSizeInfoColGroup> groups = csi.getInfo();

		Tmp tmpMap = new Tmp();
		for(CompressedSizeInfoColGroup g : groups)
			ret.addAll(compressColGroup(in, compSettings, tmpMap, g, 1));

		return ret;
	}

	private static List<AColGroup> compressColGroupsParallel(MatrixBlock in, CompressedSizeInfo csi,
		CompressionSettings compSettings, int k) {
		try {
			ExecutorService pool = CommonThreadPool.get(k);
			List<CompressTask> tasks = new ArrayList<>();

			List<List<CompressedSizeInfoColGroup>> threadGroups = makeGroups(csi.getInfo(), k);
			for(List<CompressedSizeInfoColGroup> tg : threadGroups)
				if(!tg.isEmpty())
					tasks.add(new CompressTask(in, tg, compSettings, Math.max(1, k / 2)));

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

	private static List<List<CompressedSizeInfoColGroup>> makeGroups(List<CompressedSizeInfoColGroup> groups, int k) {
		// sort by number of distinct items
		Collections.sort(groups, Comparator.comparing(g -> -g.getNumVals()));
		List<List<CompressedSizeInfoColGroup>> ret = new ArrayList<>();
		for(int i = 0; i < k; i++)
			ret.add(new ArrayList<>());

		for(int i = 0; i < groups.size(); i++)
			ret.get(i % k).add(groups.get(i));

		return ret;
	}

	static class CompressTask implements Callable<Collection<AColGroup>> {
		private final MatrixBlock _in;
		private final List<CompressedSizeInfoColGroup> _groups;
		private final CompressionSettings _compSettings;
		private final int _k;

		protected CompressTask(MatrixBlock in, List<CompressedSizeInfoColGroup> groups, CompressionSettings compSettings,
			int k) {
			_in = in;
			_groups = groups;
			_compSettings = compSettings;
			_k = k;
		}

		@Override
		public Collection<AColGroup> call() {
			try {
				ArrayList<AColGroup> res = new ArrayList<>();
				Tmp tmpMap = new Tmp();
				for(CompressedSizeInfoColGroup g : _groups)
					res.addAll(compressColGroup(_in, _compSettings, tmpMap, g, _k));
				return res;
			}
			catch(Exception e) {
				e.printStackTrace();
				throw e;
			}
		}
	}

	private static Collection<AColGroup> compressColGroup(MatrixBlock in, CompressionSettings compSettings, Tmp tmpMap,
		CompressedSizeInfoColGroup cg, int k) {
		final int inCols = compSettings.transposed ? in.getNumRows() : in.getNumColumns();
		if(LOG.isDebugEnabled() && inCols < 1000) {
			Timing time = new Timing(true);
			time.start();
			Collection<AColGroup> ret = compressColGroupExecute(in, compSettings, tmpMap, cg, k);
			LOG.debug(String.format("time[ms]: %10.2f %25s %s cols:%s", time.stop(), getColumnTypesString(ret),
				getEstimateVsActualSize(ret, cg), Arrays.toString(cg.getColumns())));
			return ret;
		}
		return compressColGroupExecute(in, compSettings, tmpMap, cg, k);
	}

	private static String getColumnTypesString(Collection<AColGroup> ret) {
		if(ret.size() == 1)
			return ret.iterator().next().getClass().getSimpleName().toString();
		else {
			StringBuilder sb = new StringBuilder();
			for(AColGroup g : ret) {
				sb.append(g.getClass().getSimpleName().toString());
				sb.append(" ");
			}
			return sb.toString();
		}
	}

	private static String getEstimateVsActualSize(Collection<AColGroup> ret, CompressedSizeInfoColGroup cg) {
		long est = cg.getMinSize();
		long act = ret.stream().mapToLong(a -> a.estimateInMemorySize()).sum();
		return String.format("[B] %10d -- %10d", est, act);
	}

	private static Collection<AColGroup> compressColGroupExecute(MatrixBlock in, CompressionSettings compSettings,
		Tmp tmpMap, CompressedSizeInfoColGroup cg, int k) {
		final int[] colIndexes = cg.getColumns();
		if(in.isInSparseFormat() && compSettings.transposed) {
			final SparseBlock sb = in.getSparseBlock();
			for(int col : colIndexes)
				if(sb.isEmpty(col))
					return compressColGroupAndExtractEmptyColumns(in, compSettings, tmpMap, cg, k);
			return Collections.singletonList(compressColGroupForced(in, compSettings, tmpMap, cg, k));
		}
		else
			return Collections.singletonList(compressColGroupForced(in, compSettings, tmpMap, cg, k));
	}

	private static Collection<AColGroup> compressColGroupAndExtractEmptyColumns(MatrixBlock in,
		CompressionSettings compSettings, Tmp tmpMap, CompressedSizeInfoColGroup cg, int k) {
		final IntArrayList e = new IntArrayList();
		final IntArrayList v = new IntArrayList();
		final SparseBlock sb = in.getSparseBlock();
		final int[] colIndexes = cg.getColumns();
		for(int col : colIndexes) {
			if(sb.isEmpty(col))
				e.appendValue(col);
			else
				v.appendValue(col);
		}
		AColGroup empty = new ColGroupEmpty(e.extractValues(true));
		if(v.size() > 0) {
			AColGroup colGroup = compressColGroupForced(in, compSettings, tmpMap, cg, v.extractValues(true), k);
			return Arrays.asList(empty, colGroup);
		}
		else
			return Collections.singletonList(empty);
	}

	private static AColGroup compressColGroupForced(MatrixBlock in, CompressionSettings compSettings, Tmp tmpMap,
		CompressedSizeInfoColGroup cg, int k) {
		final int[] colIndexes = cg.getColumns();
		return compressColGroupForced(in, compSettings, tmpMap, cg, colIndexes, k);
	}

	private static AColGroup compressColGroupForced(MatrixBlock in, CompressionSettings cs, Tmp tmp,
		CompressedSizeInfoColGroup cg, int[] colIndexes, int k) {
		final int nrUniqueEstimate = cg.getNumVals();
		CompressionType estimatedBestCompressionType = cg.getBestCompressionType();

		if(estimatedBestCompressionType == CompressionType.UNCOMPRESSED) // don't construct mapping if uncompressed
			return new ColGroupUncompressed(colIndexes, in, cs.transposed);
		else if(estimatedBestCompressionType == CompressionType.SDC && colIndexes.length == 1 && in.isInSparseFormat() &&
			cs.transposed) // Leverage the Sparse matrix, to construct SDC group
			return compressSDCFromSparseTransposedBlock(in, colIndexes, in.getNumColumns(),
				tmp.getDblCountMap(nrUniqueEstimate), cs);
		else if(colIndexes.length > 1 && estimatedBestCompressionType == CompressionType.DDC)
			return directCompressDDC(colIndexes, in, cs, cg, k);
		else {
			final int numRows = cs.transposed ? in.getNumColumns() : in.getNumRows();
			final ABitmap ubm = BitmapEncoder.extractBitmap(colIndexes, in, cs.transposed, nrUniqueEstimate,
				cs.sortTuplesByFrequency);
			return compress(colIndexes, numRows, ubm, estimatedBestCompressionType, cs, cg.getTupleSparsity());
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

		switch(compType) {
			case DDC:
				return compressDDC(colIndexes, rlen, ubm, cs, tupleSparsity);
			case RLE:
				return compressRLE(colIndexes, rlen, ubm, cs, tupleSparsity);
			case OLE:
				return compressOLE(colIndexes, rlen, ubm, cs, tupleSparsity);
			case SDC:
				return compressSDC(colIndexes, rlen, ubm, cs, tupleSparsity);
			// CONST and EMPTY are handled above switch statement.
			// UNCOMPRESSED is handled before extraction of ubm
			default:
				throw new DMLCompressionException("Not implemented compression of " + compType + "in factory.");
		}
	}

	private static AColGroup directCompressDDC(int[] colIndexes, MatrixBlock raw, CompressionSettings cs,
		CompressedSizeInfoColGroup cg, int k) {
		final int rlen = cs.transposed ? raw.getNumColumns() : raw.getNumRows();
		// use a Map that is at least char size.
		final int nVal = cg.getNumVals() < 16 ? 16 : Math.max(cg.getNumVals(), 257);
		return directCompressDDC(colIndexes, raw, cs, cg, MapToFactory.create(rlen, nVal), rlen, k);
	}

	private static AColGroup directCompressDDC(int[] colIndexes, MatrixBlock raw, CompressionSettings cs,
		CompressedSizeInfoColGroup cg, AMapToData data, int rlen, int k) {
		final int fill = data.getUpperBoundValue();
		data.fill(fill);

		DblArrayCountHashMap map = new DblArrayCountHashMap(cg.getNumVals());
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
		ADictionary dict = DictionaryFactory.create(map, colIndexes.length, extra);
		if(extra) {
			data.replace(fill, map.size());
			data.setUnique(map.size() + 1);
		}
		else
			data.setUnique(map.size());

		AMapToData resData = MapToFactory.resize(data, map.size() + (extra ? 1 : 0));
		ColGroupDDC res = new ColGroupDDC(colIndexes, rlen, dict, resData, null);
		return res;
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

		ADictionary dict = DictionaryFactory.create(ubm, tupleSparsity);
		if(ubm.getNumValues() == 1) {
			if(numZeros >= largestOffset) {
				final AOffset off = OffsetFactory.createOffset(ubm.getOffsetList()[0].extractValues(true));
				return new ColGroupSDCSingleZeros(colIndexes, rlen, dict, off, null);
			}
			else {
				dict = DictionaryFactory.moveFrequentToLastDictionaryEntry(dict, ubm, rlen, largestIndex);
				return setupSingleValueSDCColGroup(colIndexes, rlen, ubm, dict);
			}
		}
		else if(numZeros >= largestOffset)
			return setupMultiValueZeroColGroup(colIndexes, rlen, ubm, dict, cs);
		else {
			dict = DictionaryFactory.moveFrequentToLastDictionaryEntry(dict, ubm, rlen, largestIndex);
			return setupMultiValueColGroup(colIndexes, numZeros, rlen, ubm, largestIndex, dict, cs);
		}
	}

	private static AColGroup setupMultiValueZeroColGroup(int[] colIndexes, int rlen, ABitmap ubm, ADictionary dict,
		CompressionSettings cs) {
		IntArrayList[] offsets = ubm.getOffsetList();
		AInsertionSorter s = InsertionSorterFactory.create(rlen, offsets, cs.sdcSortType);
		AOffset indexes = OffsetFactory.createOffset(s.getIndexes());
		AMapToData data = s.getData();
		int[] counts = new int[offsets.length + 1];
		int sum = 0;
		for(int i = 0; i < offsets.length; i++) {
			counts[i] = offsets[i].size();
			sum += counts[i];
		}
		counts[offsets.length] = rlen - sum;
		return ColGroupSDCZeros.create(colIndexes, rlen, dict, indexes, data, counts);
	}

	private static AColGroup setupMultiValueColGroup(int[] colIndexes, int numZeros, int rlen, ABitmap ubm,
		int largestIndex, ADictionary dict, CompressionSettings cs) {
		IntArrayList[] offsets = ubm.getOffsetList();
		AInsertionSorter s = InsertionSorterFactory.createNegative(rlen, offsets, largestIndex, cs.sdcSortType);
		AOffset indexes = OffsetFactory.createOffset(s.getIndexes());
		AMapToData _data = s.getData();
		_data = MapToFactory.resize(_data, _data.getUnique() - 1);
		return ColGroupSDC.create(colIndexes, rlen, dict, indexes, _data, null);
	}

	private static AColGroup setupSingleValueSDCColGroup(int[] colIndexes, int rlen, ABitmap ubm, ADictionary dict) {
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

		return new ColGroupSDCSingle(colIndexes, rlen, dict, off, null);
	}

	private static AColGroup compressDDC(int[] colIndexes, int rlen, ABitmap ubm, CompressionSettings cs,
		double tupleSparsity) {
		boolean zeros = ubm.getNumOffsets() < rlen;
		ADictionary dict = DictionaryFactory.create(ubm, tupleSparsity, zeros);
		AMapToData data = MapToFactory.create(rlen, zeros, ubm.getOffsetList());
		return new ColGroupDDC(colIndexes, rlen, dict, data, null);
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

	private static AColGroup compressSDCFromSparseTransposedBlock(MatrixBlock mb, int[] cols, int rlen,
		DoubleCountHashMap map, CompressionSettings cs) {
		// This method should only be called if the cols argument is length 1.
		final SparseBlock sb = mb.getSparseBlock();
		final int sbRow = cols[0];
		final int apos = sb.pos(sbRow);
		final int alen = sb.size(sbRow) + apos;
		final double[] vals = sb.values(sbRow);

		// count distinct items frequencies
		for(int j = apos; j < alen; j++)
			map.increment(vals[j]);

		List<DCounts> entries = map.extractValues();
		Collections.sort(entries, Comparator.comparing(x -> -x.count));

		if(entries.get(0).count < rlen - sb.size(sbRow)) {
			// If the zero is the default value.
			final int[] counts = new int[entries.size() + 1];
			final double[] dict = new double[entries.size()];
			int sum = 0;
			for(int i = 0; i < entries.size(); i++) {
				final DCounts x = entries.get(i);
				counts[i] = x.count;
				sum += x.count;
				dict[i] = x.key;
				x.count = i;
			}

			counts[entries.size()] = rlen - sum;
			final AOffset offsets = OffsetFactory.createOffset(sb.indexes(sbRow), apos, alen);
			if(entries.size() <= 1)
				return new ColGroupSDCSingleZeros(cols, rlen, new Dictionary(dict), offsets, counts);
			else {
				final AMapToData mapToData = MapToFactory.create((alen - apos), entries.size());
				for(int j = apos; j < alen; j++)
					mapToData.set(j - apos, map.get(vals[j]));
				return ColGroupSDCZeros.create(cols, rlen, new Dictionary(dict), offsets, mapToData, counts);
			}
		}
		else {
			final ABitmap ubm = BitmapEncoder.extractBitmap(cols, mb, true, entries.size(), true);
			// zero is not the default value fall back to the standard compression path.
			return compressSDC(cols, rlen, ubm, cs, 1.0);
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
