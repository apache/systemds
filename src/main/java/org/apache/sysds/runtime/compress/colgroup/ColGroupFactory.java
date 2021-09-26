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
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToInt;
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
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * Factory pattern for constructing ColGroups.
 */
public final class ColGroupFactory {
	private static final Log LOG = LogFactory.getLog(ColGroupFactory.class.getName());

	/**
	 * The actual compression method, that handles the logic of compressing multiple columns together. This method also
	 * have the responsibility of correcting any estimation errors previously made.
	 * 
	 * @param in           The input matrix, that could have been transposed if CompSettings was set to do that
	 * @param csi          The compression Information extracted from the estimation, this contains which groups of
	 *                     columns to compress together
	 * @param compSettings The compression settings to construct the compression based on.
	 * @param k            The degree of parallelism used.
	 * @return A Resulting array of ColGroups, containing the compressed information from the input matrix block.
	 */
	public static List<AColGroup> compressColGroups(MatrixBlock in, CompressedSizeInfo csi,
		CompressionSettings compSettings, int k) {
		for(CompressedSizeInfoColGroup g : csi.getInfo())
			g.clearMap();

		if((compSettings.transposed && (in.getNumColumns() == 1)) || (!compSettings.transposed && in.getNumRows() == 1))
			throw new DMLCompressionException("Error input for compression only have 1 row");
		if(k <= 1)
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
		int[] colIndices = genColsIndices(numCols);

		if(value == 0)
			return new ColGroupEmpty(colIndices);
		else
			return getColGroupConst(colIndices, value);
	}

	/**
	 * Generate a constant column group.
	 * 
	 * @param numCols The number of columns.
	 * @param values  The value vector that contains all the unique values for each column in the matrix.
	 * @return A Constant column group.
	 */
	public static AColGroup genColGroupConst(int numCols, double[] values) {
		if(numCols <= 0)
			throw new DMLCompressionException("Invalid construction of constant column group with cols: " + numCols);
		int[] colIndices = genColsIndices(numCols);
		return getColGroupConst(colIndices, values);
	}

	/**
	 * Generate a constant column group.
	 * 
	 * @param cols  The specific column indexes that is contained in this constant group.
	 * @param value The value contained in all cells.
	 * @return A Constant column group.
	 */
	public static AColGroup getColGroupConst(int[] cols, double value) {
		final int numCols = cols.length;
		double[] values = new double[numCols];
		for(int i = 0; i < numCols; i++)
			values[i] = value;
		return getColGroupConst(cols, values);
	}

	/**
	 * Generate a constant column group.
	 * 
	 * @param cols   The specific column indexes that is contained in this constant group.
	 * @param values The value vector that contains all the unique values for each column in the matrix.
	 * @return A Constant column group.
	 */
	public static AColGroup getColGroupConst(int[] cols, double[] values) {
		ADictionary dict = new Dictionary(values);
		return new ColGroupConst(cols, dict);
	}

	/**
	 * Generate a constant column group.
	 * 
	 * @param numCols The number of columns.
	 * @param dict    The dictionary to contain int the Constant group.
	 * @return A Constant column group.
	 */
	public static AColGroup getColGroupConst(int numCols, ADictionary dict) {
		int[] colIndices = genColsIndices(numCols);
		return new ColGroupConst(colIndices, dict);
	}

	private static List<AColGroup> compressColGroupsSingleThreaded(MatrixBlock in, CompressedSizeInfo csi,
		CompressionSettings compSettings) {
		List<AColGroup> ret = new ArrayList<>(csi.getNumberColGroups());
		List<CompressedSizeInfoColGroup> groups = csi.getInfo();

		Tmp tmpMap = new Tmp();
		for(CompressedSizeInfoColGroup g : groups) {
			ret.addAll(compressColGroup(in, compSettings, tmpMap, g, 1));
		}

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

			List<AColGroup> ret = new ArrayList<>(csi.getNumberColGroups());
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

	private static class CompressTask implements Callable<Collection<AColGroup>> {
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
			ArrayList<AColGroup> res = new ArrayList<>();
			Tmp tmpMap = new Tmp();
			for(CompressedSizeInfoColGroup g : _groups)
				res.addAll(compressColGroup(_in, _compSettings, tmpMap, g, _k));
			return res;
		}
	}

	private static Collection<AColGroup> compressColGroup(MatrixBlock in, CompressionSettings compSettings, Tmp tmpMap,
		CompressedSizeInfoColGroup cg, int k) {
		final int inCols = compSettings.transposed ? in.getNumRows() : in.getNumColumns();
		if(LOG.isDebugEnabled() && inCols < 1000) {
			Timing time = new Timing(true);
			time.start();
			Collection<AColGroup> ret = compressColGroupExecute(in, compSettings, tmpMap, cg, k);
			LOG.debug(
				cg.getBestCompressionType() + "\ttime [ms]: " + time.stop() + "\tnrColumns: " + cg.getColumns().length);
			return ret;
		}
		return compressColGroupExecute(in, compSettings, tmpMap, cg, k);
	}

	private static Collection<AColGroup> compressColGroupExecute(MatrixBlock in, CompressionSettings compSettings,
		Tmp tmpMap, CompressedSizeInfoColGroup cg, int k) {
		final int[] colIndexes = cg.getColumns();
		if(in.isEmpty())
			return Collections.singletonList(new ColGroupEmpty(colIndexes));
		else if(in.isInSparseFormat() && compSettings.transposed) {
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
			return compressSDCZero(in.getSparseBlock(), colIndexes, in.getNumColumns(),
				tmp.getDblCountMap(nrUniqueEstimate));
		else {
			if(colIndexes.length > 1 && estimatedBestCompressionType == CompressionType.DDC)
				return directCompressDDC(colIndexes, in, cs, cg, k);

			final int numRows = cs.transposed ? in.getNumColumns() : in.getNumRows();
			final ABitmap ubm = BitmapEncoder.extractBitmap(colIndexes, in, cs.transposed, nrUniqueEstimate);
			return compress(colIndexes, numRows, ubm, estimatedBestCompressionType, cs, in, cg.getTupleSparsity());
		}
	}

	/**
	 * Method for compressing a column group.
	 * 
	 * @param colIndexes     The column indexes to compress
	 * @param rlen           The number of rows in the columns
	 * @param ubm            The bitmap containing all the data needed for the compression (unless Uncompressed ColGroup)
	 * @param compType       The CompressionType selected.
	 * @param cs             The compression settings used for the given compression.
	 * @param rawMatrixBlock The copy of the original input (maybe transposed) MatrixBlock
	 * @param tupleSparsity  The sparsity of the dictionary when constructed.
	 * @return A Compressed ColGroup
	 */
	public static AColGroup compress(int[] colIndexes, int rlen, ABitmap ubm, CompressionType compType,
		CompressionSettings cs, MatrixBlock rawMatrixBlock, double tupleSparsity) {

		try {
			if(ubm == null)
				return new ColGroupEmpty(colIndexes);

			final IntArrayList[] of = ubm.getOffsetList();
			if(of.length == 1 && of[0].size() == rlen) // If this always constant
				return new ColGroupConst(colIndexes, DictionaryFactory.create(ubm));

			if(LOG.isTraceEnabled())
				LOG.trace("compressing to: " + compType);

			if(cs.sortValuesByLength)
				ubm.sortValuesByFrequency();

			switch(compType) {
				case DDC:
					return compressDDC(colIndexes, rlen, ubm, cs, tupleSparsity);
				case RLE:
					return compressRLE(colIndexes, rlen, ubm, cs, tupleSparsity);
				case OLE:
					return compressOLE(colIndexes, rlen, ubm, cs, tupleSparsity);
				case SDC:
					return compressSDC(colIndexes, rlen, ubm, cs, tupleSparsity);
				case UNCOMPRESSED:
					return new ColGroupUncompressed(colIndexes, rawMatrixBlock, cs.transposed);
				case CONST:
				case EMPTY:
					throw new DMLCompressionException(
						"Should never use these column groups since the code defaults to these if applicable");
				default:
					throw new DMLCompressionException("Not implemented ColGroup Type compressed in factory.");
			}
		}
		catch(Exception e) {
			throw new DMLCompressionException("Error in construction of colGroup type: " + compType, e);
		}
	}

	private static ColGroupDDC directCompressDDC(int[] colIndexes, MatrixBlock raw, CompressionSettings cs,
		CompressedSizeInfoColGroup cg, int k) {
		final int rlen = cs.transposed ? raw.getNumColumns() : raw.getNumRows();
		// use a Map that is at least char size.
		final int nVal = Math.max(cg.getNumVals(), 257);
		return directCompressDDC(colIndexes, raw, cs, cg, MapToFactory.create(rlen, nVal), rlen, k);
	}

	private static ColGroupDDC directCompressDDC(int[] colIndexes, MatrixBlock raw, CompressionSettings cs,
		CompressedSizeInfoColGroup cg, AMapToData data, int rlen, int k) {
		final int fill = (data instanceof MapToInt) ? Integer.MAX_VALUE : Character.MAX_VALUE;
		data.fill(fill);

		DblArrayCountHashMap map = new DblArrayCountHashMap(cg.getNumVals());

		if(rlen < 10000 || k == 1)
			readToMapDDC(colIndexes, raw, map, cs, data, 0, rlen);
		else
			parallelReadToMapDDC(colIndexes, raw, map, cs, data, rlen, k);

		boolean extra = false;
		for(int i = 0; i < rlen; i++)
			if(data.getIndex(i) == fill) {
				extra = true;
				break;
			}

		ADictionary dict = DictionaryFactory.create(map, colIndexes.length, extra);
		if(extra)
			data.replace(fill, map.size());

		AMapToData resData = MapToFactory.resize(data, map.size() + (extra ? 1 : 0));
		ColGroupDDC res = new ColGroupDDC(colIndexes, rlen, dict, resData, null);
		return res;
	}

	private static void readToMapDDC(final int[] colIndexes, final MatrixBlock raw, final DblArrayCountHashMap map,
		final CompressionSettings cs, final AMapToData data, final int rl, final int ru) {
		ReaderColumnSelection reader = ReaderColumnSelection.createReader(raw, colIndexes, cs.transposed, rl, ru);
		DblArray cellVals = null;
		while((cellVals = reader.nextRow()) != null) {
			final int id = map.increment(cellVals);
			final int row = reader.getCurrentRowIndex();
			data.set(row, id);
		}
	}

	private static void parallelReadToMapDDC(final int[] colIndexes, final MatrixBlock raw,
		final DblArrayCountHashMap map, final CompressionSettings cs, final AMapToData data, final int rlen,
		final int k) {

		try {
			final int blk = Math.max(rlen / colIndexes.length / k, 128000 / colIndexes.length);
			ExecutorService pool = CommonThreadPool.get(Math.min(Math.max(rlen / blk, 1), k));
			List<readToMapDDCTask> tasks = new ArrayList<>();

			for(int i = 0; i < rlen; i += blk) {
				int end = Math.min(rlen, i + blk);
				tasks.add(new readToMapDDCTask(colIndexes, raw, map, cs, data, i, end));
			}

			for(Future<Object> t : pool.invokeAll(tasks))
				t.get();

			pool.shutdown();
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Failed to parallelize DDC compression");
		}
	}

	private static class readToMapDDCTask implements Callable<Object> {
		private final int[] _colIndexes;
		private final MatrixBlock _raw;
		private final DblArrayCountHashMap _map;
		private final CompressionSettings _cs;
		private final AMapToData _data;
		private final int _rl;
		private final int _ru;

		protected readToMapDDCTask(int[] colIndexes, MatrixBlock raw, DblArrayCountHashMap map, CompressionSettings cs,
			AMapToData data, int rl, int ru) {
			_colIndexes = colIndexes;
			_raw = raw;
			_map = map;
			_cs = cs;
			_data = data;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public Collection<AColGroup> call() {
			readToMapDDC(_colIndexes, _raw, _map, _cs, _data, _rl, _ru);
			return null;
		}
	}

	private static AColGroup compressSDC(int[] colIndexes, int rlen, ABitmap ubm, CompressionSettings cs,
		double tupleSparsity) {

		final int numZeros = (int) ((long) rlen - ubm.getNumOffsets());
		int largestOffset = 0;
		int largestIndex = 0;
		if(!cs.sortValuesByLength) {
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
		if(numZeros >= largestOffset && ubm.getOffsetList().length == 1) {
			AOffset off = OffsetFactory.create(ubm.getOffsetList()[0].extractValues(true));
			return new ColGroupSDCSingleZeros(colIndexes, rlen, dict, off);
		}
		else if(ubm.getOffsetList().length == 1) {
			dict = DictionaryFactory.moveFrequentToLastDictionaryEntry(dict, ubm, rlen, largestIndex);
			return setupSingleValueSDCColGroup(colIndexes, rlen, ubm, dict);
		}
		else if(numZeros >= largestOffset)
			return setupMultiValueZeroColGroup(colIndexes, rlen, ubm, dict);
		else {
			dict = DictionaryFactory.moveFrequentToLastDictionaryEntry(dict, ubm, rlen, largestIndex);
			return setupMultiValueColGroup(colIndexes, numZeros, rlen, ubm, largestIndex, dict);
		}
	}

	private static AColGroup setupMultiValueZeroColGroup(int[] colIndexes, int rlen, ABitmap ubm, ADictionary dict) {
		try {
			IntArrayList[] offsets = ubm.getOffsetList();
			AInsertionSorter s = InsertionSorterFactory.create(rlen, offsets);
			AOffset indexes = OffsetFactory.create(s.getIndexes());
			AMapToData data = s.getData();

			int[] counts = new int[offsets.length + 1];
			int sum = 0;
			for(int i = 0; i < offsets.length; i++) {
				counts[i] = offsets[i].size();
				sum += counts[i];
			}
			counts[offsets.length] = rlen - sum;

			ColGroupValue ret = new ColGroupSDCZeros(colIndexes, rlen, dict, indexes, data, counts);
			return ret;
		}
		catch(Exception e) {
			throw new DMLCompressionException(
				"Failed to construct SDC Zero Group with columns :" + Arrays.toString(colIndexes), e);
		}
	}

	private static AColGroup setupMultiValueColGroup(int[] colIndexes, int numZeros, int rlen, ABitmap ubm,
		int largestIndex, ADictionary dict) {
		try {
			IntArrayList[] offsets = ubm.getOffsetList();

			AInsertionSorter s = InsertionSorterFactory.create(rlen, offsets, largestIndex);
			AOffset indexes = OffsetFactory.create(s.getIndexes());
			AMapToData _data = s.getData();
			ColGroupValue ret = new ColGroupSDC(colIndexes, rlen, dict, indexes, _data);
			return ret;
		}
		catch(Exception e) {
			throw new DMLCompressionException(
				"Failed to construct SDC Group with columns :\n" + Arrays.toString(colIndexes), e);
		}
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
		AOffset off = OffsetFactory.create(indexes);

		return new ColGroupSDCSingle(colIndexes, rlen, dict, off);
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

	private static int[] genColsIndices(int numCols) {
		int[] colIndices = new int[numCols];
		for(int i = 0; i < numCols; i++)
			colIndices[i] = i;
		return colIndices;
	}

	private static AColGroup compressSDCZero(SparseBlock sb, int[] cols, int rlen, DoubleCountHashMap map) {
		final int sbRow = cols[0];
		final int apos = sb.pos(sbRow);
		final int alen = sb.size(sbRow) + apos;
		final AOffset offsets = OffsetFactory.create(sb.indexes(sbRow), apos, alen);
		final double[] vals = sb.values(sbRow);

		// count distinct items frequencies
		for(int j = apos; j < alen; j++)
			map.increment(vals[j]);

		List<DCounts> entries = map.extractValues();
		Collections.sort(entries, Comparator.comparing(x -> -x.count));
		int[] counts = new int[entries.size() + 1];
		int sum = 0;
		double[] dict = new double[entries.size()];
		for(int i = 0; i < entries.size(); i++) {
			DCounts x = entries.get(i);
			counts[i] = x.count;
			sum += x.count;
			dict[i] = x.key;
			x.count = i;
		}

		counts[entries.size()] = rlen - sum;

		AMapToData mapToData = MapToFactory.create((alen - apos), entries.size());
		for(int j = apos; j < alen; j++)
			mapToData.set(j - apos, map.get(vals[j]));

		return new ColGroupSDCZeros(cols, rlen, new Dictionary(dict), offsets, mapToData, counts);
	}

	/**
	 * Temp reuse object, to contain intermediates for compressing column groups that can be used by the same thread
	 * again for subsequent compressions.
	 */
	private static class Tmp {
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
