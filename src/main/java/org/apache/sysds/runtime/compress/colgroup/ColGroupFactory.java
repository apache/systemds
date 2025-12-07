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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.bitmap.BitmapEncoder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DeltaDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.functional.LinearRegression;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.insertionsort.AInsertionSorter;
import org.apache.sysds.runtime.compress.colgroup.insertionsort.InsertionSorterFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.cost.ACostEstimate;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.lib.CLALibCombineGroups;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.utils.ACount;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayCountHashMap;
import org.apache.sysds.runtime.compress.utils.DoubleCountHashMap;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.utils.stats.Timing;

/**
 * Factory class for constructing ColGroups.
 */
public class ColGroupFactory {
	protected static final Log LOG = LogFactory.getLog(ColGroupFactory.class.getName());

	/** Input matrix to compress */
	private final MatrixBlock in;
	/** Compression information to compress based on */
	private final CompressedSizeInfo csi;
	/** Compression settings specifying for instance if the input is transposed */
	private final CompressionSettings cs;
	/** The cost estimator to use to calculate cost of compression */
	private final ACostEstimate ce;
	/** Parallelization degree */
	private final int k;
	/** number of rows in input (taking into account if the input is transposed) */
	private final int nRow;
	/** number of columns in input (taking into account if the input is transposed) */
	private final int nCol;
	/** Thread pool to use in execution of compression */
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
	 * @return A resulting array of ColGroups, containing the compressed information from the input matrix block.
	 */
	public static List<AColGroup> compressColGroups(MatrixBlock in, CompressedSizeInfo csi, CompressionSettings cs) {
		return compressColGroups(in, csi, cs, 1);
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
			if(in instanceof CompressedMatrixBlock)
				return CLALibCombineGroups.combine((CompressedMatrixBlock) in, csi, pool, k);
			else
				return compressExecute();
		}
		catch(Exception e) {
			throw new DMLCompressionException("Compression Failed", e);
		}
		finally {
			if(pool != null)
				pool.shutdown();
		}
	}

	private List<AColGroup> compressExecute() throws Exception {
		if(in.isEmpty()) {
			AColGroup empty = ColGroupEmpty.create(cs.transposed ? in.getNumRows() : in.getNumColumns());
			return Collections.singletonList(empty);
		}
		else if(k <= 1)
			return compressColGroupsSingleThreaded();
		else
			return compressColGroupsParallel();
	}

	private List<AColGroup> compressColGroupsSingleThreaded() throws Exception {
		List<AColGroup> ret = new ArrayList<>(csi.getNumberColGroups());

		for(CompressedSizeInfoColGroup g : csi.getInfo())
			ret.add(compressColGroup(g));

		return ret;
	}

	private List<AColGroup> compressColGroupsParallel() throws Exception {

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

	protected AColGroup compressColGroup(CompressedSizeInfoColGroup cg) throws Exception {
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
		final String cols = est.getColumns().toString();
		final String wanted = est.getBestCompressionType().toString();
		if(estC < actC * 0.75) {
			String warning = "The estimate cost is significantly off : " + est;
			LOG.debug(
				String.format("time[ms]: %10.2f %25s est %10.0f -- act %10.0f distinct:%5d cols:%s wanted:%s\n\t\t%s",
					time, retType, estC, actC, act.getNumValues(), cols, wanted, warning));
		}
		else {
			LOG.debug(String.format("time[ms]: %10.2f %25s est %10.0f -- act %10.0f distinct:%5d cols:%s wanted:%s",
				time, retType, estC, actC, act.getNumValues(), cols, wanted));
		}

	}

	private AColGroup compressColGroupAllSteps(CompressedSizeInfoColGroup cg) throws Exception {
		AColGroup g = compress(cg);
		if(ce != null && ce.shouldSparsify() && nCol >= 4)
			g = sparsifyFOR(g);
		return g;
	}

	private static AColGroup sparsifyFOR(AColGroup g) {
		if(g instanceof ColGroupDDC)
			return ((ColGroupDDC) g).sparsifyFOR();
		else if(g instanceof ColGroupSDC)
			return ((ColGroupSDC) g).sparsifyFOR();
		else
			return g;
	}

	private AColGroup compress(CompressedSizeInfoColGroup cg) throws Exception {
		final IColIndex colIndexes = cg.getColumns();
		final CompressionType ct = cg.getBestCompressionType();
		final boolean t = cs.transposed;

		// Fast path compressions
		if((ct == CompressionType.EMPTY && !t) || //
			(t && colIndexes.size() == 1 && in.isInSparseFormat() // Empty Column
				&& in.getSparseBlock().isEmpty(colIndexes.get(0))))
			// TODO: handle quantization-fused compression if deemed necessary,
			// but if the matrix reaches here, it likely doesn't need quantization.
			return new ColGroupEmpty(colIndexes);
		else if(ct == CompressionType.UNCOMPRESSED) { // don't construct mapping if uncompressed
			if(cs.scaleFactors != null) {
				return ColGroupUncompressed.createQuantized(colIndexes, in, t, cs.scaleFactors);
			}
			else {
				return ColGroupUncompressed.create(colIndexes, in, t);
			}
		}
		else if((ct == CompressionType.SDC || ct == CompressionType.CONST) //
			&& in.isInSparseFormat() //
			&& t && (//
			(colIndexes.size() > 1 && cg.getNumOffs() < 0.3 * nRow) //
				|| colIndexes.size() == 1)) {
			if(cs.scaleFactors != null) {
				throw new NotImplementedException(); // TODO: handle quantization-fused compression
			}
			else {
				return compressSDCFromSparseTransposedBlock(colIndexes, cg.getNumVals(), cg.getTupleSparsity());
			}
		}
		else if(ct == CompressionType.DDC) {
			return directCompressDDC(colIndexes, cg);
		}
		else if(ct == CompressionType.DeltaDDC) {
			return directCompressDeltaDDC(colIndexes, cg);
		}
		else if(ct == CompressionType.CONST && cs.preferDeltaEncoding) {
			return directCompressDeltaDDC(colIndexes, cg);
		}
		else if(ct == CompressionType.LinearFunctional) {
			if(cs.scaleFactors != null) {
				throw new NotImplementedException(); // quantization-fused compression NOT allowed
			}
			else {
				return compressLinearFunctional(colIndexes, in, cs);
			}
		}
		else if(ct == CompressionType.DDCFOR) {
			AColGroup g = directCompressDDC(colIndexes, cg);
			if(g instanceof ColGroupDDC)
				return ColGroupDDCFOR.sparsifyFOR((ColGroupDDC) g);
			return g;
		}
		else if(ct == CompressionType.SDC && colIndexes.size() == 1 && !t) {
			return compressSDCSingleColDirectBlock(colIndexes, cg.getNumVals());
		}

		final ABitmap ubm = BitmapEncoder.extractBitmap(colIndexes, in, cg.getNumVals(), cs);
		if(ubm == null) {// no values ... therefore empty
			return new ColGroupEmpty(colIndexes);
		}
		final IntArrayList[] of = ubm.getOffsetList();
		if(of.length == 1 && of[0].size() == nRow && ct != CompressionType.DeltaDDC) { // If this always constant
			return ColGroupConst.create(colIndexes, DictionaryFactory.create(ubm));
		}

		final double tupleSparsity = colIndexes.size() > 4 ? cg.getTupleSparsity() : 1.0;

		switch(ct) {
			case RLE:
				return ColGroupRLE.compressRLE(colIndexes, ubm, nRow, tupleSparsity);
			case OLE:
				return ColGroupOLE.compressOLE(colIndexes, ubm, nRow, tupleSparsity);
			case CONST: // in case somehow one requested const, but it was not const column fall back to SDC.
			case EMPTY:
				LOG.warn("Requested " + ct + " on non constant column, fallback to SDC");
			case SDC:
				return compressSDC(colIndexes, nRow, ubm, cs, tupleSparsity);
			case SDCFOR:
				AColGroup g = compressSDC(colIndexes, nRow, ubm, cs, tupleSparsity);
				if(g instanceof ColGroupSDC)
					return ColGroupSDCFOR.sparsifyFOR((ColGroupSDC) g);
				return g;
			default:
				throw new DMLCompressionException("Not implemented compression of " + ct + " in factory.");
		}
	}

	private AColGroup compressSDCSingleColDirectBlock(IColIndex colIndexes, int nVal) {
		final DoubleCountHashMap cMap = new DoubleCountHashMap(nVal);
		final int col = colIndexes.get(0);

		countElements(cMap, col);

		double def = cMap.getMostFrequent();
		final int dictSize = cMap.size() - 1;
		if(dictSize == 0)
			return ColGroupConst.create(colIndexes, def);

		int defCount = cMap.getC(def).count;
		cMap.replaceWithUIDs(def);
		IDictionary dict = Dictionary.create(cMap.getDictionary(dictSize));
		IntArrayList offs = new IntArrayList(nRow - defCount);
		AMapToData map = MapToFactory.create(nRow - defCount, dictSize);
		if(cs.scaleFactors != null) {
			getOffsetsScaled(offs, map, cMap, col, def);
		}
		else {
			getOffsets(offs, map, cMap, col, def);
		}
		AOffset aoff = OffsetFactory.createOffset(offs);

		return ColGroupSDC.create(colIndexes, nRow, dict, new double[] {def}, aoff, map, null);
	}

	private void getOffsetsScaled(IntArrayList offs, AMapToData map, DoubleCountHashMap cMap, int col, double def) {
		final double scaleFactor = cs.scaleFactors[0]; // Single column, thus single scalar value.

		if(in.isInSparseFormat()) {
			final SparseBlock sb = in.getSparseBlock();

			if(def == 0) { // If zero is the default value
				for(int r = 0; r < nRow; r++) {
					if(sb.isEmpty(r))
						continue; // Skip explicitly storing zero values

					final int apos = sb.pos(r);
					final int alen = sb.size(r) + apos;
					final int[] aix = sb.indexes(r);
					final int idx = Arrays.binarySearch(aix, apos, alen, col);

					if(idx >= 0) {
						double v = Math.floor(sb.values(r)[idx] * scaleFactor);
						map.set(offs.size(), cMap.getId(v));
						offs.appendValue(r);
					}
				}
			}

			else { // If zero is NOT the default value, track missing values explicitly
				for(int r = 0; r < nRow; r++) {
					if(sb.isEmpty(r)) {
						map.set(offs.size(), cMap.getId(0.0));
						offs.appendValue(r);
					}
					else {
						final int apos = sb.pos(r);
						final int alen = sb.size(r) + apos;
						final int[] aix = sb.indexes(r);
						final int idx = Arrays.binarySearch(aix, apos, alen, col);

						if(idx < 0) { // Missing entry
							map.set(offs.size(), cMap.getId(0.0));
							offs.appendValue(r);
						}
						else {
							double v = Math.floor(sb.values(r)[idx] * scaleFactor);
							if(!Util.eq(v, def)) {
								map.set(offs.size(), cMap.getId(v));
								offs.appendValue(r);
							}
						}
					}
				}
			}

		}
		else if(in.getDenseBlock().isContiguous()) {
			final double[] dv = in.getDenseBlockValues();
			int off = col;

			for(int r = 0; r < nRow; r++, off += nCol) {
				double scaledValue = Math.floor(dv[off] * scaleFactor);
				if(!Util.eq(scaledValue, def)) {
					map.set(offs.size(), cMap.getId(scaledValue));
					offs.appendValue(r);
				}
			}
		}
		else {
			final DenseBlock db = in.getDenseBlock();
			for(int r = 0; r < nRow; r++) {
				final double[] dv = db.values(r);
				int off = db.pos(r) + col;
				double scaledValue = Math.floor(dv[off] * scaleFactor);
				if(!Util.eq(scaledValue, def)) {
					map.set(offs.size(), cMap.getId(scaledValue));
					offs.appendValue(r);
				}
			}
		}
	}

	private void getOffsets(IntArrayList offs, AMapToData map, DoubleCountHashMap cMap, int col, double def) {

		if(in.isInSparseFormat()) {
			final SparseBlock sb = in.getSparseBlock();
			if(def == 0) {
				for(int r = 0; r < nRow; r++) {
					if(sb.isEmpty(r))
						continue;

					final int apos = sb.pos(r);
					final int alen = sb.size(r) + apos;
					final int[] aix = sb.indexes(r);
					final int idx = Arrays.binarySearch(aix, apos, alen, col);
					if(!(idx < 0)) {
						map.set(offs.size(), cMap.getId(sb.values(r)[idx]));
						offs.appendValue(r);
					}
				}
			}
			else {
				for(int r = 0; r < nRow; r++) {
					if(sb.isEmpty(r)) {
						map.set(offs.size(), cMap.getId(0.0));
						offs.appendValue(r);
					}
					else {
						final int apos = sb.pos(r);
						final int alen = sb.size(r) + apos;
						final int[] aix = sb.indexes(r);
						final int idx = Arrays.binarySearch(aix, apos, alen, col);
						if(idx < 0) {
							map.set(offs.size(), cMap.getId(0.0));
							offs.appendValue(r);
						}
						else {
							double v = sb.values(r)[idx];
							if(!Util.eq(sb.values(r)[idx], def)) {
								map.set(offs.size(), cMap.getId(v));
								offs.appendValue(r);
							}
						}
					}
				}

			}
		}
		else if(in.getDenseBlock().isContiguous()) {
			final double[] dv = in.getDenseBlockValues();
			int off = col;

			for(int r = 0; r < nRow; r++, off += nCol)
				if(!Util.eq(dv[off], def)) {
					map.set(offs.size(), cMap.getId(dv[off]));
					offs.appendValue(r);
				}
		}
		else {
			final DenseBlock db = in.getDenseBlock();
			for(int r = 0; r < nRow; r++) {
				final double[] dv = db.values(r);
				int off = db.pos(r) + col;
				if(!Util.eq(dv[off], def)) {
					map.set(offs.size(), cMap.getId(dv[off]));
					offs.appendValue(r);
				}
			}
		}
	}

	private void countElements(DoubleCountHashMap map, int col) {
		if(cs.scaleFactors != null) {
			if(in.isInSparseFormat()) {
				countElementsSparseScaled(map, col);
			}
			else if(in.getDenseBlock().isContiguous()) {
				countElementsDenseContiguousScaled(map, col);
			}
			else {
				countElementsDenseGenericScaled(map, col);
			}
		}
		else {
			if(in.isInSparseFormat()) {
				countElementsSparse(map, col);
			}
			else if(in.getDenseBlock().isContiguous()) {
				countElementsDenseContiguous(map, col);
			}
			else {
				countElementsDenseGeneric(map, col);
			}
		}
	}

	private void countElementsSparseScaled(DoubleCountHashMap map, int col) {
		final SparseBlock sb = in.getSparseBlock();

		double scaleFactor = cs.scaleFactors[0];
		for(int r = 0; r < nRow; r++) {
			if(sb.isEmpty(r))
				map.increment(0.0);
			else {
				final int apos = sb.pos(r);
				final int alen = sb.size(r) + apos;
				final int[] aix = sb.indexes(r);
				final int idx = Arrays.binarySearch(aix, apos, alen, col);
				if(idx < 0) {
					map.increment(0.0);
				}
				else {
					map.increment(Math.floor(sb.values(r)[idx] * scaleFactor));
				}
			}
		}
	}

	private void countElementsSparse(DoubleCountHashMap map, int col) {
		final SparseBlock sb = in.getSparseBlock();

		for(int r = 0; r < nRow; r++) {
			if(sb.isEmpty(r))
				map.increment(0.0);
			else {
				final int apos = sb.pos(r);
				final int alen = sb.size(r) + apos;
				final int[] aix = sb.indexes(r);
				final int idx = Arrays.binarySearch(aix, apos, alen, col);
				if(idx < 0)
					map.increment(0.0);
				else
					map.increment(sb.values(r)[idx]);
			}
		}
	}

	private void countElementsDenseContiguousScaled(DoubleCountHashMap map, int col) {
		final double[] dv = in.getDenseBlockValues();
		int off = col;

		double scaleFactor = cs.scaleFactors[0];
		for(int r = 0; r < nRow; r++, off += nCol) {
			map.increment(Math.floor(dv[off] * scaleFactor));
		}
	}

	private void countElementsDenseContiguous(DoubleCountHashMap map, int col) {
		final double[] dv = in.getDenseBlockValues();
		int off = col;

		for(int r = 0; r < nRow; r++, off += nCol)
			map.increment(dv[off]);
	}

	private void countElementsDenseGenericScaled(DoubleCountHashMap map, int col) {
		final DenseBlock db = in.getDenseBlock();
		double scaleFactor = cs.scaleFactors[0];
		for(int r = 0; r < nRow; r++) {
			final double[] dv = db.values(r);
			int off = db.pos(r) + col;
			map.increment(Math.floor(dv[off] * scaleFactor));
		}
	}

	private void countElementsDenseGeneric(DoubleCountHashMap map, int col) {
		final DenseBlock db = in.getDenseBlock();
		for(int r = 0; r < nRow; r++) {
			final double[] dv = db.values(r);
			int off = db.pos(r) + col;
			map.increment(dv[off]);
		}
	}

	private AColGroup directCompressDDC(IColIndex colIndexes, CompressedSizeInfoColGroup cg) throws Exception {
		// testing multicol
		if(colIndexes.size() > 1) {
			return directCompressDDCMultiCol(colIndexes, cg);
		}
		else {
			return directCompressDDCSingleCol(colIndexes, cg);
		}
	}

	private AColGroup directCompressDDCSingleCol(IColIndex colIndexes, CompressedSizeInfoColGroup cg) {
		final int col = colIndexes.get(0);
		final AMapToData d = MapToFactory.create(nRow, Math.max(Math.min(cg.getNumOffs() + 1, nRow), 126));
		final DoubleCountHashMap map = new DoubleCountHashMap(cg.getNumVals());

		// unlike multi-col no special handling of zero entries are needed.
		if(cs.transposed)
			if(cs.scaleFactors != null) {
				throw new NotImplementedException(); // TODO: Handle scaled transposed columns
			}
			else {
				readToMapDDCTransposed(col, map, d);
			}
		else {
			if(cs.scaleFactors != null) {
				readToMapDDCScaled(col, map, d);
			}
			else {
				readToMapDDC(col, map, d);
			}
		}

		if(map.size() == 0)
			return new ColGroupEmpty(colIndexes);
		IDictionary dict = DictionaryFactory.create(map);

		final int nUnique = map.size();
		final AMapToData resData = d.resize(nUnique);
		return ColGroupDDC.create(colIndexes, dict, resData, null);
	}

	private AColGroup directCompressDDCMultiCol(IColIndex colIndexes, CompressedSizeInfoColGroup cg) throws Exception {
		final AMapToData d = MapToFactory.create(nRow, Math.max(Math.min(cg.getNumOffs() + 1, nRow), 126));
		final int fill = d.getUpperBoundValue();
		d.fill(fill);

		final DblArrayCountHashMap map = new DblArrayCountHashMap(Math.max(cg.getNumVals(), 64));
		boolean extra;
		if(nRow < CompressionSettings.PAR_DDC_THRESHOLD || k < csi.getNumberColGroups() || pool == null) {
			extra = readToMapDDC(colIndexes, map, d, 0, nRow, fill);
		}
		else {
			extra = parallelReadToMapDDC(colIndexes, map, d, nRow, fill, k);
		}

		if(map.size() == 0)
			// If the column was empty.
			// This is highly unlikely but could happen if forced compression of
			// not transposed column and the estimator says use DDC.
			return new ColGroupEmpty(colIndexes);

		IDictionary dict = DictionaryFactory.create(map, colIndexes.size(), extra, cg.getTupleSparsity());

		if(extra)
			d.replace(fill, map.size());
		final int nUnique = map.size() + (extra ? 1 : 0);
		final AMapToData resData = d.resize(nUnique);
		return ColGroupDDC.create(colIndexes, dict, resData, null);
	}

	private AColGroup directCompressDeltaDDC(IColIndex colIndexes, CompressedSizeInfoColGroup cg) throws Exception {
		if(cs.transposed) {
			throw new NotImplementedException("Delta encoding for transposed matrices not yet implemented");
		}
		if(cs.scaleFactors != null) {
			throw new NotImplementedException("Delta encoding with quantization not yet implemented");
		}
		
		if(colIndexes.size() > 1) {
			return directCompressDeltaDDCMultiCol(colIndexes, cg);
		}
		else {
			return directCompressDeltaDDCSingleCol(colIndexes, cg);
		}
	}

	private AColGroup directCompressDeltaDDCSingleCol(IColIndex colIndexes, CompressedSizeInfoColGroup cg) {
		final int col = colIndexes.get(0);
		final AMapToData d = MapToFactory.create(nRow, Math.max(Math.min(cg.getNumOffs() + 1, nRow), 126));
		final DoubleCountHashMap map = new DoubleCountHashMap(cg.getNumVals());

		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(in, colIndexes, cs.transposed, 0, nRow);
		DblArray cellVals = reader.nextRow();
		int r = 0;
		while(r < nRow && cellVals != null) {
			final int row = reader.getCurrentRowIndex();
			if(row == r) {
				final double val = cellVals.getData()[0];
				final int id = map.increment(val);
				d.set(row, id);
				cellVals = reader.nextRow();
				r++;
			}
			else {
				r = row;
			}
		}

		if(map.size() == 0)
			return new ColGroupEmpty(colIndexes);
		
		final double[] dictValues = map.getDictionary();
		IDictionary dict = new DeltaDictionary(dictValues, 1);

		final int nUnique = map.size();
		final AMapToData resData = d.resize(nUnique);
		return ColGroupDeltaDDC.create(colIndexes, dict, resData, null);
	}

	private AColGroup directCompressDeltaDDCMultiCol(IColIndex colIndexes, CompressedSizeInfoColGroup cg) throws Exception {
		final AMapToData d = MapToFactory.create(nRow, Math.max(Math.min(cg.getNumOffs() + 1, nRow), 126));
		final int fill = d.getUpperBoundValue();
		d.fill(fill);

		final DblArrayCountHashMap map = new DblArrayCountHashMap(Math.max(cg.getNumVals(), 64));
		boolean extra;
		if(nRow < CompressionSettings.PAR_DDC_THRESHOLD || k < csi.getNumberColGroups() || pool == null) {
			extra = readToMapDeltaDDC(colIndexes, map, d, 0, nRow, fill);
		}
		else {
			throw new NotImplementedException("Parallel delta DDC compression not yet implemented");
		}

		if(map.size() == 0)
			return new ColGroupEmpty(colIndexes);

		final ACount<DblArray>[] vals = map.extractValues();
		final int nVals = vals.length;
		final int nTuplesOut = nVals + (extra ? 1 : 0);
		final double[] dictValues = new double[nTuplesOut * colIndexes.size()];
		final int[] oldIdToNewId = new int[map.size()];
		int idx = 0;
		for(int i = 0; i < nVals; i++) {
			final ACount<DblArray> dac = vals[i];
			final double[] arrData = dac.key().getData();
			System.arraycopy(arrData, 0, dictValues, idx, colIndexes.size());
			oldIdToNewId[dac.id] = i;
			idx += colIndexes.size();
		}
		IDictionary dict = new DeltaDictionary(dictValues, colIndexes.size());

		if(extra)
			d.replace(fill, map.size());
		final int nUnique = map.size() + (extra ? 1 : 0);
		final AMapToData resData = d.resize(nUnique);
		for(int i = 0; i < nRow; i++) {
			final int oldId = resData.getIndex(i);
			if(extra && oldId == map.size()) {
				resData.set(i, nVals);
			}
			else if(oldId < oldIdToNewId.length) {
				resData.set(i, oldIdToNewId[oldId]);
			}
		}
		return ColGroupDeltaDDC.create(colIndexes, dict, resData, null);
	}

	private boolean readToMapDeltaDDC(IColIndex colIndexes, DblArrayCountHashMap map, AMapToData data, int rl, int ru,
		int fill) {
		ReaderColumnSelection reader = ReaderColumnSelection.createDeltaReader(in, colIndexes, cs.transposed, rl, ru);

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

	private boolean readToMapDDC(IColIndex colIndexes, DblArrayCountHashMap map, AMapToData data, int rl, int ru,
		int fill) {

		ReaderColumnSelection reader = (cs.scaleFactors == null) ? ReaderColumnSelection.createReader(in, colIndexes,
			cs.transposed, rl,
			ru) : ReaderColumnSelection.createQuantizedReader(in, colIndexes, cs.transposed, rl, ru, cs.scaleFactors);

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

	// TODO: Merge logic to readToMapDDC. This should be done for other scaled methods
	private void readToMapDDCScaled(int col, DoubleCountHashMap map, AMapToData data) {
		double scaleFactor = cs.scaleFactors[0];

		if(in.isInSparseFormat()) {
			// not good but could happen
			final SparseBlock sb = in.getSparseBlock();
			for(int r = 0; r < nRow; r++) {
				if(sb.isEmpty(r))
					data.set(r, map.increment(0.0));
				else {
					final int apos = sb.pos(r);
					final int alen = sb.size(r) + apos;
					final int[] aix = sb.indexes(r);
					final int idx = Arrays.binarySearch(aix, apos, alen, col);
					if(idx < 0)
						data.set(r, map.increment(0.0));
					else {
						double scaledValue = Math.floor(sb.values(r)[idx] * scaleFactor);
						data.set(r, map.increment(scaledValue));
					}
				}
			}
		}
		else if(in.getDenseBlock().isContiguous()) {
			final double[] dv = in.getDenseBlockValues();
			int off = col;
			for(int r = 0; r < nRow; r++, off += nCol) {
				double scaledValue = Math.floor(dv[off] * scaleFactor);
				data.set(r, map.increment(scaledValue));
			}
		}
		else {
			final DenseBlock db = in.getDenseBlock();
			for(int r = 0; r < nRow; r++) {
				final double[] dv = db.values(r);
				int off = db.pos(r) + col;
				double scaledValue = Math.floor(dv[off] * scaleFactor);
				data.set(r, map.increment(scaledValue));
			}
		}
	}

	private void readToMapDDC(int col, DoubleCountHashMap map, AMapToData data) {
		if(in.isInSparseFormat()) {
			// not good but could happen
			final SparseBlock sb = in.getSparseBlock();
			for(int r = 0; r < nRow; r++) {
				if(sb.isEmpty(r))
					data.set(r, map.increment(0.0));
				else {
					final int apos = sb.pos(r);
					final int alen = sb.size(r) + apos;
					final int[] aix = sb.indexes(r);
					final int idx = Arrays.binarySearch(aix, apos, alen, col);
					if(idx < 0)
						data.set(r, map.increment(0.0));
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
			final DenseBlock db = in.getDenseBlock();
			for(int r = 0; r < nRow; r++) {
				final double[] dv = db.values(r);
				int off = db.pos(r) + col;
				data.set(r, map.increment(dv[off]));
			}
		}
	}

	private void readToMapDDCTransposed(int col, DoubleCountHashMap map, AMapToData data) {
		if(in.isInSparseFormat()) {
			final SparseBlock sb = in.getSparseBlock();
			if(sb.isEmpty(col))
				throw new DMLCompressionException("Empty column in DDC compression");

			final int apos = sb.pos(col);
			final int alen = sb.size(col) + apos;
			final int[] aix = sb.indexes(col);
			final double[] aval = sb.values(col);
			// count zeros
			if(nRow > alen - apos)
				map.increment(0.0, nRow - apos - alen);
			// insert all other counts
			for(int j = apos; j < alen; j++) {
				final int id = map.increment(aval[j]);
				data.set(aix[j], id);
			}
		}
		else {
			final DenseBlock db = in.getDenseBlock();
			final double[] dv = db.values(col);
			int off = db.pos(col);
			for(int r = 0; r < nRow; r++, off++)
				data.set(r, map.increment(dv[off]));
		}
	}

	private boolean parallelReadToMapDDC(IColIndex colIndexes, DblArrayCountHashMap map, AMapToData data, int rlen,
		int fill, int k) throws Exception {

		final int blk = Math.max(rlen / colIndexes.size() / k, 64000 / colIndexes.size());

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

	private static AColGroup compressSDC(IColIndex colIndexes, int rlen, ABitmap ubm, CompressionSettings cs,
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
		final int nVal = ubm.getNumValues();

		// Currently not effecient allocation of the dictionary.
		if(nVal == 1 && numZeros >= largestOffset) {
			IDictionary dict = DictionaryFactory.create(ubm, tupleSparsity);
			final AOffset off = OffsetFactory.createOffset(ubm.getOffsetList()[0].extractValues(true));
			return ColGroupSDCSingleZeros.create(colIndexes, rlen, dict, off, null);
		}
		else if((nVal == 2 && numZeros == 0) // case 1 : two distinct non zero values
			|| (nVal == 1 && numZeros < largestOffset) // case 2: 1 non zero value more frequent than zero.
		) {
			double[] defaultTuple = new double[colIndexes.size()];
			IDictionary dict = DictionaryFactory.create(ubm, largestIndex, defaultTuple, tupleSparsity, numZeros > 0);
			return compressSDCSingle(colIndexes, rlen, ubm, largestIndex, dict, defaultTuple);
		}
		else if(numZeros >= largestOffset) {
			IDictionary dict = DictionaryFactory.create(ubm, tupleSparsity);
			return compressSDCZero(colIndexes, rlen, ubm, dict, cs);
		}
		else
			return compressSDCNormal(colIndexes, numZeros, rlen, ubm, largestIndex, tupleSparsity, cs);

	}

	private static AColGroup compressSDCZero(IColIndex colIndexes, int rlen, ABitmap ubm, IDictionary dict,
		CompressionSettings cs) {
		IntArrayList[] offsets = ubm.getOffsetList();
		AInsertionSorter s = InsertionSorterFactory.create(rlen, offsets, cs.sdcSortType);
		AOffset indexes = OffsetFactory.createOffset(s.getIndexes());
		AMapToData data = s.getData();
		data = data.resize(dict.getNumberOfValues(colIndexes.size()));
		return ColGroupSDCZeros.create(colIndexes, rlen, dict, indexes, data, null);
	}

	private static AColGroup compressSDCNormal(IColIndex colIndexes, int numZeros, int rlen, ABitmap ubm,
		int largestIndex, double tupleSparsity, CompressionSettings cs) {
		final double[] defaultTuple = new double[colIndexes.size()];
		final IDictionary dict = DictionaryFactory.create(ubm, largestIndex, defaultTuple, tupleSparsity, numZeros > 0);
		AInsertionSorter s = InsertionSorterFactory.createNegative(rlen, ubm.getOffsetList(), largestIndex,
			cs.sdcSortType);
		AOffset indexes = OffsetFactory.createOffset(s.getIndexes());
		AMapToData _data = s.getData();
		_data = _data.resize(dict.getNumberOfValues(colIndexes.size()));
		return ColGroupSDC.create(colIndexes, rlen, dict, defaultTuple, indexes, _data, null);
	}

	private static AColGroup compressSDCSingle(IColIndex colIndexes, int rlen, ABitmap ubm, int largestIndex,
		IDictionary dict, double[] defaultTuple) {
		if(ubm.getOffsetList().length > 1) {
			// flipping first bit is same as saying index 1 if zero else index 0 if one or !
			AOffset off = OffsetFactory.createOffset(ubm.getOffsetsList(largestIndex ^ 1));
			return ColGroupSDCSingle.create(colIndexes, rlen, dict, defaultTuple, off, null);
		}
		else {
			IntArrayList inv = ubm.getOffsetsList(0);
			int[] indexes = new int[rlen - inv.size()];
			int p = 0;
			int v = 0;
			for(int i = 0; i < inv.size(); i++) {
				int j = inv.get(i);
				while(v < j)
					indexes[p++] = v++;
				v++;
			}

			while(v < rlen)
				indexes[p++] = v++;
			AOffset off = OffsetFactory.createOffset(indexes);

			return ColGroupSDCSingle.create(colIndexes, rlen, dict, defaultTuple, off, null);
		}
	}

	private static AColGroup compressLinearFunctional(IColIndex colIndexes, MatrixBlock in, CompressionSettings cs) {
		double[] coefficients = LinearRegression.regressMatrixBlock(in, colIndexes, cs.transposed);
		int numRows = cs.transposed ? in.getNumColumns() : in.getNumRows();
		return ColGroupLinearFunctional.create(colIndexes, coefficients, numRows);
	}

	private AColGroup compressSDCFromSparseTransposedBlock(IColIndex cols, int nrUniqueEstimate, double tupleSparsity) {
		if(cols.size() > 1)
			return compressMultiColSDCFromSparseTransposedBlock(cols, nrUniqueEstimate, tupleSparsity);
		else
			return compressSingleColSDCFromSparseTransposedBlock(cols, nrUniqueEstimate);
	}

	private AColGroup compressMultiColSDCFromSparseTransposedBlock(IColIndex cols, int nrUniqueEstimate,
		double tupleSparsity) {

		HashSet<Integer> offsetsSet = new HashSet<>();

		SparseBlock sb = in.getSparseBlock();

		for(int i = 0; i < cols.size(); i++) {
			final int idx = cols.get(i);
			if(sb.isEmpty(idx))
				continue;

			final int apos = sb.pos(idx);
			final int alen = sb.size(idx) + apos;
			final int[] aix = sb.indexes(idx);
			for(int j = apos; j < alen; j++)
				offsetsSet.add(aix[j]);
		}

		if(offsetsSet.isEmpty())
			return new ColGroupEmpty(cols);

		int[] offsetsInt = offsetsSet.stream().mapToInt(Number::intValue).toArray();
		Arrays.sort(offsetsInt);

		MatrixBlock sub = new MatrixBlock(offsetsInt.length, cols.size(), false);
		sub.allocateDenseBlock();
		sub.setNonZeros(offsetsInt.length * cols.size());
		double[] subV = sub.getDenseBlockValues();

		for(int i = 0; i < cols.size(); i++) {
			final int idx = cols.get(i);
			if(sb.isEmpty(idx))
				continue;
			final int apos = sb.pos(idx);
			final int alen = sb.size(idx) + apos;
			final int[] aix = sb.indexes(idx);
			final double[] aval = sb.values(idx);
			int offsetsPos = 0;
			for(int j = apos; j < alen; j++) {
				while(offsetsInt[offsetsPos] < aix[j])
					offsetsPos++;
				if(offsetsInt[offsetsPos] == aix[j])
					subV[offsetsPos * cols.size() + i] = aval[j];
			}
		}
		IColIndex subCols = ColIndexFactory.create(cols.size());
		ReaderColumnSelection reader = (cs.scaleFactors == null) ? ReaderColumnSelection.createReader(sub, subCols,
			false) : ReaderColumnSelection.createQuantizedReader(sub, subCols, false, cs.scaleFactors);

		final int mapStartSize = Math.min(nrUniqueEstimate, offsetsInt.length / 2);
		DblArrayCountHashMap map = new DblArrayCountHashMap(mapStartSize);

		DblArray cellVals = null;
		AMapToData data = MapToFactory.create(offsetsInt.length, 257);

		while((cellVals = reader.nextRow()) != null) {
			final int row = reader.getCurrentRowIndex();
			data.set(row, map.increment(cellVals));
		}

		IDictionary dict = DictionaryFactory.create(map, cols.size(), false, tupleSparsity);
		data = data.resize(map.size());

		AOffset offs = OffsetFactory.createOffset(offsetsInt);
		return ColGroupSDCZeros.create(cols, in.getNumColumns(), dict, offs, data, null);
	}

	private AColGroup compressSingleColSDCFromSparseTransposedBlock(IColIndex cols, int nrUniqueEstimate) {

		// This method should only be called if the cols argument is length 1.
		final SparseBlock sb = in.getSparseBlock();
		if(sb.isEmpty(cols.get(0)))
			return new ColGroupEmpty(cols);

		final int sbRow = cols.get(0);
		final int apos = sb.pos(sbRow);
		final int alen = sb.size(sbRow) + apos;
		final double[] vals = sb.values(sbRow);
		final DoubleCountHashMap map = new DoubleCountHashMap(nrUniqueEstimate);

		// count distinct items frequencies
		for(int j = apos; j < alen; j++)
			map.increment(vals[j]);

		ACount<Double>[] entries = map.extractValues();
		Arrays.sort(entries, Comparator.comparing(x -> -x.count));

		if(entries[0].count < nRow - sb.size(sbRow)) {
			// If the zero is the default value.
			final int[] counts = new int[entries.length];
			final double[] dict = new double[entries.length];
			for(int i = 0; i < entries.length; i++) {
				final ACount<Double> x = entries[i];
				counts[i] = x.count;
				dict[i] = x.key();
				x.count = i;
			}

			final AOffset offsets = OffsetFactory.createOffset(sb.indexes(sbRow), apos, alen);
			if(entries.length <= 1)
				return ColGroupSDCSingleZeros.create(cols, nRow, Dictionary.create(dict), offsets, counts);
			else {
				final AMapToData mapToData = MapToFactory.create((alen - apos), entries.length);
				for(int j = apos; j < alen; j++)
					mapToData.set(j - apos, map.get(vals[j]));
				return ColGroupSDCZeros.create(cols, nRow, Dictionary.create(dict), offsets, mapToData, counts);
			}
		}
		else if(entries.length == 1) {
			// SDCSingle and we know all values are x or 0
			final int nonZeros = nRow - entries[0].count;
			final double x = entries[0].key();
			final double[] defaultTuple = new double[] {x};
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

			return ColGroupSDCSingle.create(cols, nRow, null, defaultTuple, offsets, counts);
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
		public Object call() throws Exception {
			for(int i = _off; i < _groups.size(); i += _step)
				_ret[i] = compressColGroup(_groups.get(i));
			return null;
		}
	}

	private class readToMapDDCTask implements Callable<Boolean> {
		private final IColIndex _colIndexes;
		private final DblArrayCountHashMap _map;
		private final AMapToData _data;
		private final int _rl;
		private final int _ru;
		private final int _fill;

		protected readToMapDDCTask(IColIndex colIndexes, DblArrayCountHashMap map, AMapToData data, int rl, int ru,
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
