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

package org.apache.sysds.runtime.transform.encode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressedArray;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IdentityDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.estim.sample.SampleEstimatorFactory;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ACompressedArray;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.DDCArray;
import org.apache.sysds.runtime.frame.data.columns.HashMapToInt;
import org.apache.sysds.runtime.frame.data.compress.ArrayCompressionStatistics;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderBin.BinMethod;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.stats.Timing;

public class CompressedEncode {
	protected static final Log LOG = LogFactory.getLog(CompressedEncode.class.getName());

	/** Row parallelization threshold for parallel creation of AMapToData for compression */
	public static int ROW_PARALLELIZATION_THRESHOLD = 10000;

	/** The encoding scheme plan */
	private final MultiColumnEncoder enc;
	/** The Input FrameBlock */
	private final FrameBlock in;
	/** The thread count of the instruction */
	private final int k;

	/** the Executor pool for parallel tasks of this encoder. */
	private final ExecutorService pool;

	private final boolean inputContainsCompressed;

	private final AtomicLong nnz = new AtomicLong();

	private CompressedEncode(MultiColumnEncoder enc, FrameBlock in, int k) {
		this.enc = enc;
		this.in = in;
		this.k = k;
		this.pool = k > 1 && CommonThreadPool.useParallelismOnThread() ? CommonThreadPool.get(k) : null;
		this.inputContainsCompressed = containsCompressed(in);
	}

	private boolean containsCompressed(FrameBlock in) {
		for(Array<?> c : in.getColumns())
			if(c instanceof ACompressedArray)
				return true;
		return false;
	}

	public static MatrixBlock encode(MultiColumnEncoder enc, FrameBlock in, int k) throws Exception {
		return new CompressedEncode(enc, in, k).apply();
	}

	private MatrixBlock apply() throws Exception {
		try {
			final List<ColumnEncoderComposite> encoders = enc.getColumnEncoders();
			final List<AColGroup> groups = isParallel() ? multiThread(encoders) : singleThread(encoders);
			final int cols = shiftGroups(groups);
			final CompressedMatrixBlock mb = new CompressedMatrixBlock(in.getNumRows(), cols, -1, false, groups);

			combineUncompressed(mb);
			mb.setNonZeros(nnz.get());
			logging(mb);
			return mb;
		}
		finally {
			if(pool != null)
				pool.shutdown();
		}
	}

	private boolean isParallel() {
		return pool != null;
	}

	private List<AColGroup> singleThread(List<ColumnEncoderComposite> encoders) throws Exception {
		List<AColGroup> groups = new ArrayList<>(encoders.size());
		for(ColumnEncoderComposite c : encoders)
			groups.add(encode(c));
		return groups;
	}

	private List<AColGroup> multiThread(List<ColumnEncoderComposite> encoders) throws Exception {
		final List<Future<AColGroup>> tasks = new ArrayList<>(encoders.size());
		for(ColumnEncoderComposite c : encoders)
			tasks.add(pool.submit(() -> encode(c)));
		final List<AColGroup> groups = new ArrayList<>(encoders.size());
		for(Future<AColGroup> t : tasks)
			groups.add(t.get());
		return groups;
	}

	/**
	 * Shift the column groups to the correct column numbers.
	 * 
	 * @param groups the groups to shift
	 * @return The total number of columns contained.
	 */
	private int shiftGroups(List<AColGroup> groups) {
		int cols = groups.get(0).getColIndices().size();
		for(int i = 1; i < groups.size(); i++) {
			groups.set(i, groups.get(i).shiftColIndices(cols));
			cols += groups.get(i).getColIndices().size();
		}
		return cols;
	}

	private AColGroup encode(ColumnEncoderComposite c) throws Exception {
		final Timing t = new Timing();
		AColGroup g = executeEncode(c);
		if(LOG.isDebugEnabled())
			LOG.debug(String.format("Encode: columns: %4d estimateDistinct: %6d distinct: %6d size: %6d time: %10f",
				c._colID, c._estNumDistincts, g.getNumValues(), g.estimateInMemorySize(), t.stop()));
		return g;
	}

	private AColGroup executeEncode(ColumnEncoderComposite c) throws Exception {
		if(c.isRecodeToDummy())
			return recodeToDummy(c);
		else if(c.isRecode())
			return recode(c);
		else if(c.isPassThrough())
			return passThrough(c);
		else if(c.isBin())
			return bin(c);
		else if(c.isBinToDummy())
			return binToDummy(c);
		else if(c.isHash())
			return hash(c);
		else if(c.isHashToDummy())
			return hashToDummy(c);
		else
			throw new NotImplementedException("Not supporting : " + c);
	}

	@SuppressWarnings("unchecked")
	private <T> AColGroup recodeToDummy(ColumnEncoderComposite c) throws Exception {
		int colId = c._colID;
		Array<T> a = (Array<T>) in.getColumn(colId - 1);
		boolean containsNull = a.containsNull();
		estimateRCDMapSize(c);
		HashMapToInt<T> map = (HashMapToInt<T>) a.getRecodeMap(c._estNumDistincts, pool, k / in.getNumColumns());

		List<ColumnEncoder> r = c.getEncoders();
		r.set(0, new ColumnEncoderRecode(colId, (HashMapToInt<Object>) map));
		int domain = map.size();
		if(containsNull && domain == 0)
			return new ColGroupEmpty(ColIndexFactory.create(1));
		IColIndex colIndexes = ColIndexFactory.create(0, domain);
		if(domain == 1 && !containsNull) {
			nnz.addAndGet(in.getNumRows());
			return ColGroupConst.create(colIndexes, new double[] {1});
		}

		ADictionary d = new IdentityDictionary(colIndexes.size(), containsNull);
		AMapToData m = createMappingAMapToData(a, map, containsNull);
		AColGroup ret = ColGroupDDC.create(colIndexes, d, m, null);
		nnz.addAndGet(ret.getNumberNonZeros(in.getNumRows()));
		return ret;
	}

	private AColGroup bin(ColumnEncoderComposite c) throws InterruptedException, ExecutionException {
		final int colId = c._colID;
		final Array<?> a = in.getColumn(colId - 1);
		final List<ColumnEncoder> r = c.getEncoders();
		final ColumnEncoderBin b = (ColumnEncoderBin) r.get(0);
		b.build(in);
		final boolean containsNull = b.containsNull;
		final IColIndex colIndexes = ColIndexFactory.create(1);

		ADictionary d = createIncrementingVector(b._numBin, containsNull);
		final AMapToData m;
		m = binEncode(a, b, containsNull);

		AColGroup ret = ColGroupDDC.create(colIndexes, d, m, null);
		nnz.addAndGet(ret.getNumberNonZeros(in.getNumRows()));
		return ret;
	}

	private AMapToData binEncode(Array<?> a, ColumnEncoderBin b, boolean nulls)
		throws InterruptedException, ExecutionException {
		final AMapToData m = MapToFactory.create(a.size(), b._numBin + (nulls ? 1 : 0));

		if(!nulls && b.getBinMethod() == BinMethod.EQUI_WIDTH) {
			final double min = b.getBinMins()[0];
			final double max = b.getBinMaxs()[b.getNumBin() - 1];
			if(Util.eq(max, min)) {
				m.fill(0);
				return m;
			}
			if(b._numBin <= 0)
				throw new RuntimeException("Invalid num bins");
		}

		final int rlen = a.size();
		if(k / in.getNumColumns() > 1 && rlen > ROW_PARALLELIZATION_THRESHOLD) {
			BinEncodeParallel(a, b, nulls, m, rlen);
		}
		else {
			if(nulls)
				binEncodeWithNulls(a, b, m, 0, a.size());
			else
				binEncodeNoNull(a, b, m, 0, a.size());

		}
		return m;
	}

	private void BinEncodeParallel(Array<?> a, ColumnEncoderBin b, boolean nulls, final AMapToData m, final int rlen)
		throws InterruptedException, ExecutionException {
		final List<Future<?>> tasks = new ArrayList<>();
		final int tk = k / in.getNumColumns();
		final int blockSize = Math.max(ROW_PARALLELIZATION_THRESHOLD / 2, rlen + tk / tk);

		for(int i = 0; i < rlen; i += blockSize) {
			final int start = i;
			final int end = Math.min(rlen, i + blockSize);
			tasks.add(pool.submit(() -> {
				if(nulls)
					binEncodeWithNulls(a, b, m, start, end);
				else
					binEncodeNoNull(a, b, m, start, end);
			}));
		}
		for(Future<?> t : tasks)
			t.get();

	}

	private void binEncodeWithNulls(Array<?> a, ColumnEncoderBin b, AMapToData m, int l, int u) {
		for(int i = l; i < u; i++) {
			final double v = a.getAsNaNDouble(i);
			if(Double.isNaN(v))
				m.set(i, b._numBin);
			else {
				int idx = (int) b.getCodeIndex(v) - 1;
				if(idx < 0)
					idx = 0;
				m.set(i, idx);
			}

		}
	}

	private final void binEncodeNoNull(Array<?> a, ColumnEncoderBin b, AMapToData m, int l, int u) {
		if(b.getBinMethod() == BinMethod.EQUI_WIDTH)
			binEncodeNoNullEqWidth(a, b, m, l, u);
		else
			binEncodeNoNullGeneric(a, b, m, l, u);
	}

	private final void binEncodeNoNullEqWidth(Array<?> a, ColumnEncoderBin b, AMapToData m, int l, int u) {
		final double min = b.getBinMins()[0];
		final double max = b.getBinMaxs()[b.getNumBin() - 1];
		for(int i = l; i < u; i++) {
			m.set(i, b.getEqWidthUnsafe(a.getAsDouble(i), min, max) - 1);
		}
	}

	private final void binEncodeNoNullGeneric(Array<?> a, ColumnEncoderBin b, AMapToData m, int l, int u) {
		final double min = b.getBinMins()[0];
		final double max = b.getBinMaxs()[b.getNumBin() - 1];
		for(int i = l; i < u; i++) {
			m.set(i, (int) b.getCodeIndex(a.getAsDouble(i), min, max) - 1);
		}
	}

	private MatrixBlockDictionary createIncrementingVector(int nVals, boolean NaN) {
		MatrixBlock bins = new MatrixBlock(nVals + (NaN ? 1 : 0), 1, false);
		for(int i = 0; i < nVals; i++)
			bins.set(i, 0, i + 1);
		if(NaN)
			bins.set(nVals, 0, Double.NaN);
		return MatrixBlockDictionary.create(bins);
	}

	private AColGroup binToDummy(ColumnEncoderComposite c) throws InterruptedException, ExecutionException {
		final int colId = c._colID;
		final Array<?> a = in.getColumn(colId - 1);
		final List<ColumnEncoder> r = c.getEncoders();
		final ColumnEncoderBin b = (ColumnEncoderBin) r.get(0);
		b.build(in); // build first since we figure out if it contains null here.
		final boolean containsNull = b.containsNull;
		IColIndex colIndexes = ColIndexFactory.create(0, b._numBin);
		ADictionary d = new IdentityDictionary(colIndexes.size(), containsNull);
		final AMapToData m;
		m = binEncode(a, b, containsNull);
		AColGroup ret = ColGroupDDC.create(colIndexes, d, m, null);
		nnz.addAndGet(ret.getNumberNonZeros(in.getNumRows()));
		return ret;
	}

	@SuppressWarnings("unchecked")
	private <T> AColGroup recode(ColumnEncoderComposite c) throws Exception {
		int colId = c._colID;
		Array<T> a = (Array<T>) in.getColumn(colId - 1);
		estimateRCDMapSize(c);
		HashMapToInt<T> map = (HashMapToInt<T>) a.getRecodeMap(c._estNumDistincts, pool, k / in.getNumColumns());
		boolean containsNull = a.containsNull();
		int domain = map.size();

		// int domain = c.getDomainSize();
		IColIndex colIndexes = ColIndexFactory.create(1);
		if(domain == 0 && containsNull) {
			return new ColGroupEmpty(colIndexes);
		}
		if(domain == 1 && !containsNull) {
			nnz.addAndGet(in.getNumRows());
			return ColGroupConst.create(colIndexes, new double[] {1});
		}
		ADictionary d = createRecodeDictionary(containsNull, domain);

		AMapToData m = createMappingAMapToData(a, map, containsNull);

		List<ColumnEncoder> r = c.getEncoders();
		r.set(0, new ColumnEncoderRecode(colId, (HashMapToInt<Object>) map));
		AColGroup ret = ColGroupDDC.create(colIndexes, d, m, null);

		nnz.addAndGet(ret.getNumberNonZeros(in.getNumRows()));
		return ret;
	}

	private ADictionary createRecodeDictionary(boolean containsNull, int domain) {
		MatrixBlock incrementing = new MatrixBlock(domain + (containsNull ? 1 : 0), 1, false);
		for(int i = 0; i < domain; i++)
			incrementing.set(i, 0, i + 1);
		if(containsNull)
			incrementing.set(domain, 0, Double.NaN);

		ADictionary d = MatrixBlockDictionary.create(incrementing);
		return d;
	}

	@SuppressWarnings("unchecked")
	private <T> AColGroup passThrough(ColumnEncoderComposite c) throws Exception {

		final IColIndex colIndexes = ColIndexFactory.create(1);
		final int colId = c._colID;
		final Array<T> a = (Array<T>) in.getColumn(colId - 1);
		if(a instanceof ACompressedArray)
			return passThroughCompressed(colIndexes, a);
		else
			return passThroughNormal(c, colIndexes, a);
	}

	private <T> AColGroup passThroughNormal(ColumnEncoderComposite c, final IColIndex colIndexes, final Array<T> a)
		throws InterruptedException, ExecutionException, Exception {
		// Take a small sample
		ArrayCompressionStatistics stats = !inputContainsCompressed ? //
			a.statistics(Math.min(1000, a.size())) : null;

		if(a.getValueType() != ValueType.BOOLEAN // if not booleans
			&& (stats == null || !stats.shouldCompress || stats.valueType != a.getValueType())) {
			LOG.error("Create temporary Uncompressed ColumnGroupArray");
			return new ColGroupUncompressedArray(a, c._colID - 1, colIndexes);
		}
		else {
			boolean containsNull = a.containsNull();
			estimateRCDMapSize(c);
			HashMapToInt<T> map = (HashMapToInt<T>) a.getRecodeMap(c._estNumDistincts, pool, k / in.getNumColumns());
			double[] vals = new double[map.size() + (containsNull ? 1 : 0)];
			if(containsNull)
				vals[map.size()] = Double.NaN;
			ValueType t = a.getValueType();
			map.forEach((k, v) -> vals[v.intValue() - 1] = UtilFunctions.objectToDouble(t, k));
			ADictionary d = Dictionary.create(vals);
			AMapToData m = createMappingAMapToData(a, map, containsNull);
			AColGroup ret = ColGroupDDC.create(colIndexes, d, m, null);
			nnz.addAndGet(ret.getNumberNonZeros(in.getNumRows()));
			return ret;
		}
	}

	private <T> AColGroup passThroughCompressed(final IColIndex colIndexes, final Array<T> a) {
		// only DDC possible currently.
		DDCArray<?> aDDC = (DDCArray<?>) a;
		Array<?> dict = aDDC.getDict();
		double[] vals = new double[dict.size()];
		if(a.containsNull())
			for(int i = 0; i < dict.size(); i++)
				vals[i] = dict.getAsNaNDouble(i);
		else
			for(int i = 0; i < dict.size(); i++)
				vals[i] = dict.getAsDouble(i);

		ADictionary d = Dictionary.create(vals);
		AColGroup ret = ColGroupDDC.create(colIndexes, d, aDDC.getMap(), null);

		nnz.addAndGet(ret.getNumberNonZeros(in.getNumRows()));
		return ret;
	}

	private <T> AMapToData createMappingAMapToData(Array<T> a, HashMapToInt<T> map, boolean containsNull)
		throws Exception {
		final int si = map.size();
		final int nRow = in.getNumRows();
		if(!containsNull && a instanceof DDCArray)
			return ((DDCArray<?>) a).getMap();

		final AMapToData m = MapToFactory.create(nRow, si + (containsNull ? 1 : 0));

		if(k / in.getNumColumns() > 1 && nRow > ROW_PARALLELIZATION_THRESHOLD)
			return CreateMappingParallel(a, map, containsNull, si, nRow, m);
		else
			return createMappingSingleThread(a, map, containsNull, si, nRow, m);
	}

	private <T> AMapToData CreateMappingParallel(Array<T> a, HashMapToInt<T> map, boolean containsNull, final int si,
		final int nRow, final AMapToData m) throws InterruptedException, ExecutionException {
		final int tk = k / in.getNumColumns();
		final int blkz = Math.max(ROW_PARALLELIZATION_THRESHOLD / 2, (nRow + tk) / tk);

		List<Future<?>> tasks = new ArrayList<>();

		for(int i = 0; i < nRow; i += blkz) {
			final int start = i;
			final int end = Math.min(nRow, i + blkz);

			tasks.add(pool.submit(() -> {
				if(containsNull)
					return createMappingAMapToDataWithNull(a, map, si, m, start, end);
				else
					return createMappingAMapToDataNoNull(a, map, m, start, end);

			}));
		}

		for(Future<?> t : tasks)
			t.get();
		return m;

	}

	private <T> AMapToData createMappingSingleThread(Array<T> a, HashMapToInt<T> map, boolean containsNull, final int si,
		final int nRow, final AMapToData m) {
		if(containsNull)
			return createMappingAMapToDataWithNull(a, map, si, m, 0, nRow);
		else
			return createMappingAMapToDataNoNull(a, map, m, 0, nRow);
	}

	private static <T> AMapToData createMappingAMapToDataNoNull(Array<T> a, HashMapToInt<T> map, AMapToData m, int start,
		int end) {
		for(int i = start; i < end; i++)
			a.setM(map, m, i);
		return m;
	}

	private static <T> AMapToData createMappingAMapToDataWithNull(Array<T> a, HashMapToInt<T> map, int si, AMapToData m,
		int start, int end) {
		for(int i = start; i < end; i++)
			a.setM(map, si, m, i);
		return m;
	}

	private AMapToData createHashMappingAMapToData(Array<?> a, int k, boolean nulls) {
		AMapToData m = MapToFactory.create(a.size(), k + (nulls ? 1 : 0));
		if(nulls) {
			for(int i = 0; i < a.size(); i++) {
				double h = Math.abs(a.hashDouble(i)) % k;
				if(Double.isNaN(h))
					m.set(i, k);
				else
					m.set(i, (int) h);
			}
		}
		else {
			for(int i = 0; i < a.size(); i++) {
				double h = Math.abs(a.hashDouble(i)) % k;
				m.set(i, (int) h);
			}
		}
		return m;
	}

	private AColGroup hash(ColumnEncoderComposite c) {
		int colId = c._colID;
		Array<?> a = in.getColumn(colId - 1);
		ColumnEncoderFeatureHash CEHash = (ColumnEncoderFeatureHash) c.getEncoders().get(0);
		int domain = (int) CEHash.getK();
		boolean nulls = a.containsNull();
		IColIndex colIndexes = ColIndexFactory.create(0, 1);
		if(domain == 0 && nulls) {
			return new ColGroupEmpty(colIndexes);
		}
		if(domain == 1 && !nulls) {
			nnz.addAndGet(in.getNumRows());
			return ColGroupConst.create(colIndexes, new double[] {1});
		}

		MatrixBlock incrementing = new MatrixBlock(domain + (nulls ? 1 : 0), 1, false);
		for(int i = 0; i < domain; i++)
			incrementing.set(i, 0, i + 1);
		if(nulls)
			incrementing.set(domain, 0, Double.NaN);

		ADictionary d = MatrixBlockDictionary.create(incrementing);

		AMapToData m = createHashMappingAMapToData(a, domain, nulls);
		AColGroup ret = ColGroupDDC.create(colIndexes, d, m, null);
		nnz.addAndGet(ret.getNumberNonZeros(in.getNumRows()));
		return ret;
	}

	private AColGroup hashToDummy(ColumnEncoderComposite c) {
		int colId = c._colID;
		Array<?> a = in.getColumn(colId - 1);
		ColumnEncoderFeatureHash CEHash = (ColumnEncoderFeatureHash) c.getEncoders().get(0);
		int domain = (int) CEHash.getK();
		boolean nulls = a.containsNull();
		IColIndex colIndexes = ColIndexFactory.create(0, domain);
		if(domain == 0 && nulls) {
			return new ColGroupEmpty(ColIndexFactory.create(1));
		}
		if(domain == 1 && !nulls) {
			nnz.addAndGet(in.getNumRows());
			return ColGroupConst.create(colIndexes, new double[] {1});
		}
		ADictionary d = new IdentityDictionary(colIndexes.size(), nulls);
		AMapToData m = createHashMappingAMapToData(a, domain, nulls);
		AColGroup ret = ColGroupDDC.create(colIndexes, d, m, null);
		nnz.addAndGet(ret.getNumberNonZeros(in.getNumRows()));
		return ret;
	}

	@SuppressWarnings("unchecked")
	private <T> void estimateRCDMapSize(ColumnEncoderComposite c) {
		if(c._estNumDistincts != 0)
			return;
		Array<T> col = (Array<T>) in.getColumn(c._colID - 1);
		if(col instanceof DDCArray) {
			DDCArray<T> ddcCol = (DDCArray<T>) col;
			c._estNumDistincts = ddcCol.getDict().size();
			return;
		}
		final int nRow = in.getNumRows();
		if(nRow <= 1024) {
			c._estNumDistincts = 10;
			return;
		}
		// 2% sample or max 3000
		int sampleSize = Math.max(Math.min(in.getNumRows() / 50, 4096 * 2), 1024);
		// Find the frequencies of distinct values in the sample
		Map<T, Integer> distinctFreq = new HashMap<>();
		for(int sind = 0; sind < sampleSize; sind++) {
			T key = col.getInternal(sind);
			if(distinctFreq.containsKey(key))
				distinctFreq.put(key, distinctFreq.get(key) + 1);
			else
				distinctFreq.put(key, 1);
		}

		// Estimate total #distincts using Hass and Stokes estimator
		int[] freq = distinctFreq.values().stream().mapToInt(v -> v).toArray();
		int estDistCount = SampleEstimatorFactory.distinctCount(freq, nRow, sampleSize,
			SampleEstimatorFactory.EstimationType.HassAndStokes);
		c._estNumDistincts = estDistCount;
	}

	private void combineUncompressed(CompressedMatrixBlock mb) {

		List<ColGroupUncompressedArray> ucg = new ArrayList<>();
		List<AColGroup> ret = new ArrayList<>();
		for(AColGroup g : mb.getColGroups()) {
			if(g instanceof ColGroupUncompressedArray)
				ucg.add((ColGroupUncompressedArray) g);
			else
				ret.add(g);
		}	
		LOG.error(ucg);
		LOG.error(ret);
		if(ucg.size() > 0){
			ret.add(combine(ucg));
			nnz.addAndGet(ret.get(ret.size()-1).getNumberNonZeros(in.getNumRows()));
		}
		mb.allocateColGroupList(ret);
	}

	private AColGroup combine(List<ColGroupUncompressedArray> ucg) {
		IColIndex combinedCols = ColIndexFactory.combine(ucg);

		ucg.sort((a,b) -> Integer.compare(a.id,b.id));
		MatrixBlock ret = new MatrixBlock(in.getNumRows(), combinedCols.size(), false);
		ret.allocateDenseBlock();
		DenseBlock db = ret.getDenseBlock();
		for(int i =0; i < in.getNumRows(); i++){
			double[] rval = db.values(i);
			int off = db.pos(i);
			for(int j = 0; j < combinedCols.size(); j++){
				rval[off + j] = ucg.get(j).array.getAsDouble(i);
			}
		}

		ret.recomputeNonZeros(k);

		return ColGroupUncompressed.create(ret, combinedCols);
	}

	private void logging(MatrixBlock mb) {
		if(LOG.isDebugEnabled()) {
			LOG.debug(String.format("Uncompressed transform encode Dense size:   %16d", mb.estimateSizeDenseInMemory()));
			LOG.debug(String.format("Uncompressed transform encode Sparse size:  %16d", mb.estimateSizeSparseInMemory()));
			LOG.debug(String.format("Compressed transform encode size:           %16d", mb.estimateSizeInMemory()));

			double ratio = Math.min(mb.estimateSizeDenseInMemory(), mb.estimateSizeSparseInMemory()) /
				mb.estimateSizeInMemory();
			double denseRatio = mb.estimateSizeDenseInMemory() / mb.estimateSizeInMemory();
			LOG.debug(String.format("Compression ratio: %10.3f", ratio));
			LOG.debug(String.format("Dense ratio:       %10.3f", denseRatio));
		}
	}
}
