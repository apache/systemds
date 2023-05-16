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
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IdentityDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;

public class CompressedEncode {
	protected static final Log LOG = LogFactory.getLog(CompressedEncode.class.getName());

	/** The encoding scheme plan */
	private final MultiColumnEncoder enc;
	/** The Input FrameBlock */
	private final FrameBlock in;
	/** The thread count of the instruction */
	private final int k;

	private CompressedEncode(MultiColumnEncoder enc, FrameBlock in, int k) {
		this.enc = enc;
		this.in = in;
		this.k = k;
	}

	public static MatrixBlock encode(MultiColumnEncoder enc, FrameBlock in, int k) {
		return new CompressedEncode(enc, in, k).apply();
	}

	private MatrixBlock apply() {
		final List<ColumnEncoderComposite> encoders = enc.getColumnEncoders();
		final List<AColGroup> groups = isParallel() ? multiThread(encoders) : singleThread(encoders);
		final int cols = shiftGroups(groups);
		final MatrixBlock mb = new CompressedMatrixBlock(in.getNumRows(), cols, -1, false, groups);
		mb.recomputeNonZeros();
		logging(mb);
		return mb;
	}

	private boolean isParallel() {
		return k > 1 && enc.getEncoders().size() > 1;
	}

	private List<AColGroup> singleThread(List<ColumnEncoderComposite> encoders) {
		List<AColGroup> groups = new ArrayList<>(encoders.size());
		for(ColumnEncoderComposite c : encoders)
			groups.add(encode(c));
		return groups;
	}

	private List<AColGroup> multiThread(List<ColumnEncoderComposite> encoders) {

		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			List<EncodeTask> tasks = new ArrayList<>(encoders.size());

			for(ColumnEncoderComposite c : encoders)
				tasks.add(new EncodeTask(c));

			List<AColGroup> groups = new ArrayList<>(encoders.size());
			for(Future<AColGroup> t : pool.invokeAll(tasks))
				groups.add(t.get());

			pool.shutdown();
			return groups;
		}
		catch(InterruptedException | ExecutionException ex) {
			pool.shutdown();
			throw new DMLRuntimeException("Failed parallel compressed transform encode", ex);
		}
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

	private AColGroup encode(ColumnEncoderComposite c) {
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
	private AColGroup recodeToDummy(ColumnEncoderComposite c) {
		int colId = c._colID;
		Array<?> a = in.getColumn(colId - 1);
		boolean containsNull = a.containsNull();
		HashMap<?, Long> map = a.getRecodeMap();
		int domain = map.size();
		IColIndex colIndexes = ColIndexFactory.create(0, domain);
		if(domain == 1)
			return ColGroupConst.create(colIndexes, new double[] {1});
		ADictionary d = new IdentityDictionary(colIndexes.size(), containsNull);
		AMapToData m = createMappingAMapToData(a, map, containsNull);
		List<ColumnEncoder> r = c.getEncoders();
		r.set(0, new ColumnEncoderRecode(colId, (HashMap<Object, Long>) map));
		return ColGroupDDC.create(colIndexes, d, m, null);
	}

	private AColGroup bin(ColumnEncoderComposite c) {
		final int colId = c._colID;
		final Array<?> a = in.getColumn(colId - 1);
		final boolean containsNull = a.containsNull();
		final List<ColumnEncoder> r = c.getEncoders();
		final ColumnEncoderBin b = (ColumnEncoderBin) r.get(0);
		b.build(in);
		final IColIndex colIndexes = ColIndexFactory.create(1);

		ADictionary d = createIncrementingVector(b._numBin, containsNull);
		AMapToData m = binEncode(a, b, containsNull);

		AColGroup ret = ColGroupDDC.create(colIndexes, d, m, null);
		try {
			ret.getNumberNonZeros(a.size());
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Failed binning \n\n" + a + "\n" + b + "\n" + d + "\n" + m, e);
		}
		return ret;
	}

	private AMapToData binEncode(Array<?> a, ColumnEncoderBin b, boolean containsNull) {
		AMapToData m = MapToFactory.create(a.size(), b._numBin + (containsNull ? 1 : 0));
		if(containsNull) {
			for(int i = 0; i < a.size(); i++) {
				double v = a.getAsNaNDouble(i);
				if(Double.isNaN(v))
					m.set(i, b._numBin);
				else
					m.set(i, (int) b.getCodeIndex(v) - 1);
			}
		}
		else {
			for(int i = 0; i < a.size(); i++)
				m.set(i, (int) b.getCodeIndex(a.getAsDouble(i)) - 1);
		}
		return m;
	}

	private MatrixBlockDictionary createIncrementingVector(int nVals, boolean NaN) {

		MatrixBlock bins = new MatrixBlock(nVals + (NaN ? 1 : 0), 1, false);
		for(int i = 0; i < nVals; i++)
			bins.quickSetValue(i, 0, i + 1);
		if(NaN)
			bins.quickSetValue(nVals, 0, Double.NaN);

		return MatrixBlockDictionary.create(bins);

	}

	private AColGroup binToDummy(ColumnEncoderComposite c) {
		final int colId = c._colID;
		final Array<?> a = in.getColumn(colId - 1);
		final boolean containsNull = a.containsNull();
		final List<ColumnEncoder> r = c.getEncoders();
		final ColumnEncoderBin b = (ColumnEncoderBin) r.get(0);
		b.build(in);
		IColIndex colIndexes = ColIndexFactory.create(0, b._numBin);
		ADictionary d = new IdentityDictionary(colIndexes.size(), containsNull);
		AMapToData m = binEncode(a, b, containsNull);
		AColGroup ret = ColGroupDDC.create(colIndexes, d, m, null);
		ret.getNumberNonZeros(a.size());
		return ret;
	}

	@SuppressWarnings("unchecked")
	private AColGroup recode(ColumnEncoderComposite c) {
		int colId = c._colID;
		Array<?> a = in.getColumn(colId - 1);
		HashMap<?, Long> map = a.getRecodeMap();
		boolean containsNull = a.containsNull();
		int domain = map.size();

		// int domain = c.getDomainSize();
		IColIndex colIndexes = ColIndexFactory.create(1);
		if(domain == 1)
			return ColGroupConst.create(colIndexes, new double[] {1});
		MatrixBlock incrementing = new MatrixBlock(domain + (containsNull ? 1 : 0) , 1, false);
		for(int i = 0; i < domain; i++)
			incrementing.quickSetValue(i, 0, i + 1);
		if(containsNull)
			incrementing.quickSetValue(domain, 0 , Double.NaN);

		ADictionary d = MatrixBlockDictionary.create(incrementing);

		AMapToData m = createMappingAMapToData(a, map, containsNull);

		List<ColumnEncoder> r = c.getEncoders();
		r.set(0, new ColumnEncoderRecode(colId, (HashMap<Object, Long>) map));

		return ColGroupDDC.create(colIndexes, d, m, null);

	}

	@SuppressWarnings("unchecked")
	private AColGroup passThrough(ColumnEncoderComposite c) {
		IColIndex colIndexes = ColIndexFactory.create(1);
		int colId = c._colID;
		Array<?> a = in.getColumn(colId - 1);
		boolean containsNull = a.containsNull();
		HashMap<Object, Long> map = (HashMap<Object, Long>) a.getRecodeMap();
		final int blockSz = ConfigurationManager.getDMLConfig().getIntValue(DMLConfig.DEFAULT_BLOCK_SIZE);
		if(map.size() >= blockSz) {
			double[] vals = (double[]) a.changeType(ValueType.FP64).get();
			MatrixBlock col = new MatrixBlock(a.size(), 1, vals);
			col.recomputeNonZeros();
			// lets make it an uncompressed column group.
			return ColGroupUncompressed.create(colIndexes, col, false);
		}
		else {
			double[] vals = new double[map.size() + (containsNull ? 1 : 0)];
			if(containsNull)
				vals[map.size()] = Double.NaN;
			ValueType t = a.getValueType();
			map.forEach((k,v) -> vals[v.intValue()] = UtilFunctions.objectToDouble(t,k));
			ADictionary d = Dictionary.create(vals);
			AMapToData m = createMappingAMapToData(a, map, containsNull);
			return ColGroupDDC.create(colIndexes, d, m, null);
		}

	}

	private AMapToData createMappingAMapToData(Array<?> a, HashMap<?, Long> map, boolean containsNull) {
		final int si = map.size();
		AMapToData m = MapToFactory.create(in.getNumRows(), si + (containsNull ? 1 : 0));
		Array<?>.ArrayIterator it = a.getIterator();
		if(containsNull){

			while(it.hasNext()) {
				Object v = it.next();
				if(v != null)
					m.set(it.getIndex(), map.get(v).intValue());
				else
					m.set(it.getIndex(),si);
			}
		}
		else{
			while(it.hasNext()) {
				Object v = it.next();
				m.set(it.getIndex(), map.get(v).intValue());
			}
		}
		return m;
	}

	private AMapToData createHashMappingAMapToData(Array<?> a, int k, boolean nulls) {
		AMapToData m = MapToFactory.create(a.size(), k + (nulls ? 1 : 0));
		if(nulls) {
			for(int i = 0; i < a.size(); i++) {
				double h = a.hashDouble(i);
				if(Double.isNaN(h)) {
					m.set(i, k);
				}
				else {
					m.set(i, (int) h % k);
				}
			}
		}
		else {
			for(int i = 0; i < a.size(); i++) {
				double h = a.hashDouble(i);
				m.set(i, (int) h % k);
			}
		}
		return m;
	}

	private AColGroup hash(ColumnEncoderComposite c) {
		int colId = c._colID;
		Array<?> a = in.getColumn(colId - 1);
		ColumnEncoderFeatureHash CEHash = (ColumnEncoderFeatureHash) c.getEncoders().get(0);

		// HashMap<?, Long> map = a.getRecodeMap();
		int domain = (int) CEHash.getK();
		boolean nulls = a.containsNull();
		IColIndex colIndexes = ColIndexFactory.create(0, 1);
		if(domain == 1)
			return ColGroupConst.create(colIndexes, new double[] {1});

			MatrixBlock incrementing = new MatrixBlock(domain + (nulls ? 1 : 0), 1, false);
			for(int i = 0; i < domain; i++)
				incrementing.quickSetValue(i, 0, i + 1);
			if(nulls)
				incrementing.quickSetValue(domain, 0, Double.NaN);
	
			ADictionary d = MatrixBlockDictionary.create(incrementing);

		AMapToData m = createHashMappingAMapToData(a, domain , nulls);
		AColGroup ret =  ColGroupDDC.create(colIndexes, d, m, null);
		ret.getNumberNonZeros(a.size());
		return ret;
	}

	private AColGroup hashToDummy(ColumnEncoderComposite c) {
		int colId = c._colID;
		Array<?> a = in.getColumn(colId - 1);
		ColumnEncoderFeatureHash CEHash = (ColumnEncoderFeatureHash) c.getEncoders().get(0);
		int domain = (int) CEHash.getK();
		boolean nulls = a.containsNull();
		IColIndex colIndexes = ColIndexFactory.create(0, domain);
		if(domain == 1)
			return ColGroupConst.create(colIndexes, new double[] {1});
		ADictionary d = new IdentityDictionary(colIndexes.size(), nulls);
		AMapToData m = createHashMappingAMapToData(a, domain, nulls);
		return ColGroupDDC.create(colIndexes, d, m, null);
	}

	private class EncodeTask implements Callable<AColGroup> {

		ColumnEncoderComposite c;

		protected EncodeTask(ColumnEncoderComposite c) {
			this.c = c;
		}

		public AColGroup call() throws Exception {
			return encode(c);
		}
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
