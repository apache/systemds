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

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
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
public class CompressedEncode {
	protected static final Log LOG = LogFactory.getLog(CompressedEncode.class.getName());

	private final MultiColumnEncoder enc;
	private final FrameBlock in;

	private CompressedEncode(MultiColumnEncoder enc, FrameBlock in) {
		this.enc = enc;
		this.in = in;
	}

	public static MatrixBlock encode(MultiColumnEncoder enc, FrameBlock in) {
		return new CompressedEncode(enc, in).apply();
	}

	private MatrixBlock apply() {
		List<ColumnEncoderComposite> encoders = enc.getColumnEncoders();

		List<AColGroup> groups = new ArrayList<>(encoders.size());

		for(ColumnEncoderComposite c : encoders)
			groups.add(encode(c));

		int cols = shiftGroups(groups);

		MatrixBlock mb = new CompressedMatrixBlock(in.getNumRows(), cols, -1, false, groups);
		mb.recomputeNonZeros();
		logging(mb);
		return mb;
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
		else
			throw new NotImplementedException("Not supporting : " + c);
	}

	@SuppressWarnings("unchecked")
	private AColGroup recodeToDummy(ColumnEncoderComposite c) {
		int colId = c._colID;
		Array<?> a = in.getColumn(colId - 1);
		HashMap<?, Long> map = a.getRecodeMap();
		int domain = map.size();

		// int domain = c.getDomainSize();
		IColIndex colIndexes = ColIndexFactory.create(0, domain);

		ADictionary d = new IdentityDictionary(colIndexes.size());

		AMapToData m = createMappingAMapToData(a, map);

		List<ColumnEncoder> r = c.getEncoders();
		r.set(0, new ColumnEncoderRecode(colId, (HashMap<Object, Long>) map));

		return ColGroupDDC.create(colIndexes, d, m, null);

	}

	@SuppressWarnings("unchecked")
	private AColGroup recode(ColumnEncoderComposite c) {
		int colId = c._colID;
		Array<?> a = in.getColumn(colId - 1);
		HashMap<?, Long> map = a.getRecodeMap();
		int domain = map.size();

		// int domain = c.getDomainSize();
		IColIndex colIndexes = ColIndexFactory.create(1);
		MatrixBlock incrementing = new MatrixBlock(domain, 1, false);
		for(int i = 0; i < domain; i++)
			incrementing.quickSetValue(i, 0, i + 1);

		ADictionary d = MatrixBlockDictionary.create(incrementing);

		AMapToData m = createMappingAMapToData(a, map);

		List<ColumnEncoder> r = c.getEncoders();
		r.set(0, new ColumnEncoderRecode(colId, (HashMap<Object, Long>) map));

		return ColGroupDDC.create(colIndexes, d, m, null);

	}

	@SuppressWarnings("unchecked")
	private AColGroup passThrough(ColumnEncoderComposite c) {
		IColIndex colIndexes = ColIndexFactory.create(1);
		int colId = c._colID;
		Array<?> a = in.getColumn(colId - 1);
		HashMap<Object, Long> map = (HashMap<Object, Long>) a.getRecodeMap();
		final int blockSz = ConfigurationManager.getDMLConfig().getIntValue(DMLConfig.DEFAULT_BLOCK_SIZE);
		if(map.size()  >= blockSz){
			double[] vals = (double[]) a.changeType(ValueType.FP64).get();
			MatrixBlock col = new MatrixBlock(a.size(), 1, vals);
			col.recomputeNonZeros();
			// lets make it an uncompressed column group.
			return ColGroupUncompressed.create(colIndexes, col, false);
		}
		else{

			double[] vals = new double[map.size() + (a.containsNull() ? 1 : 0)];
			for(int i = 0; i < a.size(); i++) {
				Object v = a.get(i);
				if(map.containsKey(v)) {
					vals[map.get(v).intValue()] = a.getAsDouble(i);
				}
				else {
					map.put(null, (long) map.size());
					vals[map.get(v).intValue()] = a.getAsDouble(i);
				}
			}
	
			ADictionary d = Dictionary.create(vals);
			AMapToData m = createMappingAMapToData(a, map);
			return ColGroupDDC.create(colIndexes, d, m, null);
		}

	}

	private AMapToData createMappingAMapToData(Array<?> a, HashMap<?, Long> map) {
		AMapToData m = MapToFactory.create(in.getNumRows(), map.size());
		Array<?>.ArrayIterator it = a.getIterator();
		while(it.hasNext()) {
			Object v = it.next();
			if(v != null) {
				m.set(it.getIndex(), map.get(v).intValue());
			}
		}
		return m;
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
