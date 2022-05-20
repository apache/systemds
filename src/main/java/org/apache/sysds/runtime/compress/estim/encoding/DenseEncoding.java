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

package org.apache.sysds.runtime.compress.estim.encoding;

import java.util.HashMap;
import java.util.Map;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;

public class DenseEncoding implements IEncode {

	private final AMapToData map;

	public DenseEncoding(AMapToData map) {
		this.map = map;
	}

	@Override
	public DenseEncoding combine(IEncode e) {
		if(e instanceof EmptyEncoding || e instanceof ConstEncoding)
			return this;
		else if(e instanceof SparseEncoding)
			return combineSparse((SparseEncoding) e);
		else
			return combineDense((DenseEncoding) e);
	}

	protected DenseEncoding combineSparse(SparseEncoding e) {
		final int maxUnique = e.getUnique() * getUnique();
		final int size = map.size();
		final int nVl = getUnique();

		// temp result
		final AMapToData ret = MapToFactory.create(size, maxUnique);

		// Iteration 1 copy dense data.
		ret.copy(map);

		// Iterate through indexes that are in the sparse encoding
		final AIterator itr = e.off.getIterator();
		final int fr = e.off.getOffsetToLast();

		int ir = itr.value();
		while(ir < fr) {
			ret.set(ir, ret.getIndex(ir) + ((e.map.getIndex(itr.getDataIndex()) + 1) * nVl));
			ir = itr.next();
		}
		ret.set(fr, ret.getIndex(fr) + ((e.map.getIndex(itr.getDataIndex()) + 1) * nVl));

		// Iteration 2 reassign indexes.
		if(maxUnique + nVl > size)
			return combineSparseHashMap(ret);
		else
			return combineSparseMapToData(ret, maxUnique, nVl);
	}

	private final DenseEncoding combineSparseHashMap(final AMapToData ret) {
		final int size = ret.size();
		final Map<Integer, Integer> m = new HashMap<>(size);
		for(int r = 0; r < size; r++) {
			final int prev = ret.getIndex(r);
			final int v = m.size();
			final Integer mv = m.putIfAbsent(prev, v);
			if(mv == null)
				ret.set(r, v);
			else
				ret.set(r, mv);
		}
		return new DenseEncoding(MapToFactory.resize(ret, m.size()));
	}

	private final DenseEncoding combineSparseMapToData(final AMapToData ret, final int maxUnique, final int nVl) {
		final int size = ret.size();
		final AMapToData m = MapToFactory.create(maxUnique, maxUnique + nVl);
		int newUID = 1;
		for(int r = 0; r < size; r++) {
			final int prev = ret.getIndex(r);
			int mv = m.getIndex(prev);
			if(mv == 0)
				mv = m.setAndGet(prev, newUID++);
			ret.set(r, mv - 1);
		}
		// Potential iteration 3 of resize
		return new DenseEncoding(MapToFactory.resize(ret, newUID - 1));
	}

	protected DenseEncoding combineDense(final DenseEncoding other) {
		try {

			if(map == other.map) // same object
				return this; // unlikely to happen but cheap to compute

			final AMapToData lm = map;
			final AMapToData rm = other.map;

			final int nVL = lm.getUnique();
			final int nVR = rm.getUnique();
			final int size = map.size();
			final int maxUnique = nVL * nVR;

			final AMapToData ret = MapToFactory.create(size, maxUnique);

			if(maxUnique > size)
				return combineDenseWithHashMap(lm, rm, size, nVL, ret);
			else
				return combineDenseWithMapToData(lm, rm, size, nVL, ret, maxUnique);
		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed to combine two dense\n" + this + "\n" + other, e);
		}
	}

	protected final DenseEncoding combineDenseWithHashMap(final AMapToData lm, final AMapToData rm, final int size,
		final int nVL, final AMapToData ret) {
		final Map<Integer, Integer> m = new HashMap<>(size);

		for(int r = 0; r < size; r++)
			addValHashMap(lm.getIndex(r) + rm.getIndex(r) * nVL, r, m, ret);
		return new DenseEncoding(MapToFactory.resize(ret, m.size()));

	}

	protected final DenseEncoding combineDenseWithMapToData(final AMapToData lm, final AMapToData rm, final int size,
		final int nVL, final AMapToData ret, final int maxUnique) {
		final AMapToData m = MapToFactory.create(maxUnique, maxUnique + 1);
		int newUID = 1;
		for(int r = 0; r < size; r++)
			newUID = addValMapToData(lm.getIndex(r) + rm.getIndex(r) * nVL, r, m, newUID, ret);
		return new DenseEncoding(MapToFactory.resize(ret, newUID - 1));
	}

	protected static int addValMapToData(final int nv, final int r, final AMapToData map, int newId,
		final AMapToData d) {
		int mv = map.getIndex(nv);
		if(mv == 0)
			mv = map.setAndGet(nv, newId++);
		d.set(r, mv - 1);
		return newId;
	}

	protected static void addValHashMap(final int nv, final int r, final Map<Integer, Integer> map, final AMapToData d) {
		final int v = map.size();
		final Integer mv = map.putIfAbsent(nv, v);
		if(mv == null)
			d.set(r, v);
		else
			d.set(r, mv);
	}

	@Override
	public int getUnique() {
		return map.getUnique();
	}

	@Override
	public EstimationFactors extractFacts(int nRows, double tupleSparsity, double matrixSparsity,
		CompressionSettings cs) {
		int largestOffs = 0;

		int[] counts = map.getCounts(new int[map.getUnique()]);
		for(int i = 0; i < counts.length; i++)
			if(counts[i] > largestOffs)
				largestOffs = counts[i];
		if(cs.isRLEAllowed())
			return new EstimationFactors(map.getUnique(), nRows, largestOffs, counts, 0, nRows, map.countRuns(), false,
				false, matrixSparsity, tupleSparsity);
		else
			return new EstimationFactors(map.getUnique(), nRows, largestOffs, counts, 0, nRows, false, false,
				matrixSparsity, tupleSparsity);

	}

	@Override
	public boolean isDense() {
		return true;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("\n");
		sb.append("mapping: ");
		sb.append(map);
		return sb.toString();
	}
}
