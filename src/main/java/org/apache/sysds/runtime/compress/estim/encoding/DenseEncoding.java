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

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;

/**
 * An Encoding that contains a value on each row of the input.
 */
public class DenseEncoding extends AEncode {

	private static boolean zeroWarn = false;

	private final AMapToData map;

	public DenseEncoding(AMapToData map) {
		this.map = map;

		if(CompressedMatrixBlock.debug) {
			if(!zeroWarn) {
				int[] freq = map.getCounts();
				for(int i = 0; i < freq.length; i++) {
					if(freq[i] == 0) {
						LOG.warn("Dense encoding contains zero encoding, indicating not all dictionary entries are in use");
						zeroWarn = true;
						break;
					}
					// throw new DMLCompressionException("Invalid counts in fact contains 0");
				}
			}
		}
	}

	@Override
	public IEncode combine(IEncode e) {
		if(e instanceof EmptyEncoding || e instanceof ConstEncoding)
			return this;
		else if(e instanceof SparseEncoding)
			return combineSparse((SparseEncoding) e);
		else
			return combineDense((DenseEncoding) e);
	}

	@Override
	public Pair<IEncode, Map<Integer, Integer>> combineWithMap(IEncode e) {
		if(e instanceof EmptyEncoding || e instanceof ConstEncoding)
			return new ImmutablePair<>(this, null);
		else if(e instanceof SparseEncoding)
			return combineSparseNoResize((SparseEncoding) e);
		else
			return combineDenseNoResize((DenseEncoding) e);
	}

	protected IEncode combineSparse(SparseEncoding e) {
		final int maxUnique = e.getUnique() * getUnique();
		final int size = map.size();
		final int nVl = getUnique();

		// temp result
		final AMapToData ret = assignSparse(e);
		// Iteration 2 reassign indexes.
		if(maxUnique + nVl > size)
			return combineSparseHashMap(ret).getLeft();
		else
			return combineSparseMapToData(ret, maxUnique, nVl);
	}

	private AMapToData assignSparse(SparseEncoding e) {
		final int maxUnique = e.getUnique() * getUnique();
		final int size = map.size();
		final int nVl = getUnique();
		// temp result
		final AMapToData ret = MapToFactory.create(size, maxUnique);

		// Iteration 1 copy dense data.
		ret.copy(map);
		final AIterator itr = e.off.getIterator();
		final int fr = e.off.getOffsetToLast();

		int ir = itr.value();
		while(ir < fr) {
			ret.set(ir, ret.getIndex(ir) + ((e.map.getIndex(itr.getDataIndex()) + 1) * nVl));
			ir = itr.next();
		}
		ret.set(fr, ret.getIndex(fr) + ((e.map.getIndex(itr.getDataIndex()) + 1) * nVl));
		return ret;
	}

	private final Pair<IEncode, Map<Integer, Integer>> combineSparseHashMap(final AMapToData ret) {
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
		return new ImmutablePair<>(new DenseEncoding(MapToFactory.resize(ret, m.size())), m);
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
		if(map == other.map) // same object
			return this; // unlikely to happen but cheap to compute

		final AMapToData lm = map;
		final AMapToData rm = other.map;

		final int nVL = lm.getUnique();
		final int nVR = rm.getUnique();
		final int size = map.size();
		int maxUnique = nVL * nVR;
		DenseEncoding retE = null;
		if(maxUnique < Math.max(nVL, nVR)) {// overflow
			maxUnique = size;
			final AMapToData ret = MapToFactory.create(size, maxUnique);
			final Map<Long, Integer> m = new HashMap<>(size);
			retE = combineDenseWithHashMapLong(lm, rm, size, nVL, ret, m);
		}
		else if(maxUnique > size && maxUnique > 2048) {
			final AMapToData ret = MapToFactory.create(size, maxUnique);
			// aka there is more maxUnique than rows.
			final Map<Integer, Integer> m = new HashMap<>(size);
			retE = combineDenseWithHashMap(lm, rm, size, nVL, ret, m);
		}
		else {
			final AMapToData ret = MapToFactory.create(size, maxUnique);
			final AMapToData m = MapToFactory.create(maxUnique, maxUnique + 1);
			retE = combineDenseWithMapToData(lm, rm, size, nVL, ret, maxUnique, m);
		}

		if(retE.getUnique() < 0) {
			throw new DMLCompressionException(
				"Failed to combine dense encodings correctly: Number unique values is lower than max input: \n\n" + this
					+ "\n\n" + other + "\n\n" + retE);
		}
		return retE;
	}

	private Pair<IEncode, Map<Integer, Integer>> combineDenseNoResize(final DenseEncoding other) {
		if(map.equals(other.map)) {
			LOG.warn("Constructing perfect mapping, this could be optimized to skip hashmap");
			final Map<Integer, Integer> m = new HashMap<>(map.size());
			for(int i = 0; i < map.getUnique(); i++)
				m.put(i * (map.getUnique() + 1) , i);
			return new ImmutablePair<>(this, m); // same object
		}

		final AMapToData lm = map;
		final AMapToData rm = other.map;

		final int nVL = lm.getUnique();
		final int nVR = rm.getUnique();
		final int size = map.size();
		final int maxUnique = (int) Math.min((long) nVL * nVR, (long) size);

		final AMapToData ret = MapToFactory.create(size, maxUnique);

		final Map<Integer, Integer> m = new HashMap<>(maxUnique);
		return new ImmutablePair<>(combineDenseWithHashMap(lm, rm, size, nVL, ret, m), m);
	}

	private Pair<IEncode, Map<Integer, Integer>> combineSparseNoResize(final SparseEncoding other) {
		final AMapToData a = assignSparse(other);
		return combineSparseHashMap(a);
	}

	protected final DenseEncoding combineDenseWithHashMapLong(final AMapToData lm, final AMapToData rm, final int size,
		final int nVL, final AMapToData ret, Map<Long, Integer> m) {

		for(int r = 0; r < size; r++)
			addValHashMap((long) lm.getIndex(r) + (long) rm.getIndex(r) * (long) nVL, r, m, ret);
		return new DenseEncoding(MapToFactory.resize(ret, m.size()));
	}

	protected final DenseEncoding combineDenseWithHashMap(final AMapToData lm, final AMapToData rm, final int size,
		final int nVL, final AMapToData ret, Map<Integer, Integer> m) {

		for(int r = 0; r < size; r++)
			addValHashMap(lm.getIndex(r) + rm.getIndex(r) * nVL, r, m, ret);
		return new DenseEncoding(MapToFactory.resize(ret, m.size()));
	}

	protected final DenseEncoding combineDenseWithMapToData(final AMapToData lm, final AMapToData rm, final int size,
		final int nVL, final AMapToData ret, final int maxUnique, final AMapToData m) {
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

	protected static void addValHashMap(final long nv, final int r, final Map<Long, Integer> map, final AMapToData d) {
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

		int[] counts = map.getCounts();
		for(int i = 0; i < counts.length; i++)
			if(counts[i] > largestOffs)
				largestOffs = counts[i];
			else if(counts[i] == 0)
				throw new DMLCompressionException("Invalid count of 0 all values should have at least one instance");

		if(cs.isRLEAllowed())
			return new EstimationFactors(map.getUnique(), nRows, largestOffs, counts, 0, nRows, map.countRuns(), false,
				false, matrixSparsity, tupleSparsity);
		else
			return new EstimationFactors(map.getUnique(), nRows, largestOffs, counts, 0, nRows, false, false,
				matrixSparsity, tupleSparsity);

	}

	public AMapToData getMap() {
		return map;
	}

	@Override
	public boolean isDense() {
		return true;
	}

	@Override
	public boolean equals(IEncode e) {
		return e instanceof DenseEncoding && ((DenseEncoding) e).map.equals(this.map);
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
