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

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToChar;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToCharPByte;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.utils.HashMapLongInt;

/**
 * An Encoding that contains a value on each row of the input.
 */
public class DenseEncoding extends AEncode {

	private static boolean zeroWarn = false;

	private final AMapToData map;

	public DenseEncoding(AMapToData map) {
		this.map = map;

		if(CompressedMatrixBlock.debug) {
			// if(!zeroWarn) {
			int[] freq = map.getCounts();
			for(int i = 0; i < freq.length && !zeroWarn; i++) {
				if(freq[i] == 0) {
					LOG.warn("Dense encoding contains zero encoding, indicating not all dictionary entries are in use");
					zeroWarn = true;

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
	public Pair<IEncode, HashMapLongInt> combineWithMap(IEncode e) {
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

	private final Pair<IEncode, HashMapLongInt> combineSparseHashMap(final AMapToData ret) {
		final int size = ret.size();
		final HashMapLongInt m = new HashMapLongInt(100);
		for(int r = 0; r < size; r++) {
			final int prev = ret.getIndex(r);
			final int v = m.size();
			final int mv = m.putIfAbsent(prev, v);
			if(mv == -1)
				ret.set(r, v);
			else
				ret.set(r, mv);
		}
		return new ImmutablePair<>(new DenseEncoding(ret.resize(m.size())), m);
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
		return new DenseEncoding(ret.resize(newUID - 1));
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
		final DenseEncoding retE;
		final AMapToData ret = MapToFactory.create(size, maxUnique);
		if(maxUnique < Math.max(nVL, nVR)) {// overflow
			final HashMapLongInt m = new HashMapLongInt(Math.max(100, size / 100));
			retE = combineDenseWithHashMapLong(lm, rm, size, nVL, ret, m);
		}
		else if(maxUnique > size && maxUnique > 2048) {
			// aka there is more maxUnique than rows.
			final HashMapLongInt m = new HashMapLongInt(Math.max(100, maxUnique / 100));
			retE = combineDenseWithHashMap(lm, rm, size, nVL, ret, m);
		}
		else {
			final AMapToData m = MapToFactory.create(maxUnique, maxUnique + 1);
			retE = combineDenseWithMapToData(lm, rm, size, nVL, ret, maxUnique, m);
		}

		if(retE.getUnique() < 0) {
			String th = this.toString();
			String ot = other.toString();
			String cm = retE.toString();

			if(th.length() > 1000)
				th = th.substring(0, 1000);
			if(ot.length() > 1000)
				ot = ot.substring(0, 1000);
			if(cm.length() > 1000)
				cm = cm.substring(0, 1000);
			throw new DMLCompressionException(
				"Failed to combine dense encodings correctly: Number unique values is lower than max input: \n\n" + th
					+ "\n\n" + ot + "\n\n" + cm);
		}
		return retE;
	}

	private Pair<IEncode, HashMapLongInt> combineDenseNoResize(final DenseEncoding other) {
		if(map.equals(other.map)) {
			return combineSameMapping();
		}

		final AMapToData lm = map;
		final AMapToData rm = other.map;

		final int nVL = lm.getUnique();
		final int nVR = rm.getUnique();
		final int size = map.size();
		final int maxUnique = (int) Math.min((long) nVL * nVR, (long) size);

		final AMapToData ret = MapToFactory.create(size, maxUnique);

		final HashMapLongInt m = new HashMapLongInt(Math.max(100, maxUnique / 1000));
		return new ImmutablePair<>(combineDenseWithHashMap(lm, rm, size, nVL, ret, m), m);
	}

	private Pair<IEncode, HashMapLongInt> combineSameMapping() {
		LOG.warn("Constructing perfect mapping, this could be optimized to skip hashmap");
		final HashMapLongInt m = new HashMapLongInt(Math.max(100, map.size() / 100));
		for(int i = 0; i < map.getUnique(); i++)
			m.putIfAbsent(i * (map.getUnique() + 1), i);
		return new ImmutablePair<>(this, m); // same object
	}

	private Pair<IEncode, HashMapLongInt> combineSparseNoResize(final SparseEncoding other) {
		final AMapToData a = assignSparse(other);
		return combineSparseHashMap(a);
	}

	protected final DenseEncoding combineDenseWithHashMapLong(final AMapToData lm, final AMapToData rm, final int size,
		final long nVL, final AMapToData ret, HashMapLongInt m) {
		if(ret instanceof MapToChar)
			for(int r = 0; r < size; r++)
				addValHashMapChar((long) lm.getIndex(r) + rm.getIndex(r) * nVL, r, m, (MapToChar) ret);
		else
			for(int r = 0; r < size; r++)
				addValHashMap((long) lm.getIndex(r) + rm.getIndex(r) * nVL, r, m, ret);
		return new DenseEncoding(ret.resize(m.size()));
	}

	protected final DenseEncoding combineDenseWithHashMap(final AMapToData lm, final AMapToData rm, final int size,
		final int nVL, final AMapToData ret, HashMapLongInt m) {
		// JIT compile instance checks.
		if(ret instanceof MapToChar)
			combineDenseWIthHashMapCharOut(lm, rm, size, nVL, (MapToChar) ret, m);
		else if(ret instanceof MapToCharPByte)
			combineDenseWIthHashMapPByteOut(lm, rm, size, nVL, (MapToCharPByte) ret, m);
		else
			combineDenseWithHashMapGeneric(lm, rm, size, nVL, ret, m);
		ret.setUnique(m.size());
		return new DenseEncoding(ret);

	}

	private final void combineDenseWIthHashMapPByteOut(final AMapToData lm, final AMapToData rm, final int size,
		final int nVL, final MapToCharPByte ret, HashMapLongInt m) {
		for(int r = 0; r < size; r++)
			addValHashMapCharByte(calculateID(lm, rm, nVL, r), r, m, ret);
	}

	private final void combineDenseWIthHashMapCharOut(final AMapToData lm, final AMapToData rm, final int size,
		final int nVL, final MapToChar ret, HashMapLongInt m) {
		if(lm instanceof MapToChar && rm instanceof MapToChar)
			combineDenseWIthHashMapAllChar(lm, rm, size, nVL, ret, m);
		else// some other combination
			combineDenseWIthHashMapCharOutGeneric(lm, rm, size, nVL, ret, m);
	}

	private final void combineDenseWIthHashMapCharOutGeneric(final AMapToData lm, final AMapToData rm, final int size,
		final int nVL, final MapToChar ret, HashMapLongInt m) {
		for(int r = 0; r < size; r++)
			addValHashMapChar(calculateID(lm, rm, nVL, r), r, m, ret);
	}

	private final void combineDenseWIthHashMapAllChar(final AMapToData lm, final AMapToData rm, final int size,
		final int nVL, final MapToChar ret, HashMapLongInt m) {
		final MapToChar lmC = (MapToChar) lm;
		final MapToChar rmC = (MapToChar) rm;
		for(int r = 0; r < size; r++)
			addValHashMapChar(lmC.getIndex(r) + rmC.getIndex(r) * nVL, r, m, ret);

	}

	protected final void combineDenseWithHashMapGeneric(final AMapToData lm, final AMapToData rm, final int size,
		final int nVL, final AMapToData ret, HashMapLongInt m) {
		for(int r = 0; r < size; r++)
			addValHashMap(calculateID(lm, rm, nVL, r), r, m, ret);
	}

	protected final DenseEncoding combineDenseWithMapToData(final AMapToData lm, final AMapToData rm, final int size,
		final int nVL, final AMapToData ret, final int maxUnique, final AMapToData m) {
		int newUID = 1;
		newUID = addValRange(lm, rm, size, nVL, ret, m, newUID, 0, size);
		ret.setUnique(newUID - 1);
		return new DenseEncoding(ret);

	}

	private int addValRange(final AMapToData lm, final AMapToData rm, final int size, final int nVL,
		final AMapToData ret, final AMapToData m, int newUID, int start, int end) {
		for(int r = start; r < end; r++)
			newUID = addValMapToData(calculateID(lm, rm, nVL, r), r, m, newUID, ret);
		return newUID;
	}

	private int calculateID(final AMapToData lm, final AMapToData rm, final int nVL, int r) {
		return lm.getIndex(r) + rm.getIndex(r) * nVL;
	}

	protected static int addValMapToData(final int nv, final int r, final AMapToData map, int newId,
		final AMapToData d) {
		int mv = map.getIndex(nv);
		if(mv == 0)
			mv = map.setAndGet(nv, newId++);
		d.set(r, mv - 1);
		return newId;
	}

	protected static void addValHashMap(final int nv, final int r, final HashMapLongInt map, final AMapToData d) {
		final int v = map.size();
		final int mv = map.putIfAbsent(nv, v);
		if(mv == -1)
			d.set(r, v);
		else
			d.set(r, mv);
	}

	protected static void addValHashMapChar(final int nv, final int r, final HashMapLongInt map, final MapToChar d) {
		final int v = map.size();
		final int mv = map.putIfAbsent(nv, v);
		if(mv == -1)
			d.set(r, v);
		else
			d.set(r, mv);
	}

	protected static void addValHashMapCharByte(final int nv, final int r, final HashMapLongInt map,
		final MapToCharPByte d) {
		final int v = map.size();
		final int mv = map.putIfAbsent(nv, v);
		if(mv == -1)
			d.set(r, v);
		else
			d.set(r, mv);
	}

	protected static void addValHashMapChar(final long nv, final int r, final HashMapLongInt map, final MapToChar d) {
		final int v = map.size();
		final int mv = map.putIfAbsent(nv, v);
		if(mv == -1)
			d.set(r, v);
		else
			d.set(r, mv);
	}

	protected static void addValHashMap(final long nv, final int r, final HashMapLongInt map, final AMapToData d) {
		final int v = map.size();
		final int mv = map.putIfAbsent(nv, v);
		if(mv == -1)
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
			else if(counts[i] == 0) {
				if(!zeroWarn) {
					LOG.warn("Invalid count of 0 all values should have at least one instance index: " + i + " of "
						+ counts.length);
					zeroWarn = true;
				}
				counts[i] = 1;
			}

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
