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
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.utils.HashMapLongInt;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

/**
 * A Encoding that contain a default value that is not encoded and every other value is encoded in the map. The logic is
 * similar to the SDC column group.
 */
public class SparseEncoding extends AEncode {

	/** A map to the distinct values contained */
	protected final AMapToData map;

	/** A Offset index structure to indicate space of zero values */
	protected final AOffset off;

	/** Total number of rows encoded */
	protected final int nRows;

	protected SparseEncoding(AMapToData map, AOffset off, int nRows) {
		this.map = map;
		this.off = off;
		this.nRows = nRows;

		if(CompressedMatrixBlock.debug) {
			int[] freq = map.getCounts();
			for(int i = 0; i < freq.length; i++) {
				if(freq[i] == 0)
					throw new DMLCompressionException("Invalid counts in fact contains 0");
			}
		}
	}

	@Override
	public IEncode combine(IEncode e) {
		if(e instanceof EmptyEncoding || e instanceof ConstEncoding)
			return this;
		else if(e instanceof SparseEncoding) {
			SparseEncoding es = (SparseEncoding) e;
			if(es.off == off && es.map == map)
				return this;
			return combineSparse(es);
		}
		else
			return e.combine(this);

	}

	@Override
	public Pair<IEncode, HashMapLongInt> combineWithMap(IEncode e) {
		if(e instanceof EmptyEncoding || e instanceof ConstEncoding)
			return new ImmutablePair<>(this, null);
		else if(e instanceof SparseEncoding) {
			SparseEncoding es = (SparseEncoding) e;
			if(es.off == off && es.map == map)
				return new ImmutablePair<>(this, null);
			return combineSparseNoResizeDense(es);
		}
		else
			throw new DMLCompressionException("Not allowed other to be dense. We should instead combine other way with sparse");
	}

	protected IEncode combineSparse(SparseEncoding e) {
		final int maxUnique = e.getUnique() * getUnique();
		final int[] d = new int[maxUnique - 1];

		final int fl = off.getOffsetToLast();
		final int fr = e.off.getOffsetToLast();
		final AIterator itl = off.getIterator();
		final AIterator itr = e.off.getIterator();

		final int nVl = getUnique();
		final int nVr = e.getUnique();

		final int sl = map.size();
		final int sr = e.map.size();

		if(sl + sr > nRows / 2)
			return combineSparseToDense(map, e.map, itl, itr, fl, fr, nVl, nVr, d, nRows, maxUnique);

		final IntArrayList retOff = new IntArrayList(Math.max(sr, sl));
		final IntArrayList tmpVals = new IntArrayList(Math.max(sr, sl));

		final int unique = combineSparse(map, e.map, itl, itr, retOff, tmpVals, fl, fr, nVl, nVr, d);

		if(retOff.size() < nRows / 4) {
			final AOffset o = OffsetFactory.createOffset(retOff);
			
			final AMapToData retMap = MapToFactory.create(tmpVals.size(), tmpVals.extractValues(), unique - 1);
			return new SparseEncoding(retMap, o, nRows);
		}
		else {
			// there will always be a zero therefore unique is not subtracted one.
			// if there is no zeros this will not be valid and crash.
			final AMapToData retMap = MapToFactory.create(nRows, unique);
			for(int i = 0; i < retOff.size(); i++)
				retMap.set(retOff.get(i), tmpVals.get(i) + 1);
			return new DenseEncoding(retMap);
		}
	}

	private Pair<IEncode, HashMapLongInt> combineSparseNoResizeDense(SparseEncoding e) {

		final int fl = off.getOffsetToLast();
		final int fr = e.off.getOffsetToLast();
		final AIterator itl = off.getIterator();
		final AIterator itr = e.off.getIterator();
		final int nVl = getUnique();
		final int nVr = e.getUnique();

		final AMapToData retMap = MapToFactory.create(nRows, (nVl + 1) * (nVr + 1));

		int il = itl.value();
		// parse through one side set all values into the dense.
		while(il < fl) {
			retMap.set(il, map.getIndex(itl.getDataIndex()) + 1);
			il = itl.next();
		}
		retMap.set(fl, map.getIndex(itl.getDataIndex()) + 1);

		int ir = itr.value();
		// parse through other side set all values with offset based on what already is there.
		while(ir < fr) {
			final int vl = retMap.getIndex(ir); // probably 0
			final int vr = e.map.getIndex(itr.getDataIndex()) + 1;
			retMap.set(ir, vl + vr * nVl);
			ir = itr.next();
		}
		retMap.set(fr, retMap.getIndex(fr) + (e.map.getIndex(itr.getDataIndex()) + 1) * nVl);

		// Full iteration to set unique elements.
		final HashMapLongInt m = new HashMapLongInt(100);
		for(int i = 0; i < retMap.size(); i++)
			addValHashMap(retMap.getIndex(i), i, m, retMap);

		return new ImmutablePair<>(new DenseEncoding(retMap.resize(m.size())), m);

	}


	protected static void addValHashMap(final int nv, final int r, final HashMapLongInt map,
		final AMapToData d) {
		final int v = map.size();
		final int mv = map.putIfAbsent(nv, v);
		if(mv == -1)
			d.set(r, v);
		else
			d.set(r, mv);
	}

	private static int combineSparse(AMapToData lMap, AMapToData rMap, AIterator itl, AIterator itr,
		final IntArrayList retOff, final IntArrayList tmpVals, final int fl, final int fr, final int nVl, final int nVr,
		final int[] d) {

		final int defR = (nVr - 1) * nVl;
		final int defL = nVl - 1;
		int newUID = 1;
		int il = itl.value();
		int ir = itr.value();

		while(il < fl && ir < fr) {
			if(il == ir) {// Both sides have a value same row.
				final int nv = lMap.getIndex(itl.getDataIndex()) + rMap.getIndex(itr.getDataIndex()) * nVl;
				newUID = addVal(nv, il, d, newUID, tmpVals, retOff);
				il = itl.next();
				ir = itr.next();
			}
			else if(il < ir) { // left side have a value before right
				final int nv = lMap.getIndex(itl.getDataIndex()) + defR;
				newUID = addVal(nv, il, d, newUID, tmpVals, retOff);
				il = itl.next();
			}
			else {// right side have a value before left
				final int nv = rMap.getIndex(itr.getDataIndex()) * nVl + defL;
				newUID = addVal(nv, ir, d, newUID, tmpVals, retOff);
				ir = itr.next();
			}
		}

		newUID = combineSparseTail(lMap, rMap, itl, itr, retOff, tmpVals, fl, fr, nVl, nVr, d, newUID);

		return newUID;
	}

	private static int combineSparseTail(AMapToData lMap, AMapToData rMap, AIterator itl, AIterator itr,
		final IntArrayList retOff, final IntArrayList tmpVals, final int fl, final int fr, final int nVl, final int nVr,
		final int[] d, int newUID) {
		final int defR = (nVr - 1) * nVl;
		final int defL = nVl - 1;
		int il = itl.value();
		int ir = itr.value();

		if(il == fl && ir == fr) {
			if(fl == fr) {
				final int nv = lMap.getIndex(itl.getDataIndex()) + rMap.getIndex(itr.getDataIndex()) * nVl;
				return addVal(nv, il, d, newUID, tmpVals, retOff);
			}
			else if(fl < fr) {// fl is first
				int nv = lMap.getIndex(itl.getDataIndex()) + defR;
				newUID = addVal(nv, il, d, newUID, tmpVals, retOff);
				nv = rMap.getIndex(itr.getDataIndex()) * nVl + defL;
				newUID = addVal(nv, fr, d, newUID, tmpVals, retOff);
			}
			else {// fl is last
				int nv = rMap.getIndex(itr.getDataIndex()) * nVl + defL;
				newUID = addVal(nv, fr, d, newUID, tmpVals, retOff);
				nv = lMap.getIndex(itl.getDataIndex()) + defR;
				newUID = addVal(nv, il, d, newUID, tmpVals, retOff);
			}
		}
		else if(il < fl) {
			if(fl < fr) {
				while(il < fl) {
					final int nv = lMap.getIndex(itl.getDataIndex()) + defR;
					newUID = addVal(nv, il, d, newUID, tmpVals, retOff);
					il = itl.next();
				}
				int nv = lMap.getIndex(itl.getDataIndex()) + defR;
				newUID = addVal(nv, il, d, newUID, tmpVals, retOff);
				nv = rMap.getIndex(itr.getDataIndex()) * nVl + defL;
				newUID = addVal(nv, fr, d, newUID, tmpVals, retOff);
				return newUID;
			}
			else {
				while(il < fr) {
					final int nv = lMap.getIndex(itl.getDataIndex()) + defR;
					newUID = addVal(nv, il, d, newUID, tmpVals, retOff);
					il = itl.next();
				}
				if(fl == fr) {
					final int nv = lMap.getIndex(itl.getDataIndex()) + rMap.getIndex(itr.getDataIndex()) * nVl;
					return addVal(nv, il, d, newUID, tmpVals, retOff);
				}
				else if(il == fr) {
					final int nv = lMap.getIndex(itl.getDataIndex()) + rMap.getIndex(itr.getDataIndex()) * nVl;
					newUID = addVal(nv, il, d, newUID, tmpVals, retOff);
					il = itl.next();
				}
				else {
					final int nv = rMap.getIndex(itr.getDataIndex()) * nVl + defL;
					newUID = addVal(nv, fr, d, newUID, tmpVals, retOff);
				}

				while(il < fl) {
					final int nv = lMap.getIndex(itl.getDataIndex()) + defR;
					newUID = addVal(nv, il, d, newUID, tmpVals, retOff);
					il = itl.next();
				}
				final int nv = lMap.getIndex(itl.getDataIndex()) + defR;
				newUID = addVal(nv, il, d, newUID, tmpVals, retOff);

			}
		}
		else { // if(ir < fr)
			if(fr < fl) {
				while(ir < fr) {
					final int nv = rMap.getIndex(itr.getDataIndex()) * nVl + defL;
					newUID = addVal(nv, ir, d, newUID, tmpVals, retOff);
					ir = itr.next();
				}
				int nv = rMap.getIndex(itr.getDataIndex()) * nVl + defL;
				newUID = addVal(nv, ir, d, newUID, tmpVals, retOff);
				nv = lMap.getIndex(itl.getDataIndex()) + defR;
				newUID = addVal(nv, fl, d, newUID, tmpVals, retOff);
				return newUID;
			}
			else {
				while(ir < fl) {
					final int nv = rMap.getIndex(itr.getDataIndex()) * nVl + defL;
					newUID = addVal(nv, ir, d, newUID, tmpVals, retOff);
					ir = itr.next();
				}

				if(fr == fl) {
					final int nv = lMap.getIndex(itl.getDataIndex()) + rMap.getIndex(itr.getDataIndex()) * nVl;
					return addVal(nv, ir, d, newUID, tmpVals, retOff);
				}
				else if(ir == fl) {
					final int nv = lMap.getIndex(itl.getDataIndex()) + rMap.getIndex(itr.getDataIndex()) * nVl;
					newUID = addVal(nv, ir, d, newUID, tmpVals, retOff);
					ir = itr.next();
				}
				else {
					final int nv = lMap.getIndex(itl.getDataIndex()) + defR;
					newUID = addVal(nv, fl, d, newUID, tmpVals, retOff);
				}

				while(ir < fr) {
					final int nv = rMap.getIndex(itr.getDataIndex()) * nVl + defL;
					newUID = addVal(nv, ir, d, newUID, tmpVals, retOff);
					ir = itr.next();
				}
				final int nv = rMap.getIndex(itr.getDataIndex()) * nVl + defL;
				newUID = addVal(nv, ir, d, newUID, tmpVals, retOff);

			}
		}

		return newUID;
	}

	private static int addVal(int nv, int offset, int[] d, int newUID, IntArrayList tmpVals, IntArrayList offsets) {
		int mapV = d[nv];
		if(mapV == 0)
			mapV = d[nv] = newUID++;
		tmpVals.appendValue(mapV - 1);
		offsets.appendValue(offset);
		return newUID;
	}

	private static DenseEncoding combineSparseToDense(AMapToData lMap, AMapToData rMap, AIterator itl, AIterator itr,
		int fl, int fr, int nVl, int nVr, int[] d, int nRows, int maxUnique) {

		final AMapToData retMap = MapToFactory.create(nRows, (nVl + 1) * (nVr + 1));
		int il = itl.value();
		// parse through one side set all values into the dense.
		while(il < fl) {
			retMap.set(il, lMap.getIndex(itl.getDataIndex()) + 1);
			il = itl.next();
		}
		retMap.set(fl, lMap.getIndex(itl.getDataIndex()) + 1);

		int ir = itr.value();
		// parse through other side set all values with offset based on what already is there.
		while(ir < fr) {
			final int vl = retMap.getIndex(ir); // probably 0
			final int vr = rMap.getIndex(itr.getDataIndex()) + 1;
			retMap.set(ir, vl + vr * nVl);
			ir = itr.next();
		}
		retMap.set(fr, retMap.getIndex(fr) + (rMap.getIndex(itr.getDataIndex()) + 1) * nVl);

		// parse through entire output reducing number of unique.
		final AMapToData tmpMap = MapToFactory.create(maxUnique, maxUnique + 1);
		int newUID = 1;
		for(int r = 0; r < retMap.size(); r++) {
			int nv = retMap.getIndex(r);
			int mv = tmpMap.getIndex(nv);
			if(mv == 0)
				mv = tmpMap.setAndGet(nv, newUID++);
			retMap.set(r, mv - 1);
		}
		// parse though other side and use all ret to set correctly.
		retMap.setUnique(newUID - 1);

		return new DenseEncoding(retMap);
	}

	@Override
	public int getUnique() {
		return map.getUnique() + 1;
	}

	@Override
	public EstimationFactors extractFacts(int nRows, double tupleSparsity, double matrixSparsity,
		CompressionSettings cs) {
		final int largestOffs = nRows - map.size(); // known largest off is zero tuples
		tupleSparsity = Math.min((double) map.size() / (double) nRows, tupleSparsity);
		final int[] counts = map.getCounts();

		if(cs.isRLEAllowed())
			return new EstimationFactors(map.getUnique(), map.size(), largestOffs, counts, 0, nRows, map.countRuns(off),
				false, true, matrixSparsity, tupleSparsity);
		else
			return new EstimationFactors(map.getUnique(), map.size(), largestOffs, counts, 0, nRows, false, true,
				matrixSparsity, tupleSparsity);

	}

	@Override
	public boolean isDense() {
		return false;
	}

	public AOffset getOffsets() {
		return off;
	}

	public AMapToData getMap() {
		return map;
	}

	public int getNumRows() {
		return nRows;
	}

	@Override
	public boolean equals(IEncode e) {
		return e instanceof SparseEncoding && //
			((SparseEncoding) e).off.equals(this.off) && //
			((SparseEncoding) e).map.equals(this.map);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("\n");
		sb.append("mapping: ");
		sb.append(map);
		sb.append("\n");
		sb.append("offsets: ");
		sb.append(off);
		return sb.toString();
	}
}
