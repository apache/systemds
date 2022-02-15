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

import java.util.Arrays;

import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

/** Most common is zero encoding */
public class SparseEncoding implements IEncode {

	/** A map to the distinct values contained */
	protected final AMapToData map;

	/** A Offset index structure to indicate space of zero values */
	protected final AOffset off;

	/** Total number of rows encoded */
	protected final int nRows;

	/** Count of Zero tuples in this encoding */
	protected final int zeroCount;

	/** Count of non zero tuples in this encoding */
	protected final int[] counts;

	protected SparseEncoding(AMapToData map, AOffset off, int zeroCount, int[] counts, int nRows) {
		this.map = map;
		this.off = off;
		this.zeroCount = zeroCount;
		this.counts = counts;
		this.nRows = nRows;
	}

	@Override
	public IEncode join(IEncode e) {
		if(e instanceof EmptyEncoding || e instanceof ConstEncoding)
			return this;
		else if(e instanceof SparseEncoding)
			return joinSparse((SparseEncoding) e);
		else
			return ((DenseEncoding) e).joinSparse(this);
	}

	protected IEncode joinSparse(SparseEncoding e) {
		if(e.map == map && e.off == off)
			return this; // unlikely to happen but cheap to compute therefore this skip is included.

		final int maxUnique = e.getUnique() * getUnique();
		final int[] d = new int[maxUnique - 1];

		// We need at least this size of offsets, but i don't know if i need more.
		final IntArrayList retOff = new IntArrayList(Math.max(e.size(), this.size()));
		final IntArrayList tmpVals = new IntArrayList(Math.max(e.size(), this.size()));

		final int fl = off.getOffsetToLast();
		final int fr = e.off.getOffsetToLast();
		final AIterator itl = off.getIterator();
		final AIterator itr = e.off.getIterator();

		final int nVl = getUnique();
		final int nVr = e.getUnique();

		final int unique = joinSparse(map, e.map, itl, itr, retOff, tmpVals, fl, fr, nVl, nVr, d);

		if(retOff.size() < nRows * 0.3) {
			final AOffset o = OffsetFactory.createOffset(retOff);
			final AMapToData retMap = MapToFactory.create(tmpVals.size(), tmpVals.extractValues(), unique);
			return new SparseEncoding(retMap, o, nRows - retOff.size(), retMap.getCounts(new int[unique - 1]), nRows);
		}
		else {
			final AMapToData retMap = MapToFactory.create(nRows, unique);
			retMap.fill(unique - 1);
			for(int i = 0; i < retOff.size(); i++)
				retMap.set(retOff.get(i), tmpVals.get(i));

			// add values.
			IEncode ret = DenseEncoding.joinDenseCount(retMap);
			return ret;
		}
	}

	private static int joinSparse(AMapToData lMap, AMapToData rMap, AIterator itl, AIterator itr, IntArrayList retOff,
		IntArrayList tmpVals, int fl, int fr, int nVl, int nVr, int[] d) {

		final int defR = (nVr - 1) * nVl;
		final int defL = nVl - 1;

		boolean doneL = false;
		boolean doneR = false;
		int newUID = 1;
		while(true) {
			final int il = itl.value();
			final int ir = itr.value();
			if(il == ir) {
				// Both sides have a value.
				final int nv = lMap.getIndex(itl.getDataIndex()) + rMap.getIndex(itr.getDataIndex()) * nVl;
				newUID = addVal(nv, il, d, newUID, tmpVals, retOff);
				if(il >= fl || ir >= fr) {
					if(il < fl)
						itl.next();
					else
						doneL = true;
					if(ir < fr)
						itr.next();
					else
						doneR = true;
					break;
				}
				// both sides.
				itl.next();
				itr.next();
			}
			else if(il < ir) {
				// left side have a value before right
				final int nv = lMap.getIndex(itl.getDataIndex()) + defR;
				newUID = addVal(nv, il, d, newUID, tmpVals, retOff);
				if(il >= fl) {
					doneL = true;
					break;
				}
				itl.next();
			}
			else {
				// right side have a value before left
				final int nv = rMap.getIndex(itr.getDataIndex()) * nVl + defL;
				newUID = addVal(nv, ir, d, newUID, tmpVals, retOff);
				if(ir >= fr) {
					doneR = true;
					break;
				}
				itr.next();
			}
		}

		// process stragglers
		if(!doneL) { // If there is stragglers in the left side
			while(true) {
				final int il = itl.value();
				final int ir = itr.value();
				int nv;
				if(ir == il)
					nv = lMap.getIndex(itl.getDataIndex()) + rMap.getIndex(itr.getDataIndex()) * nVl;
				else
					nv = lMap.getIndex(itl.getDataIndex()) + defR;
				newUID = addVal(nv, il, d, newUID, tmpVals, retOff);
				if(il >= fl)
					break;
				itl.next();
			}
		}
		else if(!doneR) {// If there is stragglers in the right side
			while(true) {
				final int il = itl.value();
				final int ir = itr.value();
				int nv;
				if(ir == il)
					nv = lMap.getIndex(itl.getDataIndex()) + rMap.getIndex(itr.getDataIndex()) * nVl;
				else
					nv = rMap.getIndex(itr.getDataIndex()) * nVl + defL;

				newUID = addVal(nv, ir, d, newUID, tmpVals, retOff);
				if(ir >= fr)
					break;
				itr.next();
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

	@Override
	public int getUnique() {
		return counts.length + 1;
	}

	@Override
	public int size() {
		return map.size();
	}

	@Override
	public EstimationFactors extractFacts(int[] cols, int nRows, double tupleSparsity, double matrixSparsity) {
		final int largestOffs = nRows - map.size(); // known largest off is zero tuples
		tupleSparsity = Math.min((double) map.size() / (double) nRows, tupleSparsity);
		return new EstimationFactors(cols.length, counts.length, map.size(), largestOffs, counts, 0, nRows, false, true,
			matrixSparsity, tupleSparsity);
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
		sb.append("\n");
		sb.append("counts:  ");
		sb.append(Arrays.toString(counts));
		return sb.toString();
	}
}
