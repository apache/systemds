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

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;

public class DenseEncoding implements IEncode {

	private final AMapToData map;
	private final int[] counts;

	protected DenseEncoding(AMapToData map, int[] counts) {
		this.map = map;
		this.counts = counts;
	}

	public DenseEncoding(AMapToData map) {
		this.map = map;
		this.counts = map.getCounts(new int[map.getUnique()]);
	}

	@Override
	public DenseEncoding combine(IEncode e) {
		if((long) (getUnique()) * e.getUnique() > Integer.MAX_VALUE)
			throw new DMLCompressionException("Invalid input to combine.");
		else if(e instanceof EmptyEncoding || e instanceof ConstEncoding)
			return this;
		else if(e instanceof SparseEncoding)
			return combineSparse((SparseEncoding) e);
		else
			return combineDense((DenseEncoding) e);
	}

	protected DenseEncoding combineSparse(SparseEncoding e) {
		final int maxUnique = e.getUnique() * getUnique();
		final int nRows = size();
		final int nVl = getUnique();
		final int defR = (e.getUnique() - 1) * nVl;
		final AMapToData m = MapToFactory.create(maxUnique, maxUnique + 1);
		final AMapToData d = MapToFactory.create(nRows, maxUnique);

		// iterate through indexes that are in the sparse encoding
		final AIterator itr = e.off.getIterator();
		final int fr = e.off.getOffsetToLast();

		int newUID = 1;
		int r = 0;
		for(; r <= fr; r++) {
			final int ir = itr.value();
			if(ir == r) {
				final int nv = map.getIndex(ir) + e.map.getIndex(itr.getDataIndex()) * nVl;
				newUID = addVal(nv, r, m, newUID, d);
				if(ir >= fr) {
					r++;
					break;
				}
				else
					itr.next();
			}
			else {
				final int nv = map.getIndex(r) + defR;
				newUID = addVal(nv, r, m, newUID, d);
			}
		}

		for(; r < nRows; r++) {
			final int nv = map.getIndex(r) + defR;
			newUID = addVal(nv, r, m, newUID, d);
		}

		// set unique.
		d.setUnique(newUID - 1);
		return new DenseEncoding(d);
	}

	private static int addVal(int nv, int r, AMapToData map, int newId, AMapToData d) {
		int mv = map.getIndex(nv);
		if(mv == 0)
			mv = map.setAndGet(nv, newId++);
		d.set(r, mv - 1);
		return newId;
	}

	protected DenseEncoding combineDense(DenseEncoding other) {
		if(map == other.map)
			return this; // unlikely to happen but cheap to compute
		final AMapToData d = combine(map, other.map);
		return new DenseEncoding(d);
	}

	public static AMapToData combine(AMapToData left, AMapToData right) {
		if(left == null)
			return right;
		else if(right == null)
			return left;

		final int nVL = left.getUnique();
		final int nVR = right.getUnique();
		final int size = left.size();
		final int maxUnique = nVL * nVR;

		final AMapToData ret = MapToFactory.create(size, maxUnique);
		final AMapToData map = MapToFactory.create(maxUnique, maxUnique + 1);

		int newUID = 1;
		for(int i = 0; i < size; i++) {
			final int nv = left.getIndex(i) + right.getIndex(i) * nVL;
			newUID = addVal(nv, i, map, newUID, ret);
		}

		ret.setUnique(newUID - 1);
		return ret;
	}

	@Override
	public int getUnique() {
		return counts.length;
	}

	@Override
	public int size() {
		return map.size();
	}

	@Override
	public EstimationFactors extractFacts(int[] cols, int nRows, double tupleSparsity, double matrixSparsity) {
		int largestOffs = 0;

		for(int i = 0; i < counts.length; i++)
			if(counts[i] > largestOffs)
				largestOffs = counts[i];

		return new EstimationFactors(cols.length, counts.length, nRows, largestOffs, counts, 0, nRows, false, false,
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
		sb.append("counts:  ");
		sb.append(Arrays.toString(counts));
		return sb.toString();
	}
}
