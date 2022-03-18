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
		final int nRows = map.size();
		final int nVl = getUnique();

		// temp result
		final AMapToData d = MapToFactory.create(nRows, maxUnique);

		// Iteration 1 copy dense data.
		d.copy(map);

		// Iterate through indexes that are in the sparse encoding
		final AIterator itr = e.off.getIterator();
		final int fr = e.off.getOffsetToLast();

		int ir = itr.value();
		while(ir < fr) {
			d.set(ir, d.getIndex(ir) + ((e.map.getIndex(itr.getDataIndex()) + 1) * nVl));
			ir = itr.next();
		}
		d.set(fr, d.getIndex(fr) + ((e.map.getIndex(itr.getDataIndex()) + 1) * nVl));

		// Iteration 2 reassign indexes.
		final AMapToData m = MapToFactory.create(maxUnique, maxUnique + nVl);
		int newUID = 1;
		for(int r = 0; r < nRows; r++) {
			final int prev = d.getIndex(r);
			int mv = m.getIndex(prev);
			if(mv == 0)
				mv = m.setAndGet(prev, newUID++);
			d.set(r, mv - 1);
		}

		// Resize output (potentially a 3. iteration).
		return new DenseEncoding(MapToFactory.resize(d, newUID - 1));
	}

	protected static int addVal(int nv, int r, AMapToData map, int newId, AMapToData d) {
		int mv = map.getIndex(nv);
		if(mv == 0)
			mv = map.setAndGet(nv, newId++);
		d.set(r, mv - 1);
		return newId;
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
			final AMapToData map = MapToFactory.create(maxUnique, maxUnique + 1);

			int newUID = 1;
			// iterate once!
			for(int r = 0; r < size; r++)
				newUID = addVal(lm.getIndex(r) + rm.getIndex(r) * nVL, r, map, newUID, ret);

			// Resize output (potentially a 2. iteration).
			return new DenseEncoding(MapToFactory.resize(ret, newUID - 1));
		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed to combine two dense\n" + this + "\n" + other, e);
		}
	}

	@Override
	public int getUnique() {
		return map.getUnique();
	}

	@Override
	public EstimationFactors extractFacts(int[] cols, int nRows, double tupleSparsity, double matrixSparsity) {
		int largestOffs = 0;

		int[] counts = map.getCounts(new int[map.getUnique()]);
		for(int i = 0; i < counts.length; i++)
			if(counts[i] > largestOffs)
				largestOffs = counts[i];

		return new EstimationFactors(cols.length, map.getUnique(), nRows, largestOffs, counts, 0, nRows, false, false,
			matrixSparsity, tupleSparsity);
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
