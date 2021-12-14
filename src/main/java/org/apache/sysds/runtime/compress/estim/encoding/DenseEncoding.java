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
		if(map.getUnique() == 0)
			throw new DMLCompressionException("Invalid Dense Encoding");
	}

	/**
	 * Protected constructor that also counts the frequencies of the values.
	 * 
	 * @param map The Map.
	 */
	protected DenseEncoding(AMapToData map) {
		this.map = map;
		final int nUnique = map.getUnique();
		if(nUnique == 0)
			throw new DMLCompressionException("Invalid Dense Encoding");
		this.counts = new int[nUnique];
		for(int i = 0; i < map.size(); i++)
			counts[map.getIndex(i)]++;
	}

	@Override
	public DenseEncoding join(IEncode e) {
		if(e instanceof EmptyEncoding || e instanceof ConstEncoding)
			return this;
		else if(e instanceof SparseEncoding)
			return joinSparse((SparseEncoding) e);
		else
			return joinDense((DenseEncoding) e);
	}

	protected DenseEncoding joinSparse(SparseEncoding e) {

		final long maxUnique = (long) e.getUnique() * getUnique();

		if(maxUnique > (long) Integer.MAX_VALUE)
			throw new DMLCompressionException(
				"Joining impossible using linearized join, since each side has a large number of unique values");

		final int nRows = size();
		final int nVl = getUnique();
		final int defR = (e.getUnique() - 1) * nVl;
		final int[] m = new int[(int) maxUnique];
		final AMapToData d = MapToFactory.create(nRows, (int) maxUnique);

		final AIterator itr = e.off.getIterator();
		final int fr = e.off.getOffsetToLast();

		int newUID = 1;
		int r = 0;
		for(; r < fr; r++) {
			final int ir = itr.value();
			if(ir == r) {
				
				final int nv = map.getIndex(r) + e.map.getIndex(itr.getDataIndex()) * nVl;
				itr.next();
				newUID = addVal(nv, r, m, newUID, d);
			}
			else {
				final int nv = map.getIndex(r) + defR;
				newUID = addVal(nv, r, m, newUID, d);
			}
		}
		// add last offset
		newUID = addVal(map.getIndex(r) + e.map.getIndex(itr.getDataIndex()) * nVl, r++, m, newUID, d);

		// add remaining rows.
		for(; r < nRows; r++) {
			final int nv = map.getIndex(r) + defR;
			newUID = addVal(nv, r, m, newUID, d);
		}

		// set unique.
		d.setUnique(newUID - 1);
		return new DenseEncoding(d);
	}

	protected static int addVal(int nv, int r, int[] m, int newUID, AMapToData d) {
		final int mapV = m[nv];
		if(mapV == 0) {
			d.set(r, newUID - 1);
			m[nv] = newUID++;
		}
		else
			d.set(r, mapV - 1);
		return newUID;
	}

	protected DenseEncoding joinDense(DenseEncoding e) {
		if(map == e.map)
			return this; // unlikely to happen but cheap to compute
		return new DenseEncoding(MapToFactory.join(map, e.map));
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
	public int[] getCounts() {
		return counts;
	}

	@Override
	public EstimationFactors computeSizeEstimation(int[] cols, int nRows, double tupleSparsity, double matrixSparsity) {
		int largestOffs = 0;

		for(int i = 0; i < counts.length; i++)
			if(counts[i] > largestOffs)
				largestOffs = counts[i];

		return new EstimationFactors(cols.length, counts.length, nRows, largestOffs, counts, 0, 0, nRows, false, false,
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
