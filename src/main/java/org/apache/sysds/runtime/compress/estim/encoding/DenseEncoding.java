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
		// for debugging correctness and efficiency but should be guaranteed by implementations creating the Dense encoding:
		// if(map.getUnique() == 0)
		// 	throw new DMLCompressionException("Invalid Dense Encoding");
		// if(map.getUnique() != counts.length)
		// 	throw new DMLCompressionException(
		// 		"Invalid number of counts not matching map:" + map.getUnique() + "  " + counts.length);
		// int u = map.getUnique();
		// for(int i = 0; i < map.size(); i++)
		// 	if(map.getIndex(i) >= u)
		// 		throw new DMLCompressionException("Invalid values contained in map:" + map.getUnique() + "  " + map);
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
				else {
					itr.next();
				}
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
		d.setUnique(newUID-1);
		return joinDenseCount(d);
	}

	private static int addVal(int nv, int r, int[] m, int newId, AMapToData d) {
		if(m[nv] == 0)
			d.set(r, (m[nv] = newId++) - 1);
		else
			d.set(r, m[nv] - 1);
		return newId;
	}

	protected DenseEncoding joinDense(DenseEncoding e) {
		if(map == e.map)
			return this; // unlikely to happen but cheap to compute
		final AMapToData d = MapToFactory.join(map, e.map);
		return joinDenseCount(d);
	}

	protected static DenseEncoding joinDenseCount(AMapToData d) {
		int[] counts = new int[d.getUnique()];
		d.getCounts(counts);
		return new DenseEncoding(d, counts);
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
