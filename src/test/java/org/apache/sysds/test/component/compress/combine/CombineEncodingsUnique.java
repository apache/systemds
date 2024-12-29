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

package org.apache.sysds.test.component.compress.combine;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.estim.encoding.DenseEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.estim.encoding.SparseEncoding;
import org.apache.sysds.runtime.compress.utils.HashMapLongInt;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class CombineEncodingsUnique {
	public static final Log LOG = LogFactory.getLog(CombineEncodingsUnique.class.getName());
	private final IEncode ae;
	private final IEncode be;

	private enum MapVar {
		V1, V2;
	}

	@Parameters
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();
		try {
			int[] unique = new int[] {1, 3, 6};
			int[] seeds = new int[] {1, 3214, 2, 13};
			int[] sizes = new int[] {10, 12, 32, 56};

			for(int u : unique) {
				for(int s : sizes) {
					for(int se : seeds) {
						tests.add(new Object[] {genDense(u, s, se, MapVar.V1), genDense(u, s, se + 1, MapVar.V2)});
						final int maxRows = 5 * s;
						SparseEncoding sp = genSparse(u, s, 5, maxRows, se, MapVar.V2);
						SparseEncoding sp2 = genSparse(u, s, 5, maxRows, se + 1, MapVar.V2);
						DenseEncoding de = genDense(u, sp.getNumRows(), se + 2, MapVar.V2);

						tests.add(new Object[] {de, sp});
						tests.add(new Object[] {sp, sp2});
						tests.add(new Object[] {sp2, sp});

					}
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	public CombineEncodingsUnique(IEncode a, IEncode b) {
		this.ae = a;
		this.be = b;
	}

	@Test
	public void combineUnique() {
		try {

			Pair<IEncode, HashMapLongInt> cec = ae.combineWithMap(be);
			IEncode ce = cec.getLeft();
			HashMapLongInt cem = cec.getRight();
			assertEquals(cem.size(), ce.getUnique());
			// check all unique values are contained.
			checkContainsAllUnique(ce);
		}
		catch(Exception e) {
			e.printStackTrace();
			LOG.error("Failed to combine " + ae + " " + be);
			fail(e.getMessage());
		}
	}

	private void checkContainsAllUnique(IEncode ce) {
		if(ce instanceof DenseEncoding) {
			DenseEncoding ced = (DenseEncoding) ce;
			AMapToData m = ced.getMap();
			Set<Integer> s = new HashSet<>();
			for(int i = 0; i < m.size(); i++) {
				s.add(m.getIndex(i));
			}

			assertEquals(m.getUnique(), s.size());
		}
		else {
			throw new NotImplementedException();
		}
	}

	private static DenseEncoding genDense(int unique, int size, int seed, MapVar v) {
		return new DenseEncoding(genMap(unique, size, seed, v));
	}

	// private static SparseEncoding genSparse(int unique, int size, int delta, int seed, MapVar v) {
	// AOffset of = genOffset(size, delta, seed);
	// AMapToData map = genMap(unique, size, seed + 1, v);
	// return EncodingFactory.createSparse(map, of, of.getOffsetToLast() + 10);
	// }

	private static SparseEncoding genSparse(int unique, int size, int delta, int nRows, int seed, MapVar v) {
		AOffset of = genOffset(size, delta, nRows, seed);
		AMapToData map = genMap(unique, size, seed + 1, v);
		return EncodingFactory.createSparse(map, of, nRows);
	}

	private static AMapToData genMap(int unique, int size, int seed, MapVar v) {
		switch(v) {
			case V1:
				return genMapV1(unique, size, seed);
			case V2:
			default:
				return genMapV2(unique, size, seed);

		}
	}

	private static AMapToData genMapV1(int unique, int size, int seed) {
		AMapToData m = MapToFactory.create(size, unique);
		for(int i = 0; i < unique; i++) {
			m.set(i, i);
		}
		Random r = new Random(seed);
		for(int i = unique; i < size; i++) {
			m.set(i, r.nextInt(unique));
		}
		return m;
	}

	private static AMapToData genMapV2(int unique, int size, int seed) {
		AMapToData m = MapToFactory.create(size, unique);
		Random r = new Random(seed);
		for(int i = 0; i < size - unique; i++) {
			m.set(i, r.nextInt(unique));
		}
		for(int i = 0; i < unique; i++) {
			m.set(i + size - unique, i);
		}
		return m;
	}

	private static AOffset genOffset(int size, int delta, int max, int seed) {
		int[] offsets = new int[size];
		Random r = new Random(seed);
		int off = offsets[0] = r.nextInt(delta);
		for(int i = 1; i < size; i++) {
			off = offsets[i] = off + 1 + r.nextInt(delta);
		}
		return OffsetFactory.createOffset(offsets);
	}
}
