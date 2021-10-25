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

package org.apache.sysds.test.component.compress.mapping;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToByte;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class MappingPreAggregateTests {

	protected static final Log LOG = LogFactory.getLog(MappingTests.class.getName());

	public final int seed;
	public final MAP_TYPE type;
	public final int size;
	private AMapToData m;
	private MapToByte ref;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		for(MAP_TYPE t : MAP_TYPE.values()) {
			tests.add(new Object[] {1, t, 13});
			tests.add(new Object[] {3, t, 13});
			tests.add(new Object[] {3, t, 63});
			tests.add(new Object[] {3, t, 64});
			tests.add(new Object[] {3, t, 65});
			tests.add(new Object[] {5, t, 1234});
			tests.add(new Object[] {5, t, 13});
		}
		return tests;
	}

	public MappingPreAggregateTests(int seed, MAP_TYPE type, int size) {
		this.seed = seed;
		this.type = type;
		this.size = size;
		genBitMap(seed);
	}

	protected AMapToData genBitMap(int seed) {
		final Random r = new Random(seed);
		m = MapToFactory.create(size, 2);
		ref = (MapToByte) MapToFactory.create(size, 255);

		for(int i = 0; i < size; i++) {
			int v = r.nextInt(2);
			m.set(i, v);
			ref.set(i, v);
		}
		m = MapToFactory.resizeForce(m, type);
		return m;
	}

	@Test
	public void testPreAggregateDense() {
		int nUnique = m.getUnique();
		int size = m.size();

		MatrixBlock mb = TestUtils.generateTestMatrixBlock(1, size, 0, 100, 1.0, seed);
		MatrixBlock pre = new MatrixBlock(1, nUnique, false);
		pre.allocateDenseBlock();

		m.preAggregateDense(mb, pre, 0, 1, 0, 100);

		MatrixBlock preRef = new MatrixBlock(1, nUnique, false);
		preRef.allocateDenseBlock();
		
		ref.preAggregateDense(mb, preRef, 0, 1,0,100);

		TestUtils.compareMatrices(preRef, pre, 0, "preaggregate not same with different maps");
	}
}
