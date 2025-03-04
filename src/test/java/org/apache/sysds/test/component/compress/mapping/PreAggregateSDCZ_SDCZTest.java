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

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class PreAggregateSDCZ_SDCZTest {

	protected static final Log LOG = LogFactory.getLog(PreAggregateSDCZ_SDCZTest.class.getName());

	private final AMapToData m;
	private final AMapToData tm;
	private final ADictionary td;
	private final AOffset of;
	private final AOffset tof;
	private final int nCol;
	private final double[] expected;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		final Random r = new Random(2321522);
		final int sm = Integer.MAX_VALUE;

		create(tests, 50, 2, 2, 1, 4, 4, r.nextInt(sm));
		create(tests, 50, 2, 2, 1, 2, 4, r.nextInt(sm));
		create(tests, 50, 2, 2, 1, 2, 10, r.nextInt(sm));

		create(tests, 50, 2, 2, 2, 2, 10, r.nextInt(sm));
		create(tests, 50, 2, 2, 4, 2, 10, r.nextInt(sm));

		create(tests, 10000, 2, 2, 4, 1000, 100, r.nextInt(sm));
		create(tests, 10000, 32, 200, 4, 1000, 100, r.nextInt(sm));
		create(tests, 10000, 150, 13, 4, 1000, 100, r.nextInt(sm));
		create(tests, 10000, 150, 130, 5, 1000, 100, r.nextInt(sm));

		create(tests, 10000, 32, 200, 1, 1000, 100, r.nextInt(sm));
		create(tests, 10000, 150, 13, 1, 1000, 100, r.nextInt(sm));
		create(tests, 10000, 150, 149, 1, 1000, 100, r.nextInt(sm));

		create(tests, 10000, 32, 200, 1, 100, 1000, r.nextInt(sm));
		create(tests, 10000, 150, 13, 1, 100, 1000, r.nextInt(sm));
		create(tests, 10000, 150, 149, 1, 100, 1000, r.nextInt(sm));

		return tests;
	}

	public PreAggregateSDCZ_SDCZTest(AMapToData m, AMapToData tm, ADictionary td, AOffset tof, AOffset of, int nCol,
		double[] expected) {
		CompressedMatrixBlock.debug = true;
		this.m = m;
		this.tm = tm;
		this.td = td;
		this.of = of;
		this.tof = tof;
		this.nCol = nCol;
		this.expected = expected;
	}

	@Test
	public void preAggregateSDCZ_DDC() {
		try {
			Dictionary ret = Dictionary.createNoCheck(new double[expected.length]);
			m.preAggregateSDCZ_SDCZ(tm, td, tof, of, ret, nCol);
			compare(ret.getValues(), expected, 0.000001);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(this.toString());
		}
	}

	private final void compare(double[] res, double[] exp, double eps) {
		assertTrue(res.length == exp.length);
		for(int i = 0; i < res.length; i++)
			if(Math.abs(res[i] - exp[i]) >= eps)
				fail("not equivalent preaggregate with\n" + m.getClass().getSimpleName() + " "
					+ tm.getClass().getSimpleName() + "\n" + m + "\n" + of + "\n" + tm + "\n" + tof + "\n" + td + "\n"
					+ " res: " + Arrays.toString(res) + "\n exp:" + Arrays.toString(exp) + "\n\n");
	}

	private static void create(ArrayList<Object[]> tests, int nRows, int nUnique1, int nUnique2, int nCol, int offR1,
		int offR2, int seed) {
		final Random r = new Random(seed);

		final AOffset of = MappingTestUtil.createRandomOffset(offR1, nRows, r);
		final AOffset tof = MappingTestUtil.createRandomOffset(offR2, nRows, r);
		final AMapToData m = MappingTestUtil.createRandomMap(of.getSize(), nUnique1, r);
		final AMapToData tm = MappingTestUtil.createRandomMap(tof.getSize(), nUnique2, r);

		double[] dv = new double[nUnique2 * nCol];
		ADictionary td = Dictionary.createNoCheck(dv);

		for(int i = 0; i < dv.length; i++)
			dv[i] = r.nextDouble();

		double[] exp = new double[nUnique1 * nCol];
		Dictionary expD = Dictionary.createNoCheck(exp);

		try {
			// use implementation to get baseline.
			m.preAggregateSDCZ_SDCZ(tm, td, tof, of, expD, nCol);
			createAllPermutations(tests, m, tm, tof, of, nUnique1, nUnique2, td, exp, nCol);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed construction\n" + tm + "\n" + td + "\n" + of + "\n" + m);
		}
	}

	private static void createAllPermutations(ArrayList<Object[]> tests, AMapToData m, AMapToData tm, AOffset tof,
		AOffset of, int nUnique1, int nUnique2, ADictionary td, double[] exp, int nCol) {
		AMapToData[] ml = MappingTestUtil.getAllHigherVersions(m);
		AMapToData[] tml = MappingTestUtil.getAllHigherVersions(tm);
		createFromList(tests, td, tof, of, nCol, exp, ml, tml);
	}

	private static void createFromList(ArrayList<Object[]> tests, ADictionary td, AOffset tof, AOffset of, int nCol,
		double[] exp, AMapToData[] ml, AMapToData[] tml) {
		for(AMapToData m : ml)
			for(AMapToData tm : tml)
				tests.add(new Object[] {m, tm, td, tof, of, nCol, exp});
	}
}
