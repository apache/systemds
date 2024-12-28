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
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class PreAggregateDDC_DDCTest {

	protected static final Log LOG = LogFactory.getLog(PreAggregateDDC_DDCTest.class.getName());

	private final AMapToData m;
	private final AMapToData tm;
	private final ADictionary td;
	private final int nCol;
	private final double[] expected;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		final Random r = new Random(2321522);
		final int sm = Integer.MAX_VALUE;

		create(tests, 10, 1, 1, 1, r.nextInt(sm));
		create(tests, 10, 1, 10, 1, r.nextInt(sm));
		create(tests, 10, 10, 1, 1, r.nextInt(sm));
		create(tests, 10, 1, 1, 2, r.nextInt(sm));
		create(tests, 10, 1, 10, 2, r.nextInt(sm));

		create(tests, 10, 1, 1, 1, r.nextInt(sm));
		create(tests, 10, 1, 2, 1, r.nextInt(sm));
		create(tests, 10, 2, 1, 1, r.nextInt(sm));
		create(tests, 10, 1, 1, 2, r.nextInt(sm));
		create(tests, 10, 1, 2, 2, r.nextInt(sm));

		create(tests, 66, 1, 1, 1, r.nextInt(sm));
		create(tests, 66, 1, 2, 1, r.nextInt(sm));
		create(tests, 66, 2, 1, 1, r.nextInt(sm));
		create(tests, 66, 1, 1, 2, r.nextInt(sm));
		create(tests, 66, 1, 2, 2, r.nextInt(sm));
		
		
		create(tests, 10, 10, 1, 2, r.nextInt(sm));
		create(tests, 10, 10, 5, 1, r.nextInt(sm));
		create(tests, 10, 10, 5, 1, r.nextInt(sm));
		create(tests, 100, 10, 5, 1, r.nextInt(sm));
		create(tests, 1000, 10, 5, 1, r.nextInt(sm));
		create(tests, 1000, 10, 5, 2, r.nextInt(sm));
		create(tests, 1000, 2, 5, 2, r.nextInt(sm));
		create(tests, 1000, 120, 12, 1, r.nextInt(sm));
		create(tests, 1000, 120, 132, 2, r.nextInt(sm));
		create(tests, 1000, 13, 132, 1, r.nextInt(sm));
		create(tests, 1000, 150, 12, 1, r.nextInt(sm));
		create(tests, 1000, 160, 152, 2, r.nextInt(sm));
		create(tests, 1000, 13, 172, 1, r.nextInt(sm));
		create(tests, 1000, 321, 2, 2, r.nextInt(sm));
		create(tests, 1000, 321, 241, 2, r.nextInt(sm));
		create(tests, 1000, 321, 543, 2, r.nextInt(sm));
		create(tests, 1000, 321, 543, 10, r.nextInt(sm));
		create(tests, 10000, 32, 2, 1, r.nextInt(sm));
		create(tests, 10000, 2, 2, 1, r.nextInt(sm));
		create(tests, 10000, 2, 2, 10, r.nextInt(sm));
		create(tests, 10005, 2, 2, 1, r.nextInt(sm));
		create(tests, 10005, 2, 2, 10, r.nextInt(sm));

		createSkewed(tests, 10000, 2, 2, 10, r.nextInt(sm), 0.1);
		createSkewed(tests, 10000, 2, 2, 10, r.nextInt(sm), 0.01);
		createSkewed(tests, 10000, 2, 2, 10, r.nextInt(sm), 0.001);

		createSkewed(tests, 10000, 2, 2, 1, r.nextInt(sm), 0.1);
		createSkewed(tests, 10000, 2, 2, 1, r.nextInt(sm), 0.01);
		createSkewed(tests, 10000, 2, 2, 1, r.nextInt(sm), 0.001);

		return tests;
	}

	public PreAggregateDDC_DDCTest(AMapToData m, AMapToData tm, ADictionary td, int nCol, double[] expected) {
		CompressedMatrixBlock.debug = true;
		this.m = m;
		this.tm = tm;
		this.td = td;
		this.nCol = nCol;
		this.expected = expected;
	}

	@Test
	public void preAggregateDDC_DDC() {
		try {
			Dictionary ret = Dictionary.createNoCheck(new double[expected.length]);
			m.preAggregateDDC_DDC(tm, td, ret, nCol);
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
				fail("not equivalent preaggregate with " + m.getClass().getSimpleName() + " "
					+ tm.getClass().getSimpleName() + "\n" + m + "\n" + tm + "\n" + td + "\n\n" + " res: "
					+ Arrays.toString(res) + "\n exp:" + Arrays.toString(exp));
	}

	private static void create(ArrayList<Object[]> tests, int nRows, int nUnique1, int nUnique2, int nCol, int seed) {
		final Random r = new Random(seed);

		AMapToData m = MapToFactory.create(nRows, nUnique1);
		AMapToData tm = MapToFactory.create(nRows, nUnique2);

		for(int i = 0; i < nRows; i++) {
			m.set(i, r.nextInt(nUnique1));
			tm.set(i, r.nextInt(nUnique2));
		}

		double[] dv = new double[nUnique2 * nCol];
		ADictionary td = Dictionary.createNoCheck(dv);

		for(int i = 0; i < dv.length; i++)
			dv[i] = r.nextDouble();

		double[] exp = new double[nUnique1 * nCol];
		Dictionary expD = Dictionary.createNoCheck(exp);

		try {

			// use implementation to get baseline.
			m.preAggregateDDC_DDC(tm, td, expD, nCol);
			createAllPermutations(tests, m, tm, nUnique1, nUnique2, td, exp, nCol);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed construction");
		}
	}

	private static void createSkewed(ArrayList<Object[]> tests, int nRows, int nUnique1, int nUnique2, int nCol,
		int seed, double fractionZero) {
		final Random r = new Random(seed);

		final AMapToData m = MappingTestUtil.createRandomMap(nRows, nUnique1, r);
		final AMapToData tm = MappingTestUtil.createRandomMap(nRows, nUnique2, r);

		double[] dv = new double[nUnique2 * nCol];
		ADictionary td = Dictionary.createNoCheck(dv);

		for(int i = 0; i < dv.length; i++)
			dv[i] = r.nextDouble();

		double[] exp = new double[nUnique1 * nCol];
		Dictionary expD = Dictionary.createNoCheck(exp);

		try {

			// use implementation to get baseline.
			m.preAggregateDDC_DDC(tm, td, expD, nCol);
			createAllPermutations(tests, m, tm, nUnique1, nUnique2, td, exp, nCol);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed construction");
		}
	}

	private static void createAllPermutations(ArrayList<Object[]> tests, AMapToData m, AMapToData tm, int nUnique1,
		int nUnique2, ADictionary td, double[] exp, int nCol) {
		AMapToData[] ml = MappingTestUtil.getAllHigherVersions(m);
		AMapToData[] tml = MappingTestUtil.getAllHigherVersions(tm);
		createFromList(tests, td, nCol, exp, ml, tml);
	}

	private static void createFromList(ArrayList<Object[]> tests, ADictionary td, int nCol, double[] exp,
		AMapToData[] ml, AMapToData[] tml) {
		for(AMapToData m : ml)
			for(AMapToData tm : tml)
				tests.add(new Object[] {m, tm, td, nCol, exp});

	}
}
