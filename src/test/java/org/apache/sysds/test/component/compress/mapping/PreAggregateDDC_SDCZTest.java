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
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class PreAggregateDDC_SDCZTest {

	protected static final Log LOG = LogFactory.getLog(PreAggregateDDC_DDCTest.class.getName());

	private final AMapToData m;
	private final AMapToData tm;
	private final ADictionary td;
	private final AOffset tof;
	private final int nCol;
	private final double[] expected;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		final Random r = new Random(2321522);
		final int sm = Integer.MAX_VALUE;

		create(tests, 10, 10, 5, 1, 4, r.nextInt(sm));
		create(tests, 10, 10, 5, 1, 4, r.nextInt(sm));

		create(tests, 10, 10, 5, 3, 4, r.nextInt(sm));
		create(tests, 10, 10, 5, 4, 4, r.nextInt(sm));

		create(tests, 300, 1425, 2, 2, 4, r.nextInt(sm));
		create(tests, 42, 232, 2, 2, 4, r.nextInt(sm));

		create(tests, 10000, 342, 23, 1, 50, r.nextInt(sm));
		create(tests, 10000, 342, 23, 3, 100, r.nextInt(sm));
		create(tests, 10000, 342, 23, 30, 1000, r.nextInt(sm));

		create(tests, 1000000, 1000, 1000, 30, 1000, r.nextInt(sm));
		create(tests, 1000000, 1000, 1000, 30, 3, r.nextInt(sm));
		create(tests, 1000000, 2, 2, 30, 3, r.nextInt(sm));

		create(tests, 10000, 2, 2, 30, 126, r.nextInt(sm));
		create(tests, 10000, 2, 2, 30, 230, r.nextInt(sm));
		create(tests, 10000, 2, 2, 30, 500, r.nextInt(sm));

		return tests;
	}

	public PreAggregateDDC_SDCZTest(AMapToData m, AMapToData tm, ADictionary td, AOffset tof, int nCol,
		double[] expected) {
		this.m = m;
		this.tm = tm;
		this.td = td;
		this.tof = tof;
		this.nCol = nCol;
		this.expected = expected;
	}

	@Test
	public void preAggregateDDC_SDCZ() {
		Dictionary ret = new Dictionary(new double[expected.length]);
		m.preAggregateDDC_SDCZ(tm, td, tof, ret, nCol);
		compare(ret.getValues(), expected, 0.000001);
	}

	private final void compare(double[] res, double[] exp, double eps) {
		assertTrue(res.length == exp.length);
		for(int i = 0; i < res.length; i++)
			if(Math.abs(res[i] - exp[i]) >= eps)
				fail("not equivalent preaggregate with " + m.getClass().getSimpleName() + " "
					+ tm.getClass().getSimpleName() + "\n" + m + "\n" + tm + "\n" + td + "\n\n" + " res: "
					+ Arrays.toString(res) + "\n exp:" + Arrays.toString(exp));
	}

	private static void create(ArrayList<Object[]> tests, int nRows, int nUnique1, int nUnique2, int nCol, int offRange,
		int seed) {
		final Random r = new Random(seed);

		IntArrayList offs = new IntArrayList();
		int off = r.nextInt(offRange);
		offs.appendValue(off);
		while(off < nRows) {
			off += r.nextInt(offRange) + 1;
			if(off < nRows)
				offs.appendValue(off);
		}
		AOffset tof = OffsetFactory.createOffset(offs);

		AMapToData tm = MapToFactory.create(offs.size(), nUnique2);
		for(int i = 0; i < offs.size(); i++)
			tm.set(i, r.nextInt(nUnique2));

		AMapToData m = MapToFactory.create(nRows, nUnique1);
		for(int i = 0; i < nRows; i++)
			m.set(i, r.nextInt(nUnique1));

		double[] dv = new double[nUnique2 * nCol];
		ADictionary td = new Dictionary(dv);

		for(int i = 0; i < dv.length; i++)
			dv[i] = r.nextDouble();

		double[] exp = new double[nUnique1 * nCol];
		Dictionary expD = new Dictionary(exp);

		try {

			// use implementation to get baseline.
			m.preAggregateDDC_SDCZ(tm, td, tof, expD, nCol);
			createAllPermutations(tests, m, tm, tof, nUnique1, nUnique2, td, exp, nCol);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed construction");
		}
	}

	private static void createAllPermutations(ArrayList<Object[]> tests, AMapToData m, AMapToData tm, AOffset tof,
		int nUnique1, int nUnique2, ADictionary td, double[] exp, int nCol) {
		// assert the number of mapTypes, to ensure that if someone adds new ones we add them here.
		assertTrue(MAP_TYPE.values().length == 4);

		// a little nasty code but it works with testing all combinations possible to construct from the input type.
		if(nUnique1 <= 2) {// bit org
			AMapToData m_byte = MapToFactory.resizeForce(m, MAP_TYPE.BYTE);
			AMapToData m_char = MapToFactory.resizeForce(m, MAP_TYPE.CHAR);
			AMapToData m_int = MapToFactory.resizeForce(m, MAP_TYPE.INT);
			if(nUnique2 <= 2) { // bit org
				AMapToData tm_byte = MapToFactory.resizeForce(tm, MAP_TYPE.BYTE);
				AMapToData tm_char = MapToFactory.resizeForce(tm, MAP_TYPE.CHAR);
				AMapToData tm_int = MapToFactory.resizeForce(tm, MAP_TYPE.INT);
				createFromList(tests, m, td, tof, nCol, exp, tm_byte, tm_char, tm_int);
				createFromList(tests, m_byte, td, tof, nCol, exp, tm, tm_byte, tm_char, tm_int);
				createFromList(tests, m_char, td, tof, nCol, exp, tm, tm_byte, tm_char, tm_int);
				createFromList(tests, m_int, td, tof, nCol, exp, tm, tm_byte, tm_char, tm_int);
			}
			else if(nUnique2 < 256) { // byte org
				AMapToData tm_char = MapToFactory.resizeForce(tm, MAP_TYPE.CHAR);
				AMapToData tm_int = MapToFactory.resizeForce(tm, MAP_TYPE.INT);
				createFromList(tests, m, td, tof, nCol, exp, tm_char, tm_int);
				createFromList(tests, m_byte, td, tof, nCol, exp, tm, tm_char, tm_int);
				createFromList(tests, m_char, td, tof, nCol, exp, tm, tm_char, tm_int);
				createFromList(tests, m_int, td, tof, nCol, exp, tm, tm_char, tm_int);
			}
			else { // char org
				AMapToData tm_int = MapToFactory.resizeForce(tm, MAP_TYPE.INT);
				createFromList(tests, m, td, tof, nCol, exp, tm_int);
				createFromList(tests, m_byte, td, tof, nCol, exp, tm, tm_int);
				createFromList(tests, m_char, td, tof, nCol, exp, tm, tm_int);
				createFromList(tests, m_int, td, tof, nCol, exp, tm, tm_int);
			}
		}
		else if(nUnique1 < 256) { // byte org
			AMapToData m_char = MapToFactory.resizeForce(m, MAP_TYPE.CHAR);
			AMapToData m_int = MapToFactory.resizeForce(m, MAP_TYPE.INT);
			if(nUnique2 < 2) { // bit org
				AMapToData tm_byte = MapToFactory.resizeForce(tm, MAP_TYPE.BYTE);
				AMapToData tm_char = MapToFactory.resizeForce(tm, MAP_TYPE.CHAR);
				AMapToData tm_int = MapToFactory.resizeForce(tm, MAP_TYPE.INT);
				createFromList(tests, m, td, tof, nCol, exp, tm_byte, tm_char, tm_int);
				createFromList(tests, m_char, td, tof, nCol, exp, tm, tm_byte, tm_char, tm_int);
				createFromList(tests, m_int, td, tof, nCol, exp, tm, tm_byte, tm_char, tm_int);
			}
			else if(nUnique2 < 256) { // byte org
				AMapToData tm_char = MapToFactory.resizeForce(tm, MAP_TYPE.CHAR);
				AMapToData tm_int = MapToFactory.resizeForce(tm, MAP_TYPE.INT);
				createFromList(tests, m, td, tof, nCol, exp, tm_char, tm_int);
				createFromList(tests, m_char, td, tof, nCol, exp, tm, tm_char, tm_int);
				createFromList(tests, m_int, td, tof, nCol, exp, tm, tm_char, tm_int);
			}
			else { // char org
				AMapToData tm_int = MapToFactory.resizeForce(tm, MAP_TYPE.INT);
				createFromList(tests, m, td, tof, nCol, exp, tm_int);
				createFromList(tests, m_char, td, tof, nCol, exp, tm, tm_int);
				createFromList(tests, m_int, td, tof, nCol, exp, tm, tm_int);
			}
		}
		else { // char org
			AMapToData m_int = MapToFactory.resizeForce(m, MAP_TYPE.INT);
			if(nUnique2 < 2) { // bit org
				AMapToData tm_byte = MapToFactory.resizeForce(tm, MAP_TYPE.BYTE);
				AMapToData tm_char = MapToFactory.resizeForce(tm, MAP_TYPE.CHAR);
				AMapToData tm_int = MapToFactory.resizeForce(tm, MAP_TYPE.INT);
				createFromList(tests, m, td, tof, nCol, exp, tm_byte, tm_char, tm_int);
				createFromList(tests, m_int, td, tof, nCol, exp, tm, tm_byte, tm_char, tm_int);
			}
			else if(nUnique2 < 256) { // byte org
				AMapToData tm_char = MapToFactory.resizeForce(tm, MAP_TYPE.CHAR);
				AMapToData tm_int = MapToFactory.resizeForce(tm, MAP_TYPE.INT);

				createFromList(tests, m, td, tof, nCol, exp, tm_char, tm_int);
				createFromList(tests, m_int, td, tof, nCol, exp, tm, tm_char, tm_int);
			}
			else { // char org
				AMapToData tm_int = MapToFactory.resizeForce(tm, MAP_TYPE.INT);
				createFromList(tests, m, td, tof, nCol, exp, tm_int);
				createFromList(tests, m_int, td, tof, nCol, exp, tm, tm_int);
			}
		}
	}

	private static void createFromList(ArrayList<Object[]> tests, AMapToData m, ADictionary td, AOffset tof, int nCol,
		double[] exp, AMapToData... tm) {
		for(AMapToData tme : tm) {
			tests.add(new Object[] {m, tme, td, tof, nCol, exp});
		}
	}
}
