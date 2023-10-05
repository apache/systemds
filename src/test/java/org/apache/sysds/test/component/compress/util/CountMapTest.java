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

package org.apache.sysds.test.component.compress.util;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.utils.ACount;
import org.apache.sysds.runtime.compress.utils.DoubleCountHashMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class CountMapTest {
	protected static final Log LOG = LogFactory.getLog(CountMapTest.class.getName());

	final int orgSize;
	final DoubleCountHashMap m;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		tests.add(new Object[] {new DoubleCountHashMap(), 3});
		tests.add(new Object[] {new DoubleCountHashMap(), 7});
		tests.add(new Object[] {new DoubleCountHashMap(51), 7});
		tests.add(new Object[] {new DoubleCountHashMap(100), 7});
		return tests;
	}

	public CountMapTest(DoubleCountHashMap m, int orgSize) {
		this.m = m;
		this.orgSize = orgSize;
	}

	@Before
	public void setup() {
		m.reset(orgSize);
	}

	@Test
	public void testDoubleCountHashMap1() {
		m.increment(1.0);
		m.increment(1.0);
		assertEquals(m.get(1.0), 2);
	}

	@Test
	public void testDoubleCountHashMap2() {
		m.increment(1.0);
		assertEquals(m.get(1.0), 1);

	}

	@Test
	public void testDoubleCountHashMap3() {
		for(int i = 0; i < 100; i++) {
			m.increment(1.0);
		}
		assertEquals(m.get(1.0), 100);

	}

	@Test
	public void testDoubleCountHashMap4() {
		for(int i = 0; i < 100; i++) {
			assertEquals(i, m.size());
			m.increment(i);
			assertEquals(i + 1, m.size());
		}
		assertEquals(m.get(1.0), 1);
		assertEquals(m.get(87.0), 1);

	}

	@Test
	public void testDoubleCountHashMap5() {
		try {

			for(int i = 0; i < 100; i++) {
				assertEquals(i, m.size());
				m.increment(i);
				assertEquals(i + 1, m.size());
			}
			assertEquals(m.get(1.0), 1);
			assertEquals(m.get(87.0), 1);
			assertEquals(-1, m.getOrDefault(140.0, -1));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testDoubleCountHashMap6() {
		for(int i = 0; i < 5; i++) {
			assertEquals(i, m.size());
			m.increment(i);
			assertEquals(i + 1, m.size());
		}
		assertEquals(1, m.getOrDefault(4.0, -1));
	}

	@Test
	public void sizeIncrease() {
		for(int i = 0; i < 100; i++) {
			assertEquals(i, m.size());
			m.increment(i);
			assertEquals(i + 1, m.size());
		}
	}

	@Test
	public void extractValues() {
		for(int i = 0; i < 100; i++) {
			m.increment(i);
		}
		ACount<Double>[] vals = m.extractValues();
		Arrays.sort(vals, Comparator.comparing((x) -> x.key()));
		for(int i = 0; i < 100; i++) {
			assertEquals(1, vals[i].count);
			assertEquals(i, vals[i].key(), 0.0);
		}
	}

	@Test
	public void extractValuesAfterSort() {
		for(int i = 0; i < 100; i++) {
			m.increment(i);
		}
		m.sortBuckets();
		ACount<Double>[] vals = m.extractValues();
		Arrays.sort(vals, Comparator.comparing((x) -> x.key()));
		for(int i = 0; i < 100; i++) {
			assertEquals(1, vals[i].count);
			assertEquals(i, vals[i].key(), 0.0);
		}
	}

	@Test
	public void complicatedExample() {
		for(int i = 0; i < 100; i++)
			for(int j = i; j < 100; j++)
				m.increment(j);
		assertEquals(100, m.size());
		for(int i = 0; i < 100; i++) {
			assertEquals("expect " + (i + 1) + " got: " + m.get((double) i) + " " + m, i + 1, m.get((double) i));
		}
	}

	@Test(expected = NullPointerException.class)
	public void getIdNull() {
		m.getId(1.0);
	}

	@Test()
	public void getId1() {
		m.increment(1.0);
		assertEquals(0.0, m.getId(1.0), 0.0);
	}

	@Test()
	public void getId2() {
		for(int i = 0; i < 20; i++) {
			m.increment(i);
		}
		assertEquals(19, m.getId(19.0));
	}

	@Test()
	public void sortBucketsSmall() {
		for(int i = 0; i < 9; i++)
			m.increment(i);

		m.sortBuckets(); // should not really do anything to behaviour
		assertEquals(4, m.getId(4.0));
		assertEquals(7, m.getId(7.0));
	}

	@Test
	public void getDictionary() {
		for(int i = 0; i < 9; i++)
			m.increment(i);

		double[] d = m.getDictionary();
		for(int i = 0; i < 9; i++)
			assertEquals(i, d[i], 0.0);
	}

	@Test()
	public void getNan() {
		m.increment(Double.NaN);
		assertEquals(0, m.getId(Double.NaN));
		m.increment(Double.NaN);
		assertEquals(0, m.getId(Double.NaN));
		m.increment(1.0);
		assertEquals(0, m.getId(Double.NaN));
		m.increment(Double.NaN);
		assertEquals(0, m.getId(Double.NaN));
	}
}
