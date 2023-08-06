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

package org.apache.sysds.test.component.compress.indexes;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.indexes.ArrayIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex.SliceResult;
import org.apache.sysds.runtime.compress.colgroup.indexes.IIterate;
import org.apache.sysds.runtime.compress.colgroup.indexes.SingleIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.TwoIndex;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.utils.MemoryEstimates;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class IndexesTest {

	private final int[] expected;
	private final IColIndex actual;

	@Parameters
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();

		try {
			// single
			tests.add(new Object[] {new int[] {0}, new SingleIndex(0)});
			tests.add(new Object[] {new int[] {334}, ColIndexFactory.create(334, 335)});
			tests.add(new Object[] {new int[] {0}, ColIndexFactory.create(1)});
			tests.add(new Object[] {new int[] {0}, ColIndexFactory.create(new int[] {0})});
			tests.add(new Object[] {new int[] {320}, ColIndexFactory.create(new int[] {320})});

			// two
			tests.add(new Object[] {new int[] {0, 1}, new TwoIndex(0, 1)});
			tests.add(new Object[] {new int[] {3214, 44444}, new TwoIndex(3214, 44444)});
			tests.add(new Object[] {new int[] {3214, 44444}, ColIndexFactory.create(new int[] {3214, 44444})});
			tests.add(new Object[] {new int[] {3214, 3215}, ColIndexFactory.create(3214, 3216)});
			tests.add(new Object[] {new int[] {0, 1}, ColIndexFactory.create(2)});

			// array
			tests.add(create(32, 14));
			tests.add(create(40, 21));
			tests.add(new Object[] {//
				new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, //
				ColIndexFactory.create(0, 10)});

			tests.add(new Object[] {//
				new int[] {0, 1, 2, 3}, //
				ColIndexFactory.create(0, 4)});

			tests.add(new Object[] {//
				new int[] {0, 1, 2, 3}, //
				ColIndexFactory.create(4)});

			tests.add(new Object[] {//
				new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, //
				ColIndexFactory.create(10)});

			tests.add(new Object[] {//
				new int[] {4, 5, 6, 7, 8, 9}, //
				ColIndexFactory.create(4, 10)});

			tests.add(createWithArray(1, 323));
			tests.add(createWithArray(2, 1414));
			tests.add(createWithArray(144, 32));
			tests.add(createWithArray(13, 23));
			tests.add(createWithArray(145, 14));
			tests.add(createWithArray(23, 51515));
			tests.add(createWithArray(66, 132));
			tests.add(createRangeWithArray(66, 132));
			tests.add(createRangeWithArray(32, 132));
			tests.add(createRangeWithArray(13, 132));
			tests.add(createRangeWithArray(4, 132));
			tests.add(createRangeWithArray(2, 132));
			tests.add(createRangeWithArray(1, 132));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	public IndexesTest(int[] expected, IColIndex actual) {
		this.expected = expected;
		this.actual = actual;
	}

	@Test
	public void testGet() {
		for(int i = 0; i < expected.length; i++) {
			assertEquals(expected[i], actual.get(i));
		}
	}

	@Test
	public void testSerialize() {
		try {
			// Serialize out
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			actual.write(fos);

			// Serialize in
			ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
			DataInputStream fis = new DataInputStream(bis);

			IColIndex n = ColIndexFactory.read(fis);

			compare(actual, n);
		}
		catch(IOException e) {
			throw new RuntimeException("Error in io", e);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void testSerializeSize() {
		try {
			// Serialize out
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			actual.write(fos);

			long actualSize = bos.size();
			long expectedSize = actual.getExactSizeOnDisk();

			assertEquals(expectedSize, actualSize);
		}
		catch(IOException e) {
			throw new RuntimeException("Error in io", e);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void testSize() {
		assertEquals(expected.length, actual.size());
	}

	@Test
	public void iterator() {
		compare(expected, actual.iterator());
	}

	@Test
	public void factoryCreate() {
		compare(expected, ColIndexFactory.create(expected));
	}

	@Test
	public void shift() {
		shift(5);
	}

	@Test
	public void shift2() {
		shift(1342);
	}

	@Test
	public void slice() {
		SliceResult sr = actual.slice(-10, expected[expected.length - 1] + 1);
		String errStr = actual.toString();
		assertEquals(errStr, 0, sr.idStart);
		assertEquals(errStr, expected.length, sr.idEnd);
		assertEquals(errStr, expected[0] + 10, sr.ret.get(0));
	}

	@Test
	public void slice_1() {
		if(expected[0] > 1) {

			SliceResult sr = actual.slice(1, expected[expected.length - 1] + 1);
			String errStr = actual.toString();
			assertEquals(errStr, 0, sr.idStart);
			assertEquals(errStr, expected.length, sr.idEnd);
			assertEquals(errStr, expected[0] - 1, sr.ret.get(0));
		}
	}

	@Test
	public void slice_2() {
		if(expected[0] <= 1) {

			SliceResult sr = actual.slice(1, expected[expected.length - 1] + 1);
			String errStr = actual.toString();
			if(sr.ret != null) {
				assertEquals(errStr, 1, sr.idStart);
				assertEquals(errStr, expected.length, sr.idEnd);
				assertEquals(errStr, expected[1] - 1, sr.ret.get(0));
			}
		}
	}

	@Test
	public void equals() {
		assertEquals(actual, ColIndexFactory.create(expected));
	}

	@Test
	public void isContiguous() {
		boolean c = expected[expected.length - 1] - expected[0] + 1 == expected.length;
		assertEquals(c, actual.isContiguous());
	}

	@Test
	public void combineSingleOneAbove() {
		IColIndex b = new SingleIndex(expected[expected.length - 1] + 1);
		IColIndex c = actual.combine(b);
		assertTrue(c.containsStrict(actual, b));
		assertTrue(c.containsStrict(b, actual));
	}

	@Test
	public void combineSingleOneBellow() {
		IColIndex b = new SingleIndex(expected[0] - 1);
		IColIndex c = actual.combine(b);
		assertTrue(c.containsStrict(actual, b));
		assertTrue(c.containsStrict(b, actual));
	}

	@Test
	public void combineSingleHighAbove() {
		IColIndex b = new SingleIndex(expected[expected.length - 1] + 1342);
		IColIndex c = actual.combine(b);
		assertTrue(c.containsStrict(actual, b));
		assertTrue(c.containsStrict(b, actual));
	}

	@Test
	public void combineTwoAbove() {
		int oa = expected[expected.length - 1] + 1;
		IColIndex b = new TwoIndex(oa, oa + 1);
		IColIndex c = actual.combine(b);
		assertTrue(c.containsStrict(actual, b));
		assertTrue(c.containsStrict(b, actual));
	}

	@Test
	public void combineTwoAround() {
		IColIndex b = new TwoIndex(expected[0] - 1, expected[expected.length - 1] + 1);
		IColIndex c = actual.combine(b);
		assertTrue(c.containsStrict(actual, b));
		assertTrue(c.containsStrict(b, actual));
	}

	@Test
	public void combineTwoBellow() {
		IColIndex b = new TwoIndex(expected[0] - 2, expected[0] - 1);
		IColIndex c = actual.combine(b);
		assertTrue(c.containsStrict(actual, b));
		assertTrue(c.containsStrict(b, actual));
	}

	@Test
	public void hashCodeEquals() {
		assertEquals(actual.hashCode(), ColIndexFactory.create(expected).hashCode());
	}

	@Test
	public void estimateInMemorySizeIsNotToBig() {
		assertTrue(MemoryEstimates.intArrayCost(expected.length) >= actual.estimateInMemorySize() - 16);
	}

	private void shift(int i) {
		compare(expected, actual.shift(i), i);
	}

	public static void compare(int[] expected, IColIndex actual) {
		assertEquals(expected.length, actual.size());
		for(int i = 0; i < expected.length; i++)
			assertEquals(expected[i], actual.get(i));
	}

	private static void compare(int[] expected, IColIndex actual, int off) {
		assertEquals(expected.length, actual.size());
		for(int i = 0; i < expected.length; i++)
			assertEquals(expected[i] + off, actual.get(i));
	}

	private static void compare(IColIndex expected, IColIndex actual) {
		assertEquals(expected.size(), actual.size());
		for(int i = 0; i < expected.size(); i++)
			assertEquals(expected.get(i), actual.get(i));
	}

	private static void compare(int[] expected, IIterate actual) {
		for(int i = 0; i < expected.length; i++) {
			assertTrue(actual.hasNext());
			assertEquals(i, actual.i());
			assertEquals(expected[i], actual.next());
		}
		assertFalse(actual.hasNext());
	}

	private static Object[] create(int size, int seed) {
		int[] cols = new int[size];
		Random r = new Random(seed);
		cols[0] = r.nextInt(1000) + 1;
		for(int i = 1; i < size; i++) {
			cols[i] = cols[i - 1] + r.nextInt(1000) + 1;
		}
		return new Object[] {cols, ColIndexFactory.create(cols)};
	}

	private static Object[] createWithArray(int size, int seed) {
		IntArrayList cols = randomInc(size, seed);
		return new Object[] {cols.extractValues(true), ColIndexFactory.create(cols)};
	}

	public static IntArrayList randomInc(int size, int seed) {
		IntArrayList cols = new IntArrayList();
		Random r = new Random(seed);
		cols.appendValue(r.nextInt(1000) + 1);
		for(int i = 1; i < size; i++) {
			int prev = cols.get(i - 1);
			cols.appendValue(r.nextInt(1000) + 1 + prev);
		}
		return cols;
	}

	private static Object[] createRangeWithArray(int size, int seed) {
		IntArrayList cols = new IntArrayList();
		Random r = new Random(seed);
		cols.appendValue(r.nextInt(1000) + 1);
		for(int i = 1; i < size; i++) {
			int prev = cols.get(i - 1);
			cols.appendValue(1 + prev);
		}
		IColIndex ret = ColIndexFactory.create(cols);
		if(!(ret instanceof ArrayIndex))
			return new Object[] {cols.extractValues(true), ret};
		else
			throw new DMLRuntimeException("Invalid construction of range array");
	}
}
