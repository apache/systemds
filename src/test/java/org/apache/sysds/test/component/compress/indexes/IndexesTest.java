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
import static org.junit.Assert.assertNotEquals;
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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.ArrayIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.CombinedIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex.SliceResult;
import org.apache.sysds.runtime.compress.colgroup.indexes.IIterate;
import org.apache.sysds.runtime.compress.colgroup.indexes.RangeIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.SingleIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.TwoIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.TwoRangesIndex;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.utils.MemoryEstimates;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class IndexesTest {
	public static final Log LOG = LogFactory.getLog(IndexesTest.class.getName());

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

			tests.add(new Object[] {//
				new int[] {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}, //
				ColIndexFactory.create(4, 19)});
			tests.add(new Object[] {//
				new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}, //
				ColIndexFactory.create(0, 19)});
			tests.add(new Object[] {//
				new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}, //
				ColIndexFactory.create(1, 19)});

			tests.add(new Object[] {//
				new int[] {1, 2, 3, 4, 5, 6}, //
				ColIndexFactory.create(1, 7)});
			tests.add(new Object[] {//
				new int[] {2, 3, 4, 5, 6}, //
				ColIndexFactory.create(2, 7)});
			tests.add(new Object[] {//
				new int[] {3, 4, 5, 6}, //
				ColIndexFactory.create(3, 7)});

			tests.add(createWithArray(1, 323));
			tests.add(createWithArray(2, 1414));
			tests.add(createWithArray(144, 32));
			tests.add(createWithArray(13, 23));
			tests.add(createWithArray(145, 14));
			tests.add(createWithArray(300, 14));
			tests.add(createWithArray(23, 51515));
			tests.add(createWithArray(128, 321));
			tests.add(createWithArray(129, 1324));
			tests.add(createWithArray(127, 1323));
			tests.add(createWithArray(66, 132));
			tests.add(createRangeWithArray(66, 132));
			tests.add(createRangeWithArray(32, 132));
			tests.add(createRangeWithArray(13, 132));
			tests.add(createRangeWithArray(4, 132));
			tests.add(createRangeWithArray(2, 132));
			tests.add(createRangeWithArray(1, 132));
			tests.add(createTwoRange(1, 10, 20, 30));
			tests.add(createTwoRange(1, 10, 22, 30));
			tests.add(createTwoRange(9, 11, 22, 30));
			tests.add(createTwoRange(9, 11, 22, 60));
			tests.add(createCombined(9, 11, 22));
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

			long actualSize = bos.size();
			// Serialize in
			ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
			DataInputStream fis = new DataInputStream(bis);

			IColIndex n = ColIndexFactory.read(fis);
			long expectedSize = actual.getExactSizeOnDisk();

			compare(actual, n);
			assertEquals(actual.toString(), expectedSize, actualSize);
		}
		catch(IOException e) {
			throw new RuntimeException("Error in io " + actual, e);
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

			assertEquals(actual.toString(), expectedSize, actualSize);
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
		if(expected[0] < 1) {

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
	public void slice_3() {
		for(int e = 0; e < actual.size(); e++) {

			SliceResult sr = actual.slice(expected[e], expected[expected.length - 1] + 1);
			String errStr = actual.toString();
			if(sr.ret != null) {
				IColIndex a = sr.ret;
				assertEquals(errStr, a.size(), actual.size() - e);
				assertEquals(errStr, a.get(0), 0);
				assertEquals(errStr, a.get(a.size() - 1), expected[expected.length - 1] - expected[e]);
			}
		}
	}

	@Test
	public void slice_4() {
		SliceResult sr = actual.slice(-10, -1);
		assertEquals(null, sr.ret);
		assertEquals(0, sr.idEnd);
		assertEquals(0, sr.idStart);
	}

	@Test
	public void slice_5_moreThanRange() {
		SliceResult sr = actual.slice(-10, expected[expected.length - 1] + 10);
		assertTrue(sr.toString() + "  " + actual, sr.ret.contains(expected[0] + 10));
		assertEquals(0, sr.idStart);
	}

	@Test
	public void slice_5_SubRange() {
		if(expected.length > 5) {

			SliceResult sr = actual.slice(4, expected[5] + 1);

			assertEquals(actual.toString(), expected[5], sr.ret.get(sr.ret.size() - 1) + 4);
		}
	}

	@Test
	public void equals() {
		assertEquals(actual, ColIndexFactory.create(expected));
	}

	@Test
	public void equalsSizeDiff_range() {
		if(actual.size() == 10)
			return;

		IColIndex a = new RangeIndex(0, 10);
		assertNotEquals(actual, a);
	}

	@Test
	public void equalsSizeDiff_twoRanges() {
		if(actual.size() == 10)
			return;

		IColIndex a = new TwoRangesIndex(new RangeIndex(0, 5), new RangeIndex(6, 10));
		assertNotEquals(actual, a);
	}

	@Test
	public void equalsSizeDiff_twoRanges2() {
		if(actual.size() == 10 + 3)
			return;
		RangeIndex a = new RangeIndex(1, 10);
		RangeIndex b = new RangeIndex(22, 25);
		TwoRangesIndex c = (TwoRangesIndex) a.combine(b);
		assertNotEquals(actual, c);
	}

	@Test
	public void equalsCombine(){
		RangeIndex a = new RangeIndex(9, 11);
		SingleIndex b = new SingleIndex(22);
		IColIndex c = a.combine(b);
		if(eq(expected, c)){
			LOG.error(c.size());
			compare(expected, c);
			compare(c, actual);
		}

	}

	@Test
	public void equalsItself() {
		assertEquals(actual, actual);
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
		try {
			IColIndex b = new TwoIndex(expected[0] - 1, expected[expected.length - 1] + 1);
			IColIndex c = actual.combine(b);
			assertTrue(c.containsStrict(actual, b));
			assertTrue(c.containsStrict(b, actual));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
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
		if(!(actual instanceof TwoRangesIndex))
			assertEquals(actual.hashCode(), ColIndexFactory.create(expected).hashCode());
	}

	@Test
	public void estimateInMemorySizeIsNotToBig() {
		if(actual instanceof CombinedIndex)
			assertTrue(MemoryEstimates.intArrayCost(expected.length) >= actual.estimateInMemorySize() - 64);
		else
			assertTrue(MemoryEstimates.intArrayCost(expected.length) >= actual.estimateInMemorySize() - 16);
	}

	@Test
	public void containsInt1() {
		assertTrue(actual.contains(expected[0]));
	}

	@Test
	public void containsInt2() {
		assertTrue(actual.contains(expected[expected.length - 1]));
	}

	@Test
	public void containsIntAllElements() {
		for(int i = 0; i < expected.length; i++)
			assertTrue(actual.contains(expected[i]));
	}

	@Test
	public void containsIntNot1() {
		assertFalse(actual.contains(expected[expected.length - 1] + 3));
	}

	@Test
	public void containsIntNot2() {
		assertFalse(actual.toString(), actual.contains(expected[0] - 1));
	}

	@Test
	public void containsIntNotAllInbetween() {
		int j = 0;
		for(int i = expected[0]; i < expected[expected.length - 1]; i++) {
			if(i == expected[j]) {
				j++;
				assertTrue(actual.toString(), actual.contains(i));
			}
			else {
				assertFalse(actual.toString(), actual.contains(i));
			}
		}
	}

	@Test
	public void containsAnySingle() {
		assertTrue(actual.containsAny(new SingleIndex(expected[expected.length - 1])));
	}

	@Test
	public void containsAnySingleFalse1() {
		assertFalse(actual.containsAny(new SingleIndex(expected[expected.length - 1] + 1)));
	}

	@Test
	public void containsAnySingleFalse2() {
		assertFalse(actual.containsAny(new SingleIndex(expected[0] - 1)));
	}

	@Test
	public void containsAnyTwo() {
		assertTrue(actual.containsAny(new TwoIndex(expected[expected.length - 1], expected[expected.length - 1] + 4)));
	}

	@Test
	public void containsAnyTwoFalse() {
		assertFalse(
			actual.containsAny(new TwoIndex(expected[expected.length - 1] + 1, expected[expected.length - 1] + 4)));
	}

	@Test
	public void iteratorsV() {
		IIterate i = actual.iterator();
		while(i.hasNext()) {
			int v = i.v();
			assertEquals(actual.toString(), v, i.next());
		}
	}

	@Test
	public void averageOfIndex() {
		double a = actual.avgOfIndex();
		double s = 0.0;
		for(int i = 0; i < expected.length; i++)
			s += expected[i];

		assertEquals(actual.toString(), s / expected.length, a, 0.0000001);
	}

	@Test
	public void isSorted() {
		assertTrue(actual.isSorted());
	}

	@Test
	public void sort() {
		assertTrue(actual.isSorted());
		try {

			actual.sort();// should do nothing
		}
		catch(DMLCompressionException e) {
			// okay
		}
		assertTrue(actual.isSorted());
	}

	@Test
	public void getReorderingIndex() {
		try {

			int[] ro = actual.getReorderingIndex();
			if(ro != null) {
				for(int i = 0; i < ro.length - 1; i++) {
					assertTrue(ro[i] < ro[i + 1]);
				}
			}
		}
		catch(DMLCompressionException e) {
			// okay
		}
	}

	@Test
	public void findIndexBefore() {
		final String er = actual.toString();
		assertEquals(er, -1, actual.findIndex(expected[0] - 1));
		assertEquals(er, -1, actual.findIndex(expected[0] - 10));
		assertEquals(er, -1, actual.findIndex(expected[0] - 100));
	}

	@Test
	public void findIndexAll() {
		final String er = actual.toString();
		for(int i = 0; i < expected.length; i++) {
			assertEquals(er, i, actual.findIndex(expected[i]));
		}
	}

	@Test
	public void findIndexAllMinus1() {
		final String er = actual.toString();
		for(int i = 1; i < expected.length; i++) {
			if(expected[i - 1] == expected[i] - 1) {
				assertEquals(er, i - 1, actual.findIndex(expected[i] - 1));
			}
			else {
				assertEquals(er, i * -1 - 1, actual.findIndex(expected[i] - 1));

			}
		}
	}

	@Test
	public void findIndexAfter() {
		final int el = expected.length;
		final String er = actual.toString();
		assertEquals(er, -el - 1, actual.findIndex(expected[el - 1] + 1));
		assertEquals(er, -el - 1, actual.findIndex(expected[el - 1] + 10));
		assertEquals(er, -el - 1, actual.findIndex(expected[el - 1] + 100));
	}

	@Test
	public void testHash() {
		// flawed test in the case hashes can collide, but it should be unlikely.
		IColIndex a = ColIndexFactory.createI(1, 2, 3, 1342);
		if(a.equals(actual)) {
			assertEquals(a.hashCode(), actual.hashCode());
		}
		else {
			assertNotEquals(a.hashCode(), actual.hashCode());
		}
	}

	private void shift(int i) {
		compare(expected, actual.shift(i), i);
	}

	private static boolean eq(int[] expected, IColIndex actual) {
		if(expected.length == actual.size()) {
			for(int i = 0; i < expected.length; i++)
				if(expected[i] != actual.get(i))
					return false;
			return true;
		}
		else
			return false;
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

	private static Object[] createTwoRange(int l1, int u1, int l2, int u2) {
		RangeIndex a = new RangeIndex(l1, u1);
		RangeIndex b = new RangeIndex(l2, u2);
		TwoRangesIndex c = (TwoRangesIndex) a.combine(b);
		int[] exp = new int[u1 - l1 + u2 - l2];
		for(int i = l1, j = 0; i < u1; i++, j++)
			exp[j] = i;
		for(int i = l2, j = u1 - l1; i < u2; i++, j++)
			exp[j] = i;
		return new Object[] {exp, c};
	}

	private static Object[] createCombined(int l1, int u1, int o) {
		RangeIndex a = new RangeIndex(l1, u1);
		SingleIndex b = new SingleIndex(o);
		IColIndex c = a.combine(b);
		int[] exp = new int[u1 - l1 + 1];

		for(int i = l1, j = 0; i < u1; i++, j++)
			exp[j] = i;

		exp[exp.length - 1] = o;

		return new Object[] {exp, c};

	}
}
