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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.indexes.ArrayIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex.SliceResult;
import org.apache.sysds.runtime.compress.colgroup.indexes.RangeIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.SingleIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.TwoIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.TwoRangesIndex;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.junit.Test;
import org.mockito.Mockito;

import scala.util.Random;

public class CustomIndexTest {
	@Test
	public void testSingeSlice1() {
		SingleIndex id = new SingleIndex(13);
		SliceResult r = id.slice(0, 14);
		assertEquals(0, r.idStart);
		assertEquals(1, r.idEnd);
		assertEquals(13, r.ret.get(0));
	}

	@Test
	public void testSingeSlice2() {
		SingleIndex id = new SingleIndex(13);
		SliceResult r = id.slice(2, 14);
		assertEquals(0, r.idStart);
		assertEquals(1, r.idEnd);
		assertEquals(13 - 2, r.ret.get(0));
	}

	@Test
	public void testSingeSlice3() {
		SingleIndex id = new SingleIndex(13);
		SliceResult r = id.slice(13, 14);
		assertEquals(0, r.idStart);
		assertEquals(1, r.idEnd);
		assertEquals(0, r.ret.get(0));
	}

	@Test
	public void testSingeSlice4() {
		SingleIndex id = new SingleIndex(13);
		SliceResult r = id.slice(14, 15);
		assertEquals(0, r.idStart);
		assertEquals(0, r.idEnd);
		assertEquals(null, r.ret);
	}

	@Test
	public void testSingeSlice5() {
		SingleIndex id = new SingleIndex(13);
		SliceResult r = id.slice(12, 13);
		assertEquals(0, r.idStart);
		assertEquals(0, r.idEnd);
		assertEquals(null, r.ret);
	}

	@Test
	public void testTwoSlice1() {
		testSlice1(new TwoIndex(10, 20));
	}

	@Test
	public void testArraySlice1() {
		testSlice1(new ArrayIndex(new int[] {10, 20}));
	}

	private void testSlice1(IColIndex i) {
		SliceResult r = i.slice(4, 24);

		assertEquals(0, r.idStart);
		assertEquals(2, r.idEnd);
		assertEquals(10 - 4, r.ret.get(0));
		assertEquals(20 - 4, r.ret.get(1));
	}

	@Test
	public void testTwoSlice2() {
		testSlice2(new TwoIndex(10, 20));
	}

	@Test
	public void testArraySlice2() {
		testSlice2(new ArrayIndex(new int[] {10, 20}));
	}

	private void testSlice2(IColIndex i) {
		SliceResult r = i.slice(11, 24);
		assertEquals(1, r.idStart);
		assertEquals(2, r.idEnd);
		assertEquals(20 - 11, r.ret.get(0));
	}

	@Test
	public void testTwoSlice3() {
		testSlice3(new TwoIndex(10, 20));
	}

	@Test
	public void testArraySlice3() {
		testSlice3(new ArrayIndex(new int[] {10, 20}));
	}

	private void testSlice3(IColIndex i) {
		SliceResult r = i.slice(5, 20);
		assertEquals(0, r.idStart);
		assertEquals(1, r.idEnd);
		assertEquals(10 - 5, r.ret.get(0));
	}

	@Test
	public void testTwoSlice4() {
		testSlice4(new TwoIndex(10, 20));
	}

	@Test
	public void testArraySlice4() {
		testSlice4(new ArrayIndex(new int[] {10, 20}));
	}

	private void testSlice4(IColIndex i) {
		SliceResult r = i.slice(5, 10);
		assertEquals(0, r.idStart);
		assertEquals(0, r.idEnd);
		assertEquals(null, r.ret);
	}

	@Test
	public void testTwoSlice5() {
		testSlice5(new TwoIndex(10, 20));
	}

	@Test
	public void testArraySlice5() {
		testSlice5(new ArrayIndex(new int[] {10, 20}));
	}

	private void testSlice5(IColIndex i) {
		SliceResult r = i.slice(21, 30);
		assertEquals(0, r.idStart);
		assertEquals(0, r.idEnd);
		assertEquals(null, r.ret);
	}

	@Test
	public void testTwoSlice6() {
		testSlice6(new TwoIndex(10, 20));
	}

	@Test
	public void testArraySlice6() {
		testSlice6(new ArrayIndex(new int[] {10, 20}));
	}

	private void testSlice6(IColIndex i) {
		SliceResult r = i.slice(0, 30);
		assertEquals(0, r.idStart);
		assertEquals(2, r.idEnd);
		assertEquals(10, r.ret.get(0));
		assertEquals(20, r.ret.get(1));
	}

	@Test
	public void testTwoSlice7() {
		testSlice7(new TwoIndex(10, 20));
	}

	@Test
	public void testArraySlice7() {
		testSlice7(new ArrayIndex(new int[] {10, 20}));
	}

	private void testSlice7(IColIndex i) {
		SliceResult r = i.slice(0, 15);
		assertEquals(0, r.idStart);
		assertEquals(1, r.idEnd);
		assertEquals(10, r.ret.get(0));
	}

	@Test
	public void testTwoSlice8() {
		testSlice8(new TwoIndex(10, 20));
	}

	@Test
	public void testArraySlice8() {
		testSlice8(new ArrayIndex(new int[] {10, 20}));
	}

	private void testSlice8(IColIndex i) {
		SliceResult r = i.slice(10, 15);
		assertEquals(0, r.idStart);
		assertEquals(1, r.idEnd);
		assertEquals(0, r.ret.get(0));
	}

	@Test
	public void testRangeSlice1() {
		SliceResult r = new RangeIndex(0, 10).slice(10, 15);
		assertEquals(0, r.idStart);
		assertEquals(0, r.idEnd);
		assertEquals(null, r.ret);
	}

	@Test
	public void testRangeSlice1AsArray() {
		SliceResult r = new ArrayIndex(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}).slice(10, 15);
		assertEquals(0, r.idStart);
		assertEquals(0, r.idEnd);
		assertEquals(null, r.ret);
	}

	@Test
	public void testRangeSlice2() {
		SliceResult r = new RangeIndex(15, 100).slice(10, 15);
		assertEquals(0, r.idStart);
		assertEquals(0, r.idEnd);
		assertEquals(null, r.ret);
	}

	@Test
	public void testRangeSlice3() {
		SliceResult r = new RangeIndex(14, 100).slice(10, 15);
		assertEquals(0, r.idStart);
		assertEquals(1, r.idEnd);
		assertEquals(14 - 10, r.ret.get(0));
	}

	@Test
	public void testRangeSlice3AsArray() {
		SliceResult r = new ArrayIndex(new int[] {14, 15, 16, 17, 18, 100}).slice(10, 15);
		assertEquals(0, r.idStart);
		assertEquals(1, r.idEnd);
		assertEquals(14 - 10, r.ret.get(0));
	}

	@Test
	public void findIndex1() {
		assertEquals(-1, new ArrayIndex(new int[] {14, 15, 16, 17, 18, 100}).findIndex(4));
	}

	@Test
	public void findIndex2() {
		assertEquals(0, new ArrayIndex(new int[] {14, 15, 16, 17, 18, 100}).findIndex(14));
	}

	@Test
	public void findIndex3() {
		assertEquals(1, new ArrayIndex(new int[] {14, 15, 16, 17, 18, 100}).findIndex(15));
	}

	@Test
	public void findIndex4() {
		assertEquals(4, new ArrayIndex(new int[] {14, 15, 16, 17, 18, 100}).findIndex(18));
	}

	@Test
	public void findIndex5() {
		assertEquals(-6, new ArrayIndex(new int[] {14, 15, 16, 17, 18, 100}).findIndex(19));
	}

	@Test
	public void findIndex6() {
		assertEquals(-7, new ArrayIndex(new int[] {14, 15, 16, 17, 18, 100}).findIndex(101));
	}

	@Test
	public void findIndex7() {
		assertEquals(-3, new ArrayIndex(new int[] {14, 18}).findIndex(101));
		assertEquals(-3, new TwoIndex(14, 18).findIndex(101));
	}

	@Test
	public void findIndex8() {
		assertEquals(-2, new ArrayIndex(new int[] {14, 18}).findIndex(15));
		assertEquals(-2, new TwoIndex(14, 18).findIndex(15));
	}

	@Test
	public void findIndex9() {
		assertEquals(-1, new ArrayIndex(new int[] {14, 18}).findIndex(4));
		assertEquals(-1, new TwoIndex(14, 18).findIndex(4));
	}

	@Test
	public void findIndex10() {
		assertEquals(0, new ArrayIndex(new int[] {14, 18}).findIndex(14));
		assertEquals(0, new TwoIndex(14, 18).findIndex(14));
	}

	@Test
	public void findIndex11() {
		assertEquals(1, new ArrayIndex(new int[] {14, 18}).findIndex(18));
		assertEquals(1, new TwoIndex(14, 18).findIndex(18));
	}

	@Test
	public void findIndex12() {
		assertEquals(0, new ArrayIndex(new int[] {14}).findIndex(14));
		assertEquals(0, new TwoIndex(14, 18).findIndex(14));
		assertEquals(0, new SingleIndex(14).findIndex(14));
	}

	@Test
	public void findIndex13() {
		assertEquals(-1, new ArrayIndex(new int[] {14}).findIndex(13));
		assertEquals(-1, new TwoIndex(14, 18).findIndex(13));
		assertEquals(-1, new SingleIndex(14).findIndex(13));
	}

	@Test
	public void findIndex14() {
		assertEquals(-2, new ArrayIndex(new int[] {14}).findIndex(15));
		assertEquals(-2, new TwoIndex(14, 18).findIndex(15));
		assertEquals(-2, new SingleIndex(14).findIndex(15));
	}

	@Test
	public void combine1() {
		IColIndex a = new SingleIndex(0);
		IColIndex b = new SingleIndex(1);
		IColIndex c = a.combine(b);
		assertTrue(c.contains(a, b));
	}

	@Test
	public void combine2() {
		IColIndex a = new SingleIndex(0);
		IColIndex b = new SingleIndex(2);
		IColIndex c = a.combine(b);
		assertTrue(c.contains(a, b));
	}

	@Test
	public void contains1() {
		IColIndex a = new SingleIndex(0);
		IColIndex b = new SingleIndex(2);
		assertFalse(b.contains(a, a));
		assertFalse(b.containsStrict(a, a));
	}

	@Test
	public void contains2() {
		IColIndex a = new SingleIndex(0);
		IColIndex b = new SingleIndex(2);
		assertFalse(b.contains(a, null));
		assertFalse(b.containsStrict(a, null));
	}

	@Test
	public void contains3() {
		IColIndex b = new SingleIndex(2);
		assertFalse(b.contains(null, null));
		assertFalse(b.containsStrict(null, null));
	}

	@Test
	public void contains4() {
		IColIndex a = new SingleIndex(0);
		IColIndex b = new SingleIndex(2);
		assertFalse(b.contains(null, a));
		assertFalse(b.containsStrict(null, a));
	}

	@Test
	public void contains5() {
		IColIndex a = new SingleIndex(0);
		IColIndex b = new SingleIndex(2);
		assertFalse(b.contains(a, b));
		assertFalse(b.containsStrict(a, b));
	}

	@Test
	public void contains6() {
		IColIndex a = new SingleIndex(0);
		IColIndex b = new SingleIndex(2);
		assertFalse(b.contains(b, a));
		assertFalse(b.containsStrict(b, a));
	}

	@Test
	public void contains7() {
		IColIndex a = new SingleIndex(1);
		IColIndex b = new SingleIndex(2);
		IColIndex c = new TwoIndex(1, 2);
		assertTrue(c.contains(a, b));
		assertTrue(c.containsStrict(a, b));
	}

	@Test
	public void contains8() {
		// IColIndex a = new SingleIndex(1);
		IColIndex b = new SingleIndex(2);
		IColIndex c = new TwoIndex(1, 2);
		assertTrue(c.contains(b, b));
		assertTrue(c.containsStrict(b, b));// could be considered wrong but is not verified in code.
	}

	@Test
	public void contains9() {
		// IColIndex a = new SingleIndex(1);
		IColIndex b = new TwoIndex(2, 3);
		IColIndex c = new ArrayIndex(new int[] {1, 2, 3});
		assertTrue(c.contains(b, b));
		assertFalse(c.containsStrict(b, b));
	}

	@Test
	public void contains10() {
		IColIndex a = new SingleIndex(1);
		IColIndex b = new TwoIndex(2, 3);
		IColIndex c = new ArrayIndex(new int[] {1, 2, 3});
		assertTrue(c.contains(a, b));
		assertTrue(c.containsStrict(a, b));
	}

	@Test
	public void contains11() {
		IColIndex a = new SingleIndex(2);
		IColIndex b = new TwoIndex(4, 5);
		IColIndex c = new ArrayIndex(new int[] {1, 2, 3});
		assertFalse(c.contains(a, b));
		assertFalse(c.containsStrict(a, b));
	}

	@Test
	public void contains12() {
		IColIndex a = new SingleIndex(2);
		IColIndex b = new TwoIndex(3, 5);
		IColIndex c = new ArrayIndex(new int[] {1, 2, 3});
		assertTrue(c.contains(a, b));
		assertFalse(c.containsStrict(a, b));
	}

	@Test
	public void contains13() {
		IColIndex a = new SingleIndex(10);
		IColIndex b = new TwoIndex(3, 5);
		IColIndex c = new ArrayIndex(new int[] {1, 2, 3});
		assertFalse(c.contains(a, b));
		assertFalse(c.containsStrict(a, b));
	}

	@Test
	public void toStringSlice() {
		assertTrue(null != new SingleIndex(10).slice(0, 100).toString());
	}

	@Test
	public void indexRange1() {
		IColIndex a = new RangeIndex(0, 1000);
		assertTrue(a.findIndex(1000) < 0);
	}

	@Test
	public void indexRange2() {
		IColIndex a = new RangeIndex(0, 1000);
		assertTrue(a.findIndex(10000) < 0);
	}

	@Test
	public void indexRange3() {
		IColIndex a = new RangeIndex(0, 1000);
		assertEquals(-1001, a.findIndex(10000));
	}

	@Test
	public void indexRange4() {
		assertEquals(-11, new RangeIndex(0, 10).findIndex(10000));
		assertEquals(-11, new ArrayIndex(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}).findIndex(10000));
	}

	@Test
	public void indexRange5() {
		assertEquals(3, new RangeIndex(0, 10).findIndex(3));
		assertEquals(3, new ArrayIndex(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}).findIndex(3));
	}

	@Test
	public void indexRange6() {
		assertEquals(-1, new RangeIndex(4, 10).findIndex(3));
		assertEquals(-1, new ArrayIndex(new int[] {4, 5, 6, 7, 8, 9}).findIndex(3));
	}

	@Test
	public void indexRange7() {
		assertEquals(0, new RangeIndex(4, 10).findIndex(4));
		assertEquals(0, new ArrayIndex(new int[] {4, 5, 6, 7, 8, 9}).findIndex(4));
	}

	@Test
	public void indexRange8() {
		assertEquals(4, new RangeIndex(4, 10).findIndex(8));
		assertEquals(4, new ArrayIndex(new int[] {4, 5, 6, 7, 8, 9}).findIndex(8));
	}

	@Test
	public void indexRange9() {
		assertEquals(-7, new RangeIndex(4, 10).findIndex(10));
		assertEquals(-7, new ArrayIndex(new int[] {4, 5, 6, 7, 8, 9}).findIndex(10));
	}

	@Test
	public void indexRange10() {
		assertEquals(-4, new RangeIndex(7, 10).findIndex(10));
		assertEquals(-4, new ArrayIndex(new int[] {7, 8, 9}).findIndex(10));
	}

	@Test
	public void combineTwo() {
		IColIndex a = new TwoIndex(1, 2);
		IColIndex b = new RangeIndex(5, 10);
		IColIndex c = a.combine(b);
		IColIndex ex = new ArrayIndex(new int[] {1, 2, 5, 6, 7, 8, 9});
		assertEquals(ex, c);
	}

	@Test
	public void combineColGroups1() {
		List<AColGroup> arr = new ArrayList<>(2);
		arr.add(Mockito.mock());
		Mockito.when(arr.get(0).getNumCols()).thenReturn(2);
		Mockito.when(arr.get(0).getColIndices()).thenReturn(new TwoIndex(1, 2));
		arr.add(Mockito.mock());
		Mockito.when(arr.get(1).getNumCols()).thenReturn(2);
		Mockito.when(arr.get(1).getColIndices()).thenReturn(new TwoIndex(3, 4));

		IColIndex com = ColIndexFactory.combine(arr);
		assertEquals(new RangeIndex(1, 5), com);
	}

	@Test
	public void combineColGroups2() {
		List<AColGroup> arr = new ArrayList<>(2);
		arr.add(Mockito.mock());
		Mockito.when(arr.get(0).getNumCols()).thenReturn(2);
		Mockito.when(arr.get(0).getColIndices()).thenReturn(new TwoIndex(1, 2));
		arr.add(Mockito.mock());
		Mockito.when(arr.get(1).getNumCols()).thenReturn(2);
		Mockito.when(arr.get(1).getColIndices()).thenReturn(new TwoIndex(3, 4));
		arr.add(Mockito.mock());
		Mockito.when(arr.get(2).getNumCols()).thenReturn(2);
		Mockito.when(arr.get(2).getColIndices()).thenReturn(new TwoIndex(5, 6));

		IColIndex com = ColIndexFactory.combine(arr);
		assertEquals(new RangeIndex(1, 7), com);
	}

	@Test
	public void compareOld() {
		final int r = 4;
		final int l = 10;
		final int[] colIdx1 = Util.genColsIndicesOffset(r, l);
		final IColIndex colIdx2 = ColIndexFactory.create(l, r + l);
		IndexesTest.compare(colIdx1, colIdx2);
	}

	@Test
	public void compareOld2() {
		final int domain = 32;
		int[] colIndexes = Util.genColsIndices(0, domain);
		IColIndex colIndexes2 = ColIndexFactory.create(0, domain);
		IndexesTest.compare(colIndexes, colIndexes2);
	}

	@Test
	public void getCombine_1() {
		IColIndex a = ColIndexFactory.createI(1, 2, 3);
		IColIndex b = ColIndexFactory.createI(4, 5, 6);
		IColIndex c = ColIndexFactory.combine(a, b);
		assertTrue(c.equals(ColIndexFactory.createI(1, 2, 3, 4, 5, 6)));
	}

	@Test
	public void getCombine_2() {
		IColIndex a = ColIndexFactory.createI(1, 3);
		IColIndex b = ColIndexFactory.createI(4, 5, 6);
		IColIndex c = ColIndexFactory.combine(a, b);
		assertTrue(c.equals(ColIndexFactory.createI(1, 3, 4, 5, 6)));
	}

	@Test
	public void getCombine_3() {
		IColIndex a = ColIndexFactory.createI(1, 3);
		IColIndex b = ColIndexFactory.createI(2, 5, 6);
		IColIndex c = ColIndexFactory.combine(a, b);
		assertTrue(c.equals(ColIndexFactory.createI(1, 2, 3, 5, 6)));
	}

	@Test
	public void getCombine_4() {
		IColIndex a = ColIndexFactory.createI(1, 3);
		IColIndex b = ColIndexFactory.createI(0, 2, 6);
		IColIndex c = ColIndexFactory.combine(a, b);
		assertTrue(c.equals(ColIndexFactory.createI(0, 1, 2, 3, 6)));
	}

	@Test
	public void getMapping_1() {
		IColIndex c = ColIndexFactory.createI(1, 2, 3);
		IColIndex a = ColIndexFactory.createI(1);
		IColIndex am = ColIndexFactory.getColumnMapping(c, a);
		assertTrue(am.equals(ColIndexFactory.createI(0)));
	}

	@Test
	public void getMapping_2() {
		IColIndex c = ColIndexFactory.createI(1, 2, 3);
		IColIndex a = ColIndexFactory.createI(2);
		IColIndex am = ColIndexFactory.getColumnMapping(c, a);
		assertTrue(am.equals(ColIndexFactory.createI(1)));
	}

	@Test
	public void getMapping_3() {
		IColIndex c = ColIndexFactory.createI(1, 3);
		IColIndex a = ColIndexFactory.createI(3);
		IColIndex am = ColIndexFactory.getColumnMapping(c, a);
		assertTrue(am.equals(ColIndexFactory.createI(1)));
	}

	@Test
	public void getMapping_4() {
		IColIndex c = ColIndexFactory.createI(1, 2, 3);
		IColIndex a = ColIndexFactory.createI(3);
		IColIndex am = ColIndexFactory.getColumnMapping(c, a);
		assertTrue(am.equals(ColIndexFactory.createI(2)));
	}

	@Test
	public void getMapping_5() {
		IColIndex c = ColIndexFactory.createI(1, 2, 3);
		IColIndex a = ColIndexFactory.createI(2, 3);
		IColIndex am = ColIndexFactory.getColumnMapping(c, a);
		assertTrue(am.equals(ColIndexFactory.createI(1, 2)));
	}

	@Test
	public void getMapping_6() {
		IColIndex c = ColIndexFactory.createI(1, 10, 100, 1000);
		IColIndex a = ColIndexFactory.createI(10, 1000);
		IColIndex am = ColIndexFactory.getColumnMapping(c, a);
		assertTrue(am.equals(ColIndexFactory.createI(1, 3)));
	}

	@Test
	public void rangeIndexSameEstimationCost() {
		assertEquals(ColIndexFactory.estimateMemoryCost(10000, true), ColIndexFactory.estimateMemoryCost(1000000, true));
	}

	@Test
	public void rangeIndexSameEstimationCost2() {
		assertEquals(ColIndexFactory.estimateMemoryCost(134, true), ColIndexFactory.estimateMemoryCost(4215, true));
	}

	@Test
	public void rangeIndexSameEstimationCost3() {
		assertEquals(ColIndexFactory.estimateMemoryCost(1000000, true), ColIndexFactory.estimateMemoryCost(4215, true));
	}

	@Test
	public void rangeIndexSameEstimationCost4() {
		assertTrue(ColIndexFactory.estimateMemoryCost(2, true) <= ColIndexFactory.estimateMemoryCost(4215, true));
	}

	@Test
	public void rangeIndexSameEstimationCost5() {
		assertTrue(ColIndexFactory.estimateMemoryCost(1, true) <= ColIndexFactory.estimateMemoryCost(4215, true));
	}

	@Test
	public void rangeIndexSameEstimationCost6() {
		assertTrue(ColIndexFactory.estimateMemoryCost(1, true) <= ColIndexFactory.estimateMemoryCost(2, true));
	}

	@Test
	public void rangeIndexSameEstimationCost7() {
		assertTrue(ColIndexFactory.estimateMemoryCost(1, false) <= ColIndexFactory.estimateMemoryCost(2, false));
	}

	@Test
	public void rangeIndexSameEstimationCost8() {
		assertTrue(ColIndexFactory.estimateMemoryCost(10000, true) < ColIndexFactory.estimateMemoryCost(20, false));
	}

	@Test
	public void estimateVSActual() {
		Random r = new Random(134);
		for(int i = 1; i < 50; i++) {
			IColIndex rl = ColIndexFactory.create(IndexesTest.randomInc(i, r.nextInt()));
			assertEquals(ColIndexFactory.estimateMemoryCost(i, false), rl.estimateInMemorySize());
		}
	}

	@Test(expected = DMLCompressionException.class)
	public void rangeReordering() {
		ColIndexFactory.create(0, 10).getReorderingIndex();
	}

	@Test(expected = DMLCompressionException.class)
	public void rangeSort() {
		ColIndexFactory.create(0, 10).sort();
	}

	@Test
	public void rangeIsSorted() {
		assertTrue(ColIndexFactory.create(0, 10).isSorted());
	}

	@Test(expected = DMLCompressionException.class)
	public void oneReordering() {
		ColIndexFactory.createI(0).getReorderingIndex();
	}

	@Test(expected = DMLCompressionException.class)
	public void oneSort() {
		ColIndexFactory.createI(10).sort();
	}

	@Test
	public void oneIsSorted() {
		assertTrue(ColIndexFactory.createI(10).isSorted());
	}

	@Test
	public void twoReordering1() {
		assertTrue(Arrays.equals(new int[] {0, 1}, ColIndexFactory.createI(1, 10).getReorderingIndex()));
	}

	@Test
	public void twoReordering2() {
		assertTrue(Arrays.equals(new int[] {1, 0}, ColIndexFactory.createI(10, 1).getReorderingIndex()));
	}

	@Test
	public void twoSort2() {
		assertTrue(ColIndexFactory.createI(1, 10).equals(ColIndexFactory.createI(10, 1).sort()));
	}

	@Test
	public void twoSort1() {
		assertTrue(ColIndexFactory.createI(1, 10).equals(ColIndexFactory.createI(1, 10).sort()));
	}

	@Test
	public void twoSorted1() {
		assertTrue(ColIndexFactory.createI(0, 10).isSorted());
	}

	@Test
	public void twoSorted2() {
		assertFalse(ColIndexFactory.createI(10, -1).isSorted());
	}

	@Test
	public void isSortedArray1() {
		assertTrue(ColIndexFactory.createI(0, 1, 5, 7, 9).isSorted());
	}

	@Test
	public void isSortedArray2() {
		assertFalse(new ArrayIndex(new int[] {0, 1, 5, 3, 9}).isSorted());
	}

	@Test
	public void isSortedArray3() {
		assertFalse(new ArrayIndex(new int[] {0, 1, 5, 9, -13}).isSorted());
	}

	@Test
	public void isSortedArray4() {
		assertFalse(new ArrayIndex(new int[] {0, 1, 0, 1, 0}).isSorted());
	}

	@Test
	public void combine_1() {
		IColIndex a = ColIndexFactory.createI(0, 1, 2, 3);
		IColIndex b = ColIndexFactory.createI(4, 5, 6, 7);
		IColIndex e = ColIndexFactory.createI(0, 1, 2, 3, 4, 5, 6, 7);
		assertEquals(e, a.combine(b));
	}

	@Test
	public void sortArray() {
		IColIndex a = new ArrayIndex(new int[] {6, 7, 3, 2, 8});
		IColIndex b = a.sort();
		IColIndex e = ColIndexFactory.createI(2, 3, 6, 7, 8);
		assertFalse(a.isSorted());
		assertTrue(b.isSorted());
		assertNotEquals(e, a);
		assertEquals(e, b);
	}

	@Test
	public void sortArray2() {
		IColIndex a = new ArrayIndex(new int[] {6, 7, 3, 2, 8});
		IColIndex b = a.sort();
		IColIndex e = ColIndexFactory.createI(2, 3, 6, 7, 8);
		assertFalse(a.isSorted());
		assertTrue(b.isSorted());
		assertNotEquals(e, a);
		assertEquals(e, b);
	}

	@Test
	public void getReorderingIndex() {
		IColIndex a = new ArrayIndex(new int[] {6, 4, 3, 2, 1});
		int[] b = a.getReorderingIndex();
		int[] e = new int[] {4, 3, 2, 1, 0};
		assertTrue(Arrays.equals(e, b));
	}

	@Test
	public void combineToRangeFromArray() {
		IColIndex a = ColIndexFactory.createI(0, 2, 4, 6, 8);
		IColIndex b = ColIndexFactory.createI(1, 3, 5, 7, 9);
		IColIndex e = ColIndexFactory.create(0, 10);
		assertEquals(e, a.combine(b));
	}

	@Test
	public void combineToRangeFromArray2() {
		IColIndex a = ColIndexFactory.createI(0, 2, 4, 6, 8);
		IColIndex b = ColIndexFactory.createI(1, 3, 5, 7);
		IColIndex e = ColIndexFactory.create(0, 9);
		assertEquals(e, a.combine(b));
	}

	@Test
	public void combineToRangeFromArray3() {
		IColIndex a = ColIndexFactory.createI(2, 4, 6, 8);
		IColIndex b = ColIndexFactory.createI(1, 3, 5, 7);
		IColIndex e = ColIndexFactory.create(1, 9);
		assertEquals(e, a.combine(b));
	}

	@Test
	public void combineToRangeFromArray4() {
		IColIndex a = ColIndexFactory.createI(2, 4, 6, 8);
		IColIndex b = ColIndexFactory.createI(1, 3, 5, 7, 9, 10, 11);
		IColIndex e = ColIndexFactory.create(1, 12);
		assertEquals(e, a.combine(b));
	}

	@Test
	public void avgIndex() {
		IColIndex a = ColIndexFactory.createI(2, 4, 6, 8);
		assertEquals(5.0, a.avgOfIndex(), 0.01);
	}

	@Test
	public void avgIndex2() {
		IColIndex a = ColIndexFactory.createI(2, 4, 6);
		assertEquals(4.0, a.avgOfIndex(), 0.01);
	}

	@Test
	public void avgIndex3() {
		IColIndex a = ColIndexFactory.createI(2, 6);
		assertEquals(4.0, a.avgOfIndex(), 0.01);
	}

	@Test
	public void avgIndex4() {
		IColIndex a = ColIndexFactory.createI(2);
		assertEquals(2.0, a.avgOfIndex(), 0.01);
	}

	@Test
	public void avgIndex5() {
		IColIndex a = ColIndexFactory.create(0, 10);
		assertEquals(4.5, a.avgOfIndex(), 0.01);
	}

	@Test
	public void combineColGroups() {
		AColGroup a = mock(AColGroup.class);
		when(a.getColIndices()).thenReturn(ColIndexFactory.createI(1, 2, 5, 6));
		AColGroup b = mock(AColGroup.class);
		when(b.getColIndices()).thenReturn(ColIndexFactory.createI(3, 4, 8));
		IColIndex e = ColIndexFactory.createI(1, 2, 3, 4, 5, 6, 8);
		assertEquals(e, ColIndexFactory.combine(a, b));
	}

	@Test
	public void combineArrayOfIndexes() {
		List<IColIndex> l = new ArrayList<>();
		l.add(ColIndexFactory.createI(1));
		l.add(ColIndexFactory.createI(3, 5));
		l.add(ColIndexFactory.createI(4, 7, 8, 9));
		l.add(ColIndexFactory.createI(10, 11, 12, 13, 14));

		IColIndex e = ColIndexFactory.createI(1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14);
		assertEquals(e, ColIndexFactory.combineIndexes(l));
	}

	@Test
	public void containsAny() {
		IColIndex a = ColIndexFactory.createI(27, 28, 29);
		IColIndex b = ColIndexFactory.createI(61, 62, 63);
		IColIndex c = a.combine(b);
		assertTrue(c instanceof TwoRangesIndex);

		IColIndex d = ColIndexFactory.createI(12);

		assertFalse(c.containsAny(d));
		assertFalse(d.containsAny(c));
	}

	@Test
	public void combineRanges() {
		IColIndex a = ColIndexFactory.createI(1, 2, 3, 4);
		IColIndex b = ColIndexFactory.createI(5, 6, 7, 8);
		IColIndex e = ColIndexFactory.createI(1, 2, 3, 4, 5, 6, 7, 8);
		assertEquals(e, a.combine(b));
	}

	@Test
	public void combineRanges2() {
		IColIndex b = ColIndexFactory.createI(1, 2, 3, 4);
		IColIndex a = ColIndexFactory.createI(5, 6, 7, 8);
		IColIndex e = ColIndexFactory.createI(1, 2, 3, 4, 5, 6, 7, 8);
		assertEquals(e, a.combine(b));
	}

	@Test
	public void combineRanges3() {
		IColIndex b = ColIndexFactory.createI(1, 2, 3, 4);
		IColIndex a = ColIndexFactory.createI(6, 7, 8, 9);
		IColIndex e = ColIndexFactory.createI(1, 2, 3, 4, 6, 7, 8, 9);
		assertEquals(e, a.combine(b));
	}

	@Test
	public void combineRanges4() {
		IColIndex a = ColIndexFactory.createI(1, 2, 3, 4);
		IColIndex b = ColIndexFactory.createI(6, 7, 8, 9);
		IColIndex e = ColIndexFactory.createI(1, 2, 3, 4, 6, 7, 8, 9);
		assertEquals(e, a.combine(b));
	}

	@Test
	public void containsTest() {
		// to get coverage
		IColIndex a = new TwoRangesIndex(new RangeIndex(1, 10), new RangeIndex(5, 10));
		assertTrue(a.contains(7));
		assertTrue(a.contains(2));
		assertTrue(a.contains(9));
		assertFalse(a.contains(-1));
		assertFalse(a.contains(11));
		assertFalse(a.contains(10));
	}

	@Test
	public void containsTest2() {
		// to get coverage
		IColIndex a = new TwoRangesIndex(new RangeIndex(1, 4), new RangeIndex(11, 20));
		assertFalse(a.contains(7));
		assertTrue(a.contains(2));
		assertTrue(a.contains(11));
		assertFalse(a.contains(-1));
		assertFalse(a.contains(20));
		assertFalse(a.contains(10));
	}

	@Test
	public void containsAnyArray1() {
		IColIndex a = new TwoRangesIndex(new RangeIndex(1, 4), new RangeIndex(11, 20));
		IColIndex b = new RangeIndex(7, 15);
		assertTrue(a.containsAny(b));
	}

	@Test
	public void containsAnyArrayF1() {
		IColIndex a = new TwoRangesIndex(new RangeIndex(1, 4), new RangeIndex(11, 20));
		IColIndex b = new RangeIndex(20, 25);
		assertFalse(a.containsAny(b));
	}

	@Test
	public void containsAnyArrayF2() {
		IColIndex a = new TwoRangesIndex(new RangeIndex(1, 4), new RangeIndex(11, 20));
		IColIndex b = new RangeIndex(4, 11);
		assertFalse(a.containsAny(b));
	}

	@Test
	public void containsAnyArray2() {
		IColIndex a = new TwoRangesIndex(new RangeIndex(1, 4), new RangeIndex(11, 20));
		IColIndex b = new RangeIndex(3, 11);
		assertTrue(a.containsAny(b));
	}

	@Test
	public void reordering1(){
		IColIndex a = ColIndexFactory.createI(1,3,5);
		IColIndex b = ColIndexFactory.createI(2);

		assertFalse(IColIndex.inOrder(a, b));
		Pair<int[], int[]> r = IColIndex.reorderingIndexes(a, b);
		
		int[] ra = r.getKey();
		int[] rb = r.getValue();

		assertArrayEquals(new int[]{0,2,3}, ra);
		assertArrayEquals(new int[]{1}, rb);
	}

	@Test
	public void reordering2(){
		IColIndex a = ColIndexFactory.createI(1,3,5);
		IColIndex b = ColIndexFactory.createI(2,4);

		assertFalse(IColIndex.inOrder(a, b));
		Pair<int[], int[]> r = IColIndex.reorderingIndexes(a, b);
		
		int[] ra = r.getKey();
		int[] rb = r.getValue();

		assertArrayEquals(new int[]{0,2,4}, ra);
		assertArrayEquals(new int[]{1,3}, rb);
	}

	@Test
	public void reordering3(){
		IColIndex a = ColIndexFactory.createI(1,3,5);
		IColIndex b = ColIndexFactory.createI(0, 2,4);

		assertFalse(IColIndex.inOrder(a, b));
		Pair<int[], int[]> r = IColIndex.reorderingIndexes(a, b);
		
		int[] ra = r.getKey();
		int[] rb = r.getValue();

		assertArrayEquals(new int[]{1,3,5}, ra);
		assertArrayEquals(new int[]{0,2,4}, rb);
	}

	@Test
	public void reordering4(){
		IColIndex a = ColIndexFactory.createI(1,5);
		IColIndex b = ColIndexFactory.createI(0,2,3,4);

		assertFalse(IColIndex.inOrder(a, b));
		Pair<int[], int[]> r = IColIndex.reorderingIndexes(a, b);
		
		int[] ra = r.getKey();
		int[] rb = r.getValue();

		assertArrayEquals(new int[]{1,5}, ra);
		assertArrayEquals(new int[]{0,2,3,4}, rb);
	}
}
