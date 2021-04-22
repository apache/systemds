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

package org.apache.sysds.test.component.compress.insertionsorter;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.tree.AInsertionSorter;
import org.apache.sysds.runtime.compress.colgroup.tree.MergeSort;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.junit.Test;

public class MergeSortTest {
	// private static final Log LOG = LogFactory.getLog(MergeSortTest.class.getName());

	@Test
	public void testInsertionSingleValue_01() {
		AInsertionSorter t = new MergeSort(1, 1, 1);
		IntArrayList[] array = new IntArrayList[1];
		array[0] = new IntArrayList(new int[] {0});
		t.insert(array);
		int[] resIndexes = t.getIndexes();
		assertEquals(0, resIndexes[0]);
	}

	@Test
	public void testInsertionSingleValue_02() {
		AInsertionSorter t = new MergeSort(1, 1, 1);
		IntArrayList[] array = new IntArrayList[1];
		array[0] = new IntArrayList(new int[] {1});
		t.insert(array);
		int[] resIndexes = t.getIndexes();
		assertEquals(1, resIndexes[0]);
	}

	@Test
	public void testInsertionTwoValues_01() {
		AInsertionSorter t = new MergeSort(2, 1, 10);
		IntArrayList[] array = new IntArrayList[1];
		array[0] = new IntArrayList(new int[] {1, 3});
		t.insert(array);
		int[] resIndexes = t.getIndexes();
		assertEquals(1, resIndexes[0]);
		assertEquals(3, resIndexes[1]);
		assertEquals(2, resIndexes.length);
	}

	@Test
	public void testInsertionMore_01() {
		AInsertionSorter t = new MergeSort(9, 1, 20);
		IntArrayList[] array = new IntArrayList[1];
		array[0] = new IntArrayList(new int[] {1, 3, 5, 6, 7, 9, 10, 11, 12});
		t.insert(array);
		int[] resIndexes = t.getIndexes();
		assertEquals(1, resIndexes[0]);
		assertEquals(3, resIndexes[1]);
		assertEquals(12, resIndexes[8]);
		assertEquals(9, resIndexes.length);
	}

	@Test
	public void testInsertionTwoUnique_01() {
		AInsertionSorter t = new MergeSort(9, 2, 20);
		IntArrayList[] array = new IntArrayList[2];
		array[0] = new IntArrayList(new int[] {1, 3, 5, 6, 7, 9});
		array[1] = new IntArrayList(new int[] {10, 11, 12});

		try {
			t.insert(array);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		int[] resIndexes = t.getIndexes();
		AMapToData data = t.getData();
		assertEquals(1, resIndexes[0]);
		assertEquals(0, data.getIndex(0));
		assertEquals(3, resIndexes[1]);
		assertEquals(12, resIndexes[8]);
		assertEquals(1, data.getIndex(8));
		assertEquals(9, resIndexes.length);
	}

	@Test
	public void testInsertionTwoUnique_02() {
		AInsertionSorter t = new MergeSort(5, 2, 20);
		IntArrayList[] array = new IntArrayList[2];
		array[0] = new IntArrayList(new int[] {1, 3, 5});
		array[1] = new IntArrayList(new int[] {2, 4,});

		try {
			t.insert(array);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		int[] resIndexes = t.getIndexes();
		AMapToData data = t.getData();
		assertArrayEquals(new int[] {1, 2, 3, 4, 5}, resIndexes);
		assertEquals(0, data.getIndex(0));
		assertEquals(1, data.getIndex(1));
	}

	@Test
	public void testInsertionTwoUnique_03() {
		AInsertionSorter t = new MergeSort(5, 2, 20);
		IntArrayList[] array = new IntArrayList[2];
		array[0] = new IntArrayList(new int[] {1, 3, 5});
		array[1] = new IntArrayList(new int[] {0, 4,});

		try {
			t.insert(array);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		int[] resIndexes = t.getIndexes();
		AMapToData data = t.getData();
		assertArrayEquals(new int[] {0, 1, 3, 4, 5}, resIndexes);
		assertEquals(1, data.getIndex(0));
		assertEquals(0, data.getIndex(1));
	}

	@Test
	public void testInsertionNegative_01() {
		AInsertionSorter t = new MergeSort(3, 1, 5);
		IntArrayList[] array = new IntArrayList[1];
		array[0] = new IntArrayList(new int[] {1, 3, 5});

		try {
			t.insert(array, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		int[] resIndexes = t.getIndexes();
		assertArrayEquals(new int[] {0, 2, 4}, resIndexes);
	}

	@Test
	public void testInsertionNegative_02() {
		AInsertionSorter t = new MergeSort(2, 1, 5);
		IntArrayList[] array = new IntArrayList[1];
		array[0] = new IntArrayList(new int[] {0, 1, 3, 5});

		try {
			t.insert(array, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		int[] resIndexes = t.getIndexes();
		assertArrayEquals(new int[] {2, 4}, resIndexes);
	}

	@Test
	public void testInsertionNegative_03() {
		AInsertionSorter t = new MergeSort(3, 2, 5);
		IntArrayList[] array = new IntArrayList[2];
		array[0] = new IntArrayList(new int[] {0, 1, 5});
		array[1] = new IntArrayList(new int[] {3});

		try {
			t.insert(array, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		int[] resIndexes = t.getIndexes();
		assertArrayEquals(new int[] {2, 3, 4}, resIndexes);
	}

	@Test
	public void testInsertionNegative_04() {
		AInsertionSorter t = new MergeSort(4, 2, 5);
		IntArrayList[] array = new IntArrayList[2];
		array[0] = new IntArrayList(new int[] {0, 5});
		array[1] = new IntArrayList(new int[] {1, 3});

		try {
			t.insert(array, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		int[] resIndexes = t.getIndexes();
		assertArrayEquals(new int[] {1, 2, 3, 4}, resIndexes);
	}

	@Test
	public void testInsertionNegative_05() {
		AInsertionSorter t = new MergeSort(4, 2, 10);
		IntArrayList[] array = new IntArrayList[2];
		array[0] = new IntArrayList(new int[] {0, 5, 6, 7, 8, 9, 10});
		array[1] = new IntArrayList(new int[] {1, 3});

		try {
			t.insert(array, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		int[] resIndexes = t.getIndexes();
		assertArrayEquals(new int[] {1, 2, 3, 4}, resIndexes);
	}

	@Test
	public void testInsertionNegative_06() {
		AInsertionSorter t = new MergeSort(4, 2, 10);
		IntArrayList[] array = new IntArrayList[2];
		array[0] = new IntArrayList(new int[] {0, 3, 5, 6, 7, 8, 9});
		array[1] = new IntArrayList(new int[] {1, 10});

		try {
			t.insert(array, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		int[] resIndexes = t.getIndexes();
		assertArrayEquals(new int[] {1, 2, 4, 10}, resIndexes);
	}

	@Test
	public void testInsertionNegative_07() {
		AInsertionSorter t = new MergeSort(4, 2, 10);
		IntArrayList[] array = new IntArrayList[2];
		array[0] = new IntArrayList(new int[] {1, 3, 5, 6, 7, 8, 9});
		array[1] = new IntArrayList(new int[] {0, 10});

		try {
			t.insert(array, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		int[] resIndexes = t.getIndexes();
		assertArrayEquals(new int[] {0, 2, 4, 10}, resIndexes);
	}

	@Test
	public void testInsertion_01() {
		AInsertionSorter t = new MergeSort(5, 3, 20);
		IntArrayList[] array = new IntArrayList[3];
		array[0] = new IntArrayList(new int[] {1, 3, 5});
		array[1] = new IntArrayList(new int[] {4,});
		array[2] = new IntArrayList(new int[] {0,});

		try {
			t.insert(array);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		int[] resIndexes = t.getIndexes();
		assertArrayEquals(new int[] {0, 1, 3, 4, 5}, resIndexes);
	}
}
