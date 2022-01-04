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

package org.apache.sysds.test.component.compress.insertionsort;

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.runtime.compress.colgroup.insertionsort.AInsertionSorter;
import org.apache.sysds.runtime.compress.colgroup.insertionsort.InsertionSorterFactory;
import org.apache.sysds.runtime.compress.colgroup.insertionsort.InsertionSorterFactory.SORT_TYPE;
import org.apache.sysds.runtime.compress.colgroup.insertionsort.MaterializeSort;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class TestInsertionSorters {

	public final int[][] data;
	public final SORT_TYPE st;
	public final int numRows;

	public final int[] expectedIndexes;
	public final int[] expectedData;

	private final IntArrayList[] offsets;
	private final int negativeIndex;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		for(SORT_TYPE t : SORT_TYPE.values()) {
			tests
				.add(new Object[] {10, new int[][] {new int[] {4, 7, 9}}, t, -1, new int[] {4, 7, 9}, new int[] {0, 0, 0}});
			tests.add(new Object[] {10, new int[][] {new int[] {4, 7, 9}, new int[] {1, 5, 8}}, t, -1,
				new int[] {1, 4, 5, 7, 8, 9}, new int[] {1, 0, 1, 0, 1, 0}});
			tests.add(new Object[] {10, new int[][] {new int[] {4, 7, 9}, new int[] {1, 5, 8}}, t, 0,
				new int[] {0, 1, 2, 3, 5, 6, 8}, new int[] {1, 0, 1, 1, 0, 1, 0}});
			tests.add(new Object[] {10, new int[][] {new int[] {1, 2, 3, 4, 5}}, t, 0, new int[] {0, 6, 7, 8, 9},
				new int[] {0, 0, 0, 0, 0}});
			tests.add(new Object[] {10, new int[][] {new int[] {1, 2, 3, 4, 5}, new int[] {0}}, t, 0,
				new int[] {0, 6, 7, 8, 9}, new int[] {0, 1, 1, 1, 1}});
			tests.add(new Object[] {10, new int[][] {new int[] {1, 2, 3, 4, 5}, new int[] {0}}, t, 1,
				new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9}, new int[] {0, 0, 0, 0, 0, 1, 1, 1, 1}});

			tests.add(new Object[] {2, new int[][] {new int[] {1}, new int[] {0}}, t, 1, new int[] {1}, new int[] {0}});
			tests.add(new Object[] {2, new int[][] {new int[] {1}, new int[] {0}}, t, 0, new int[] {0}, new int[] {0}});
			tests.add(new Object[] {3, new int[][] {new int[] {1}, new int[] {0}, new int[] {2}}, t, 0, new int[] {0, 2},
				new int[] {0, 1}});

			tests.add(new Object[] {4, new int[][] {new int[] {1}, new int[] {0}, new int[] {2}, new int[] {3}}, t, 0,
				new int[] {0, 2, 3}, new int[] {0, 1, 2}});

			tests.add(new Object[] {10, new int[][] {new int[] {2}, new int[] {6}, new int[] {0, 1, 3, 4, 7, 8, 9}}, t, 2,
				new int[] {2, 5, 6}, new int[] {0, 2, 1}});
			tests.add(new Object[] {10, new int[][] {new int[] {5}, new int[] {6}, new int[] {0, 1, 3, 4, 7, 8, 9}}, t, 2,
				new int[] {2, 5, 6}, new int[] {2, 0, 1}});
			tests.add(new Object[] {10, new int[][] {new int[] {5}, new int[] {2}, new int[] {0, 1, 3, 4, 7, 8, 9}}, t, 2,
				new int[] {2, 5, 6}, new int[] {1, 0, 2}});
			tests.add(new Object[] {10, new int[][] {new int[] {5}, new int[] {2}, new int[] {0, 1, 3, 4, 7, 8}}, t, 2,
				new int[] {2, 5, 6, 9}, new int[] {1, 0, 2, 2}});
			tests.add(new Object[] {10, new int[][] {new int[] {5}, new int[] {2}, new int[] {0, 1, 3, 4, 7}}, t, 2,
				new int[] {2, 5, 6, 8, 9}, new int[] {1, 0, 2, 2, 2}});
			tests.add(new Object[] {10, new int[][] {new int[] {5, 8}, new int[] {2}, new int[] {0, 1, 3, 4, 7}}, t, 2,
				new int[] {2, 5, 6, 8, 9}, new int[] {1, 0, 2, 0, 2}});

			tests.add(new Object[] {10, new int[][] {new int[] {0, 1, 3, 4, 7}, new int[] {5, 8}, new int[] {2}}, t, 0,
				new int[] {2, 5, 6, 8, 9}, new int[] {1, 0, 2, 0, 2}});
			tests.add(new Object[] {10, new int[][] {new int[] {0, 1, 3, 4, 7}, new int[] {5, 8}, new int[] {2}}, t, -1,
				new int[] {0, 1, 2, 3, 4, 5, 7, 8}, new int[] {0, 0, 2, 0, 0, 1, 0, 1}});

			tests.add(new Object[] {10, new int[][] {new int[] {0, 1, 3, 4, 7}, new int[] {5, 8}, new int[] {2, 9}}, t, 0,
				new int[] {2, 5, 6, 8, 9}, new int[] {1, 0, 2, 0, 1}});

			tests.add(gen(240, 10, t));
			tests.add(gen2(20, 10, t));
		}

		return tests;
	}

	private static Object[] gen(int size, int offsets, SORT_TYPE t) {
		final int[] expectedIndexes = new int[size];
		final int[] expectedData = new int[size];
		final int[][] ar = new int[offsets][];
		final int offsetsSize = size / offsets;
		for(int i = 0; i < offsets; i++)
			ar[i] = new int[offsetsSize];
		for(int i = 0; i < size; i++) {
			final int bucket = i / offsetsSize;
			final int index = i % offsetsSize;
			ar[bucket][index] = i;
			expectedIndexes[i] = i;
			expectedData[i] = bucket;
		}
		return new Object[] {size, ar, t, -1, expectedIndexes, expectedData};
	}

	private static Object[] gen2(int size, int offsets, SORT_TYPE t) {
		final int offsetsSize = size / offsets;
		final int[] expectedIndexes = new int[size - offsetsSize];
		final int[] expectedData = new int[size - offsetsSize];
		final int[][] ar = new int[offsets][];
		for(int i = 0; i < offsets; i++)
			ar[i] = new int[offsetsSize];
		for(int i = 0; i < size; i++) {
			final int bucket = i / offsetsSize;
			final int index = i % offsetsSize;
			ar[bucket][index] = i;
			if(i >= offsetsSize) {
				expectedIndexes[i - offsetsSize] = i;
				expectedData[i - offsetsSize] = bucket - 1;
			}
		}

		return new Object[] {size, ar, t, 0, expectedIndexes, expectedData};
	}

	public TestInsertionSorters(int numRows, int[][] data, SORT_TYPE st, int negativeIndex, int[] expectedIndexes,
		int[] expectedData) {
		this.data = data;
		this.st = st;
		this.expectedIndexes = expectedIndexes;
		this.expectedData = expectedData;
		this.numRows = numRows;
		this.negativeIndex = negativeIndex;

		offsets = new IntArrayList[data.length];
		for(int i = 0; i < data.length; i++)
			offsets[i] = new IntArrayList(data[i]);

	}

	@Test
	public void testInsertionSingle() {
		try {
			AInsertionSorter res = negativeIndex < 0 ? InsertionSorterFactory.create(numRows, offsets,
				st) : InsertionSorterFactory.createNegative(numRows, offsets, negativeIndex, st);
			if(!Arrays.equals(expectedIndexes, res.getIndexes()))
				fail(st.toString() + "\n\t" + Arrays.toString(expectedIndexes) + "\n\t" + Arrays.toString(res.getIndexes())
					+ "\n");
			compareData(res.getData());
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
	}

	private void compareData(AMapToData m) {
		for(int i = 0; i < expectedData.length; i++)
			if(expectedData[i] != m.getIndex(i))
				fail("compare data failed with technique: " + st.toString() + "\n\t" + Arrays.toString(expectedData)
					+ "\n\t" + m.toString() + "\n" + "differed at index " + i);
	}

	@BeforeClass
	public static void setCacheSize() {
		MaterializeSort.CACHE_BLOCK = 100;
	}

	@AfterClass
	public static void setCacheAfter() {
		MaterializeSort.CACHE_BLOCK = 1000;
	}
}
