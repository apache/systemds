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

package org.apache.sysds.test.component.compress.offset;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory.OFF_TYPE;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import scala.util.Random;

@RunWith(value = Parameterized.class)
public class LargeOffsetTest {

	protected static final Log LOG = LogFactory.getLog(LargeOffsetTest.class.getName());

	public int[] data;
	public OFF_TYPE type;
	private AOffset o;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		// It is assumed that the input is in sorted order, all values are positive and there are no duplicates.
		for(OFF_TYPE t : OFF_TYPE.values()) {
			for(int i = 0; i < 4; i++) {
				// tests.add(new Object[]{gen(100, 10, i),t});
				// tests.add(new Object[]{gen(1000, 10, i),t});
				tests.add(new Object[] {gen(3030, 10, i), t});
				tests.add(new Object[] {gen(3030, 300, i), t});
				tests.add(new Object[] {gen(10000, 501, i), t});
				tests.add(new Object[] {gen(1000, 3000, i), t});
				tests.add(new Object[] {gen(1000, 10000, i), t});
			}
		}
		return tests;
	}

	public LargeOffsetTest(int[] data, OFF_TYPE type) {
		CompressedMatrixBlock.debug = true;
		this.data = data;
		this.type = type;
		this.o = OffsetTestUtil.getOffset(data, type);
		o.clearSkipList();
	}

	@Test
	public void testConstruction() {
		try {
			OffsetTests.compare(o, data);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void IteratorAtStart() {
		try {
			int idx = data.length / 3;
			AIterator it = o.getIterator(data[idx]);
			compare(it, data, idx);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void IteratorAtMiddle() {
		try {
			int idx = data.length / 2;
			AIterator it = o.getIterator(data[idx]);
			compare(it, data, idx);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void IteratorAtEnd() {
		try {
			int idx = data.length / 4 * 3;
			AIterator it = o.getIterator(data[idx]);
			compare(it, data, idx);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void slice() {
		try {
			if(data.length > 50) {

				AOffset sl = o.slice(data[4], data[data.length - 30]).offsetSlice;

				AIterator it = sl.getIterator();
				assertEquals(0, it.value());
				for(int i = 4; i < data.length - 30; i++) {
					if(it.getDataIndex() != i - 4) {
						fail("not equivalent index reached: expected: " + (i - 4) + " got: " + it.getDataIndex());
					}
					if(data[i] - data[4] != it.value()) {
						fail("not equivalent index expected: " + (data[i] - data[4]) + " but got " + it.value() + " at index "
							+ it.getDataIndex() + " correct index is: " + (i - 4));
					}
					if(i == data.length - 31) { // stop here.
						break;
					}
					it.next();
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void getSkipList() {
		o.constructSkipList();
		o.clearSkipList();
		o.constructSkipList();
	}

	@Test
	public void getIteratorClearSkipList() {
		o.constructSkipList();
		o.clearSkipList();
		AIterator it = o.getIterator(data[10]);
		assertEquals(data[10], it.value());
		AIterator it2 = o.getIterator(data[10]);
		assertTrue(it.equals(it2));

		AIterator it3 = o.getIterator(data[data.length-2]);
		assertEquals(data[data.length-2], it3.value());
	}

	private static void compare(AIterator it, int[] data, int off) {
		for(; off < data.length; off++) {
			assertEquals(data[off], it.value());
			if(off + 1 < data.length)
				it.next();
		}
	}

	private static int[] gen(int size, int maxSkip, int seed) {
		int[] of = new int[size];
		Random r = new Random(seed);
		of[0] = r.nextInt(maxSkip);
		for(int i = 1; i < size; i++) {
			of[i] = r.nextInt(maxSkip) + of[i - 1] + 1;
		}
		return of;
	}
}
