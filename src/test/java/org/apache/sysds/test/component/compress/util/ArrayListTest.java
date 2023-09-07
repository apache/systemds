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
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.junit.Test;

public class ArrayListTest {

	@Test
	public void allocate() {
		assertEquals(4, new IntArrayList(4).extractValues().length);
	}

	@Test
	public void allocate2() {
		assertEquals(16, new IntArrayList(16).extractValues().length);
	}

	@Test
	public void sizeEmpty() {
		assertEquals(0, new IntArrayList(16).size());
	}

	@Test
	public void sizeEmpty2() {
		assertEquals(0, new IntArrayList(32).size());
	}

	@Test
	public void sizeEmpty3() {
		assertEquals(0, new IntArrayList().size());
	}

	@Test(expected = DMLCompressionException.class)
	public void directError() {
		new IntArrayList(null);
	}

	@Test
	public void directAllocation() {
		IntArrayList a = new IntArrayList(new int[] {1, 2, 3});

		assertEquals(3, a.size());
		assertEquals(1, a.get(0));
		assertEquals(2, a.get(1));
		assertEquals(3, a.get(2));
	}

	@Test
	public void appendValue() {
		IntArrayList a = new IntArrayList();

		for(int i = 0; i < 10; i++) {
			a.appendValue(i);
		}

		assertEquals(10, a.size());
		assertEquals(6, a.get(6));
	}

	@Test
	public void appendValue2() {
		IntArrayList a = new IntArrayList(new int[] {1, 2, 3});

		for(int i = 0; i < 10; i++) {
			a.appendValue(i);
		}

		assertEquals(13, a.size());
		assertEquals(6, a.get(6 + 3));
	}

	@Test
	public void appendValueArray() {
		IntArrayList a = new IntArrayList(new int[] {1, 2, 3});
		IntArrayList b = new IntArrayList(new int[] {4, 5, 6});
		a.appendValue(b);

		assertEquals(6, a.size());
		assertEquals(6, a.get(5));
	}

	@Test
	public void appendValueArray2() {
		IntArrayList a = new IntArrayList(new int[] {1, 2, 3});
		IntArrayList b = new IntArrayList(new int[] {4, 5, 6});
		a.appendValue(b);
		a.appendValue(b);
		a.appendValue(b);

		assertEquals(12, a.size());
		assertEquals(6, a.get(5));
	}

	@Test
	public void appendValueArray3() {
		IntArrayList a = new IntArrayList(new int[] {1, 2, 3});
		IntArrayList b = new IntArrayList(new int[] {4, 5, 6});
		a.appendValue(b);
		a.appendValue(b);
		a.appendValue(b);
		int[] ex = a.extractValues();

		assertTrue(ex.length >= a.size());
		assertEquals(6, a.get(5));
		assertEquals(6, ex[5]);
	}


	@Test
	public void appendValueArray4() {
		IntArrayList a = new IntArrayList(new int[] {1, 2, 3});
		IntArrayList b = new IntArrayList(new int[] {4, 5, 6});
		for(int i = 0; i < 10; i++){

			a.appendValue(b);
			a.appendValue(1);
		}
		int[] ex = a.extractValues();

		assertTrue(ex.length >= a.size());
		assertEquals(10*3+3 + 10,  a.size());
		assertEquals(6, a.get(5));
		assertEquals(6, ex[5]);
	}

	@Test
	public void extract() {
		IntArrayList a = new IntArrayList();
		for(int i = 0; i < 2; i++)
			a.appendValue(i);

		int[] ex = a.extractValues();
		assertTrue(ex.length > a.size());
		int[] et = a.extractValues(true);
		assertTrue(et.length == a.size());
		assertEquals(1, a.get(1));
		assertEquals(1, ex[1]);
		assertEquals(1, et[1]);
	}

	@Test
	public void toStringTest() {
		IntArrayList a = new IntArrayList();
		for(int i = 0; i < 2; i++)
			a.appendValue(i);
		String as = a.toString();
		// int[] ex = a.extractValues();
		// assertTrue(ex.length > a.size());
		int[] et = a.extractValues(true);
		String es = Arrays.toString(et);
		assertEquals(es, as);
	}

	@Test
	public void toStringEmpty() {
		IntArrayList a = new IntArrayList();
		// for(int i = 0; i < 2; i++)
		// a.appendValue(i);
		String as = a.toString();
		// int[] ex = a.extractValues();
		// assertTrue(ex.length > a.size());
		int[] et = a.extractValues(true);
		String es = Arrays.toString(et);
		assertEquals(es, as);
	}

	@Test
	public void extractExactEach() {
		IntArrayList a = new IntArrayList();
		for(int i = 0; i < 10; i++) {
			assertEquals(i, a.extractValues(true).length);
			assertTrue(i <= a.extractValues(false).length);
			a.appendValue(i);
		}
		assertTrue(10 <= a.extractValues(false).length);
		assertEquals(10, a.extractValues(true).length);
	}

	@Test
	public void reset1(){
		IntArrayList a = new IntArrayList();
		a.appendValue(1);
		a.appendValue(2);
		assertEquals(2, a.size());
		a.reset();
		assertEquals(0, a.size());
		
	}
}
