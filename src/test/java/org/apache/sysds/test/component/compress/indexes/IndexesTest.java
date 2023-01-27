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

import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.IIterate;
import org.apache.sysds.runtime.compress.colgroup.indexes.SingleIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.TwoIndex;
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

	@Test(expected = ArrayIndexOutOfBoundsException.class)
	public void outOfBounds() {
		actual.get(expected.length);
	}

	@Test(expected = ArrayIndexOutOfBoundsException.class)
	public void negative() {
		actual.get(-1);
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
	public void estimateInMemorySizeIsNotToBig() {
		assertTrue(MemoryEstimates.intArrayCost(expected.length) > actual.estimateInMemorySize() - 16);
	}

	private void shift(int i) {
		compare(expected, actual.shift(i), i);
	}

	private static void compare(int[] expected, IColIndex actual) {
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
			assertEquals(expected[i], actual.next());
		}
		assertFalse(actual.hasNext());
	}
}
