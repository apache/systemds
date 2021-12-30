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
import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetByte;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetChar;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory.OFF_TYPE;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class OffsetTests {
	protected static final Log LOG = LogFactory.getLog(OffsetTests.class.getName());

	private static final long sizeTolerance = 100;

	public int[] data;
	public OFF_TYPE type;
	private AOffset o;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		// It is assumed that the input is in sorted order, all values are positive and there are no duplicates.
		for(OFF_TYPE t : OFF_TYPE.values()) {
			tests.add(new Object[] {new int[] {1, 2}, t});
			tests.add(new Object[] {new int[] {2, 142}, t});
			tests.add(new Object[] {new int[] {142, 421}, t});
			tests.add(new Object[] {new int[] {1, 1023}, t});
			tests.add(new Object[] {new int[] {1023, 1024}, t});
			tests.add(new Object[] {new int[] {1023}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 4, 5}, t});
			tests.add(new Object[] {new int[] {0}, t});
			tests.add(new Object[] {new int[] {Character.MAX_VALUE, ((int) Character.MAX_VALUE) + 1}, t});
			tests.add(new Object[] {new int[] {Character.MAX_VALUE, ((int) Character.MAX_VALUE) * 2}, t});
			tests.add(new Object[] {new int[] {0, 256}, t});
			tests.add(new Object[] {new int[] {0, 254}, t});
			tests.add(new Object[] {new int[] {0, Character.MAX_VALUE}, t});
			tests.add(new Object[] {new int[] {0, Character.MAX_VALUE, ((int) Character.MAX_VALUE) * 2}, t});
			tests.add(new Object[] {new int[] {2, Character.MAX_VALUE + 2}, t});
			tests.add(new Object[] {new int[] {0, ((int) Character.MAX_VALUE) + 1}, t});
			tests.add(new Object[] {new int[] {0, ((int) Character.MAX_VALUE) - 1}, t});
			tests.add(new Object[] {new int[] {0, 256 * 2}, t});
			tests.add(new Object[] {new int[] {0, 255 * 2}, t});
			tests.add(new Object[] {new int[] {0, 254 * 2}, t});
			tests.add(new Object[] {new int[] {0, 510, 765}, t});
			tests.add(new Object[] {new int[] {0, 254 * 3}, t});
			tests.add(new Object[] {new int[] {0, 255, 255 * 2, 255 * 3}, t});
			tests.add(new Object[] {new int[] {0, 255 * 2, 255 * 3}, t});
			tests.add(new Object[] {new int[] {0, 255 * 2, 255 * 3, 255 * 10}, t});
			tests.add(new Object[] {new int[] {0, 255 * 3}, t});
			tests.add(new Object[] {new int[] {0, 255 * 4}, t});
			tests.add(new Object[] {new int[] {0, 256 * 3}, t});
			tests.add(new Object[] {new int[] {255 * 3, 255 * 5}, t});
			tests.add(new Object[] {new int[] {1000000, 1000000 + 255 * 5}, t});
			tests.add(new Object[] {new int[] {100000000, 100000000 + 255 * 5}, t});
			tests.add(new Object[] {new int[] {100000000, 100001275, 100001530}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 255 * 4, 1500}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 4, 5}, t});
			tests.add(new Object[] {new int[] {2458248, 2458249, 2458253, 2458254, 2458256, 2458257, 2458258, 2458262,
				2458264, 2458266, 2458267, 2458271, 2458272, 2458275, 2458276, 2458281}, t});
		}
		return tests;
	}

	public OffsetTests(int[] data, OFF_TYPE type) {
		this.data = data;
		this.type = type;
		switch(type) {
			case BYTE:
				this.o = new OffsetByte(data);
				break;
			case CHAR:
				this.o = new OffsetChar(data);
				break;
			default:
				throw new NotImplementedException("not implemented");
		}
	}

	@Test
	public void testConstruction() {
		try {
			compare(o, data);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void testCacheExists() {
		if(data.length > 2) {
			AIterator i = o.getIterator();
			i.next();
			o.cacheIterator(i, data[1]);
			AIterator ii = o.getIterator(data[1]);
			assertTrue(ii.equals(i));
			ii.next();
			assertFalse(ii.equals(i));
		}
	}

	@Test
	public void testCacheDontExists() {
		if(data.length > 3) {
			AIterator i = o.getIterator();
			i.next();
			o.cacheIterator(i, data[1]);
			AIterator ii = o.getIterator(data[2]);
			assertFalse(ii.equals(i));
		}
	}

	@Test
	public void testSerialization() {
		try {
			// Serialize out
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			o.write(fos);

			// Serialize in
			ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
			DataInputStream fis = new DataInputStream(bis);

			AOffset n = OffsetFactory.readIn(fis);
			compare(n, data);
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
	public void testGetSize() {
		assertEquals(data.length, o.getSize());
	}

	@Test
	public void testOnDiskSizeInBytes() {
		try {
			// Serialize out
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			o.write(fos);

			int size = bos.toByteArray().length;
			assertEquals(size, o.getExactSizeOnDisk());
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
	public void testInMemoryEstimateIsSameAsActualOrLarger() {
		try {
			final long inMemorySize = o.getInMemorySize();
			long estimatedSize;
			switch(type) {
				case BYTE:
					estimatedSize = OffsetByte.estimateInMemorySize(data.length, data[data.length - 1] - data[0]);
					break;
				case CHAR:
					estimatedSize = OffsetChar.estimateInMemorySize(data.length, data[data.length - 1] - data[0]);
					break;
				default:
					throw new DMLCompressionException("Unknown input");
			}
			if(!(inMemorySize <= estimatedSize + sizeTolerance)) {

				fail("in memory size: " + inMemorySize + " is not smaller than estimate: " + estimatedSize
					+ " with tolerance " + sizeTolerance);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void testSkipToContainedIndex() {
		try {
			assertEquals(data[data.length - 1], o.getIterator().skipTo(data[data.length - 1]));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed skipping to last index");
		}
	}

	@Test
	public void testSkipToContainedIndexPlusOne() {
		try {
			assertNotEquals(data[data.length - 1] + 1, o.getIterator().skipTo(data[data.length - 1]));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed skipping to last index");
		}
	}

	@Test
	public void testSkipToContainedIndexPlusN() {
		try {
			if(data.length > 1)
				assertTrue(data[1] <= o.getIterator().skipTo(data[1] + 1));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed skipping to last index");
		}
	}

	@Test
	public void testSkipToContainedIndexMinusOne() {
		try {
			int v = data[data.length - 1];
			int maxDiff = 1;
			assertTrue(v <= o.getIterator().skipTo(v - 1) + maxDiff);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed skipping to last index");
		}
	}

	@Test
	public void testSkipToContainedIndexMinusN() {
		try {
			int v = data[data.length - 1];
			int maxDiff = 142;
			assertTrue(v <= o.getIterator().skipTo(v - 1) + maxDiff);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed skipping to last index");
		}
	}

	@Test
	public void testToString() {
		String os = o.toString();
		os = os.substring(os.indexOf("["), os.length());
		String vs = Arrays.toString(data);
		if(!os.equals(vs)) {
			fail("The two array string are not equivalent with " + type + "\n" + os + " : " + vs);
		}
	}

	@Test
	public void testIsNotOverFirstDataPoint() {
		assertFalse(o.getIterator().isNotOver(data[0]));
	}

	@Test
	public void testIsNotOverSecondDataPointOnInit() {
		if(data.length > 1)
			assertTrue(o.getIterator().isNotOver(data[1]));
	}

	@Test
	public void testIsNotOverSecondDataPointOnInitToSecond() {
		if(data.length > 2)
			assertFalse(o.getIterator(data[1]).isNotOver(data[1]));
	}

	@Test
	public void testIsOverFirstDataPointOnInitToSecond() {
		if(data.length > 2)
			assertFalse(o.getIterator(data[1]).isNotOver(data[0]));
	}

	@Test
	public void testAskForLastElement() {
		if(data.length == 2)
			assertTrue(o.getIterator(data[1]).getDataIndex() == 1);
	}

	@Test
	public void testAskForLastElementP1IsNull() {
		if(data.length == 2)
			assertTrue(o.getIterator(data[1] + 1) == null);
	}

	@Test
	public void testGetDataIndexOnInit() {
		assertTrue(o.getIterator().getDataIndex() == 0);
	}

	@Test
	public void testGetDataIndexOnInitSkipToFirst() {
		if(data.length > 2)
			assertTrue(o.getIterator(data[1]).getDataIndex() == 1);
	}

	@Test
	public void testGetDataIndexOnInitSkipToN() {
		if(data.length > 3)
			assertTrue(o.getIterator(data[2]).getDataIndex() == 2);
	}

	@Test
	public void testGetDataAfterNext() {
		if(data.length > 1)
			testGetDataAfterNextN(o.getIterator());
	}

	@Test
	public void testGetDataAfterNext2() {
		if(data.length > 2)
			testGetDataAfterNextN(o.getIterator(2));
	}

	public void testGetDataAfterNextN(AIterator it) {
		int d = it.getDataIndex();
		it.next();
		assertEquals(d + 1, it.getDataIndex());
	}

	@Test
	public void testGetDataAfterNextComb() {
		if(data.length > 1)
			testGetDataAfterNextCombN(o.getIterator());
	}

	@Test
	public void testGetDataAfterNextComb2() {
		if(data.length > 2)
			testGetDataAfterNextCombN(o.getIterator(2));
	}

	public void testGetDataAfterNextCombN(AIterator it) {
		int d = it.getDataIndex();
		it.next();
		assertEquals(d + 1, it.getDataIndex());
	}

	@Test
	public void testGetUnreasonablyHighSkip() {
		assertTrue(o.getIterator(Integer.MAX_VALUE - 1000) == null);
	}

	@Test
	public void testCacheNullIterator() {
		o.cacheIterator(null, 21415);
	}

	protected static void compare(AOffset o, int[] v) {
		AIterator i = o.getIterator();
		if(v[0] != i.value())
			fail("incorrect result using : " + o.getClass().getSimpleName() + " expected: " + Arrays.toString(v)
				+ " but was :" + o.toString());
		for(int j = 1; j < v.length; j++) {
			i.next();
			if(v[j] != i.value())
				fail("incorrect result using : " + o.getClass().getSimpleName() + " expected: " + Arrays.toString(v)
					+ " but was :" + o.toString());
		}
		if(i.getOffsetsIndex() != o.getOffsetsLength())
			fail("The allocated offsets are longer than needed: idx " + i.getOffsetsIndex() + " vs len "
				+ o.getOffsetsLength() + "\n" + Arrays.toString(v));
	}
}
