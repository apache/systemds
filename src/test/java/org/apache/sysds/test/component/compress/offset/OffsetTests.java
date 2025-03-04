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
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AOffsetsGroup;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset.OffsetSliceInfo;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffsetIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetByte;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetChar;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetEmpty;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory.OFF_TYPE;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetSingle;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetTwo;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class OffsetTests {
	static {
		CompressedMatrixBlock.debug = true;
	}

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
			tests.add(new Object[] {new int[] {}, t});
			tests.add(new Object[] {new int[] {1, 2}, t});
			tests.add(new Object[] {new int[] {1, 3, 5, 7}, t});
			tests.add(new Object[] {new int[] {2, 3, 6, 7}, t});
			tests.add(new Object[] {new int[] {2, 5, 10, 12}, t});
			tests.add(new Object[] {new int[] {2, 6, 8, 14}, t});
			tests.add(new Object[] {new int[] {2, 142}, t});
			tests.add(new Object[] {new int[] {142, 421}, t});
			tests.add(new Object[] {new int[] {1, 1023}, t});
			tests.add(new Object[] {new int[] {1023, 1024}, t});
			tests.add(new Object[] {new int[] {1023}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 4, 5}, t});
			tests.add(new Object[] {new int[] {0}, t});
			tests.add(new Object[] {new int[] {500}, t});
			tests.add(new Object[] {new int[] {1442}, t});
			tests.add(new Object[] {new int[] {Character.MAX_VALUE, Character.MAX_VALUE + 1}, t});
			tests.add(new Object[] {new int[] {0, 256}, t});
			tests.add(new Object[] {new int[] {0, 254}, t});
			tests.add(new Object[] {new int[] {0, Character.MAX_VALUE}, t});
			tests.add(new Object[] {new int[] {2, Character.MAX_VALUE + 2}, t});
			tests.add(new Object[] {new int[] {0, Character.MAX_VALUE + 1}, t});
			tests.add(new Object[] {new int[] {0, Character.MAX_VALUE - 1}, t});
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
			tests.add(new Object[] {new int[] {0, 100, 200, 300, 400, 500, 600}, t});
			tests.add(new Object[] {new int[] {0, 200, 400, 600, 800, 1000, 1200}, t});
			tests.add(new Object[] {new int[] {255 * 3, 255 * 5}, t});
			tests.add(new Object[] {new int[] {1000000, 1000000 + 255 * 5}, t});
			tests.add(new Object[] {new int[] {100000000, 100000000 + 255 * 5}, t});
			tests.add(new Object[] {new int[] {100000000, 100001275, 100001530}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 255 * 4, 1500}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 4, 5}, t});
			tests.add(new Object[] {new int[] {2458248, 2458249, 2458253, 2458254, 2458256, 2458257, 2458258, 2458262,
				2458264, 2458266, 2458267, 2458271, 2458272, 2458275, 2458276, 2458281}, t});

			tests.add(new Object[] {gen(100, 100, 1), t});
			for(int i = 0; i < 10; i++) {
				tests.add(new Object[] {gen(100, 200, i), t});
				tests.add(new Object[] {gen(100, 250, i + 10230), t});
				tests.add(new Object[] {gen(100, 4, i + 120), t});
				tests.add(new Object[] {gen(100, 350, i + 1030), t});
				tests.add(new Object[] {gen(30, 1000, i + 101420), t});
				tests.add(new Object[] {gen(30, 3000, i + 101420), t});

			}
		}
		tests.add(new Object[] {new int[] {Character.MAX_VALUE, Character.MAX_VALUE * 2}, OFF_TYPE.CHAR});
		tests.add(new Object[] {new int[] {0, Character.MAX_VALUE, Character.MAX_VALUE * 2}, OFF_TYPE.CHAR});
		tests.add(new Object[] {
			new int[] {1, Character.MAX_VALUE * 2 + 3, Character.MAX_VALUE * 4 + 4, Character.MAX_VALUE * 16 + 4},
			OFF_TYPE.CHAR});
		return tests;
	}

	/**
	 * Generate a valid offset
	 * 
	 * @param i    Number of offsets
	 * @param j    distance to sample from between each
	 * @param seed The seed
	 * @return a valid offset
	 */
	public static int[] gen(int i, int j, int seed) {
		int[] a = new int[i];
		Random r = new Random(seed);
		int o = r.nextInt(j);
		a[0] = o;
		for(int k = 1; k < i; k++) {
			o += r.nextInt(j) + 1;
			a[k] = o;
		}
		return a;
	}

	public OffsetTests(int[] data, OFF_TYPE type) {
		CompressedMatrixBlock.debug = true;
		this.data = data;
		this.type = type;
		this.o = OffsetTestUtil.getOffset(data, type);
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
	public void testConstructionOffsetIteratorOnly() {
		try {
			compareOffsetIterator(o, data);
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
	public void offsetSet() {
		AIterator a = o.getIterator();
		if(a != null) {
			a.setOff(324);
			assertTrue(a.value() == 324);
		}
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

			if(data.length == 0)
				estimatedSize = OffsetEmpty.estimateInMemorySize();
			else if(data.length == 1)
				estimatedSize = OffsetSingle.estimateInMemorySize();
			else if(data.length == 2)
				estimatedSize = OffsetTwo.estimateInMemorySize();
			else {

				switch(type) {
					case BYTE:
					case UBYTE:
						final int correctionByte = OffsetFactory.correctionByte(data[data.length - 1] - data[0], data.length);
						estimatedSize = OffsetByte.estimateInMemorySize(data.length + correctionByte);
						break;
					case CHAR:
						final int correctionChar = OffsetFactory.correctionChar(data[data.length - 1] - data[0], data.length);
						estimatedSize = OffsetChar.estimateInMemorySize(data.length + correctionChar);
						break;
					default:
						throw new DMLCompressionException("Unknown input");
				}
			}

			if(!(inMemorySize <= estimatedSize + sizeTolerance)) {
				fail("in memory size: " + inMemorySize + " is not smaller than estimate: " + estimatedSize
					+ " with tolerance " + sizeTolerance + "\nEncoded:" + o + "\nData:" + Arrays.toString(data));
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
			if(data.length > 0)
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
			if(data.length > 0)
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
			if(data.length > 0) {
				int v = data[data.length - 1];
				int maxDiff = 1;
				assertTrue(v <= o.getIterator().skipTo(v - 1) + maxDiff);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed skipping to last index");
		}
	}

	@Test
	public void testSkipToContainedIndexMinusN() {
		try {
			if(data.length > 0) {
				int v = data[data.length - 1];
				int maxDiff = 142;
				assertTrue(v <= o.getIterator().skipTo(v - 1) + maxDiff);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed skipping to last index");
		}
	}

	@Test
	public void testToString() {
		try {
			String os = o.toString();
			if(data.length > 0) {
				os = os.substring(os.indexOf("["), os.length());
				String vs = Arrays.toString(data);
				if(!os.equals(vs)) {
					fail("The two array string are not equivalent with " + type + "\n" + os + " : " + vs);
				}
			}
			else {
				assertTrue(os.contains("Empty"));
			}
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}

	@Test
	public void testIsNotOverFirstDataPoint() {

		if(data.length > 0)
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

		if(data.length > 0)
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

	@Test
	public void testCloneIterator() {
		if(o.getIterator() != null)
			assertTrue(o.getIterator().clone().equals(o.getIterator()));
	}

	@Test
	public void testCloneIteratorNext() {
		if(data.length > 1) {

			AIterator a = o.getIterator().clone();
			AIterator b = o.getIterator();
			a.next();
			b.next();
			b = b.clone();
			assertTrue(a.equals(b));
		}
	}

	@Test
	public void testCloneIteratorOffsetNext() {
		if(data.length > 1) {
			AOffsetIterator a = o.getOffsetIterator();
			AOffsetIterator b = o.getOffsetIterator();
			a.next();
			b.next();
			assertTrue(a.value() == b.value());
		}
	}

	@Test
	public void testIteratorToString() {
		AOffsetIterator a = o.getOffsetIterator();
		if(a != null)
			a.toString();

		AIterator b = o.getIterator();
		if(b != null)
			b.toString();
	}

	@Test
	public void testIteratorSkipToOverEnd() {
		AIterator a = o.getIterator();
		if(a != null) {
			a.skipTo(Integer.MAX_VALUE);
		}
	}

	@Test
	public void testNextOn1Element() {
		if(data.length == 1) {
			AIterator a = o.getIterator();
			a.next();
			a.next();
			a.next();

			AOffsetIterator b = o.getOffsetIterator();
			b.next();
			b.next();
			// should be possible... but not really relevant.
		}
	}

	@Test
	public void sliceFirst100() {
		slice(0, 100);
	}

	@Test
	public void slice1_2() {
		slice(1, 2);
	}

	@Test
	public void slice3_4() {
		slice(3, 4);
	}

	@Test
	public void sliceRange() {
		for(int i = 0; i < 100; i++) {
			// not allowed to slice smaller or equal range.
			// slice(i, i - 1);
			// slice(i, i);
			slice(i, i + 1);
			slice(i, i + 2);
			slice(i, i + 3);
		}
	}

	@Test
	public void sliceRange256() {
		slice(250, 260);
		slice(254, 260);
		slice(254, 257);
		slice(255, 257);
		slice(255, 256);
	}

	@Test
	public void sliceRange2x256() {
		slice(250 * 2, 260 * 2);
		slice(254 * 2, 260 * 2);
		slice(254 * 2, 257 * 2);
		slice(255 * 2, 257 * 2);
		slice(255 * 2, 256 * 2);
	}

	@Test
	public void sliceToLast() {
		if(data.length > 1)
			slice(0, data[data.length - 1]);
	}

	@Test
	public void sliceToLastMissingFirst100() {
		if(data.length > 1)
			slice(100, data[data.length - 1]);
	}

	@Test
	public void slice100to10000() {
		slice(100, 10000);
	}

	@Test
	public void slice1to4() {
		slice(1, 4);
	}

	@Test
	public void slice() {
		if(data.length > 1) {
			int n = data[data.length - 1];
			for(int i = 0; i < n && i < 100; i++) {
				for(int j = i+1; j < n + 1 && j < 100; j++) {
					slice(i, j, false);
				}
			}
		}
	}

	@Test
	public void sliceAllSpecific() {
		if(data.length > 1)
			slice(data[0], data[data.length - 1] + 1);
	}

	private void slice(int l, int u) {
		slice(l, u, false);
	}

	private void slice(int l, int u, boolean str) {
		try {

			OffsetSliceInfo a = o.slice(l, u);
			if(str)
				a.offsetSlice.toString();
			if(data.length > 0 && data[data.length - 1] > u) {

				AIterator it = a.offsetSlice.getIterator();
				a.offsetSlice.verify(a.uIndex - a.lIndex);
				int i = 0;
				while(i < data.length && data[i] < l)
					i++;
					
				if(! (a.offsetSlice instanceof OffsetEmpty)){
					int t = 0;
					final int lasstSliceOffset = a.offsetSlice.getOffsetToLast();
					while(data[i] < u) {
						assertEquals(data[i] - l, it.value());
						if(lasstSliceOffset > it.value())
							it.next();
						i++;
						t++;
					}

					int sliceSize = a.offsetSlice.getSize();
					if(sliceSize != t) {
						fail("Slice size is not equal to elements that should have been sliced:\n" + sliceSize + " vs " + t
							+ "\n" + a + "\nrange: " + l + " " + u + "\ninput " + o);
					}
				}
				else {
					if( i < data.length){
						assertTrue(data[i] >= u);
					}
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to slice range: " + l + " -> " + u + " in:\n" + o);
		}

	}


	@Test
	public void verify() {
		o.verify(o.getSize());
	}



	@Test
	public void append() {
		if(data.length > 0) {
			final int ll = data[data.length - 1] + 1;
			final AOffset r = o.append(o, ll);
			compareAppendOne(r, ll);
		}
	}

	@Test
	public void append_v2() {
		if(data.length > 0) {
			final int ll = data[data.length - 1] + 3;
			final AOffset r = o.append(o, ll);
			compareAppendOne(r, ll);
		}
	}

	@Test
	public void appendN_one() {
		if(data.length > 0) {
			final int ll = data[data.length - 1] + 1;
			final AOffset r = o.appendN(new AOffsetsGroup[] {new Con(o), new Con(o)}, ll);
			compareAppendOne(r, ll);
		}
	}

	@Test
	public void appendN_one_v2() {
		if(data.length > 0) {
			final int ll = data[data.length - 1] + 3;
			final AOffset r = o.appendN(new AOffsetsGroup[] {new Con(o), new Con(o)}, ll);
			compareAppendOne(r, ll);
		}
	}

	@Test
	public void compareAppend() {
		if(data.length > 0) {
			final int ll = data[data.length - 1] + 3;
			final AOffset r1 = o.appendN(new AOffsetsGroup[] {new Con(o), new Con(o)}, ll);
			final AOffset r2 = o.append(o, ll);
			compare(r1, r2);
		}
	}

	@Test
	public void compareAppend_v2() {
		if(data.length > 0) {
			final int ll = data[data.length - 1] + 100;
			final AOffset r1 = o.appendN(new AOffsetsGroup[] {new Con(o), new Con(o)}, ll);
			final AOffset r2 = o.append(o, ll);
			compare(r1, r2);
		}
	}

	@Test
	public void compareAppend_v3() {
		if(data.length > 0) {
			final int ll = data[data.length - 1] + 1000;
			final AOffset r1 = o.appendN(new AOffsetsGroup[] {new Con(o), new Con(o)}, ll);
			final AOffset r2 = o.append(o, ll);
			compare(r1, r2);
		}
	}

	@Test
	public void compareAppend_ot() {
		if(data.length > 0) {
			final int ll = data[data.length - 1] + 1000;
			final AOffset r1 = o.appendN(new AOffsetsGroup[] {new Con(o), new Con(OffsetFactory.createOffset(data))}, ll);
			final AOffset r2 = o.append(o, ll);
			compare(r1, r2);
		}
	}

	@Test
	public void compareAppend_2x() {

		try {

			if(data.length > 0) {

				final int ll = data[data.length - 1] + 100;
				final AOffset r = o.appendN(new AOffsetsGroup[] {new Con(o), new Con(o), new Con(o)}, ll);
				final AOffset t2 = o.append(o, ll).append(o, ll * 2);
				compare(r, t2);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void compareAppend_2x_v2() {

		try {

			if(data.length > 0) {

				final int ll = data[data.length - 1] + 280;
				final AOffset r = o.appendN(new AOffsetsGroup[] {new Con(o), new Con(o), new Con(o)}, ll);
				final AOffset t2 = o.append(o, ll).append(o, ll * 2);
				compare(r, t2);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private void compareAppendOne(AOffset r, int ll) {
		if(r.getSize() != data.length * 2)
			fail("Invalid return length:   is: " + o.getSize() + " Should be " + (data.length * 2));
		final AIterator i = r.getIterator();
		if(o.getSize() > 0) {
			if(i.value() != data[0])
				fail("incorrect value");
			for(int j = 1; j < data.length; j++) {
				i.next();
				if(data[j] != i.value())
					fail("Incorrect value: expected: " + (data[j]) + " Was: " + i.value());

			}
			for(int j = data.length; j < data.length * 2; j++) {
				i.next();
				if(data[j - data.length] + ll != i.value())
					fail("Incorrect value: expected: " + (data[j - data.length] + ll) + " Was: " + i.value());

			}
			assertEquals(data[data.length - 1] + ll, i.value());
			assertEquals(data[data.length - 1] + ll, r.getOffsetToLast());
		}
	}

	@Test
	public void moveIndex1() {

		AOffset b = o.moveIndex(10);
		compareMoved(b, data, -10);
	}

	@Test
	public void getLength() {
		assertTrue(o.getLength() + 1 >= data.length);
	}

	@Test(expected = Exception.class)
	public void invalidReverse() {
		int last = o.getOffsetToLast();
		o.reverse(last - 1);
	}

	@Test
	public void constructSkipList() {
		try {

			o.constructSkipList();
			o.constructSkipList();
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private void compareMoved(AOffset o, int[] v, int m) {
		AIterator i = o.getIterator();
		if(o.getSize() != v.length) {
			fail("Incorrect result sizes : " + o + " " + Arrays.toString(v));
		}
		if(o.getSize() > 0) {
			assertEquals(o.getOffsetToLast(), v[v.length - 1] + m);
			if(v[0] + m != i.value())
				fail("incorrect result using : " + o.getClass().getSimpleName() + " expected: " + Arrays.toString(v)
					+ " but was :" + o.toString());
			for(int j = 1; j < v.length; j++) {
				i.next();
				if(v[j] + m != i.value())
					fail("incorrect result using : " + o.getClass().getSimpleName() + " expected: " + Arrays.toString(v)
						+ " but was :" + o.toString());
			}
		}
	}

	private class Con implements AOffsetsGroup {

		AOffset a;

		protected Con(AOffset a) {
			this.a = a;
		}

		@Override
		public AOffset getOffsets() {
			return a;
		}

	}

	public static void compare(AOffset o, int[] v) {
		AIterator i = o.getIterator();
		if(o.getSize() != v.length) {
			fail("Incorrect result sizes : " + o + " " + Arrays.toString(v));
		}
		if(o.getSize() > 0) {
			assertEquals(o.getOffsetToLast(), v[v.length - 1]);
			if(v[0] != i.value())
				fail("incorrect result using : " + o.getClass().getSimpleName() + " expected: " + Arrays.toString(v)
					+ " but was :" + o.toString());
			for(int j = 1; j < v.length; j++) {
				i.next();
				if(v[j] != i.value())
					fail("incorrect result using : " + o.getClass().getSimpleName() + " expected: " + Arrays.toString(v)
						+ " but was :" + o.toString());
			}
		}
	}

	public static void compare(AOffset o, AOffset b) {
		assertEquals(o.getOffsetToFirst(), b.getOffsetToFirst());
		assertEquals(o.getOffsetToLast(), b.getOffsetToLast());

		AIterator io = o.getIterator();
		AIterator ib = b.getIterator();

		assertEquals(io.value(), ib.value());
		int s = o.getSize();
		assertEquals(s, b.getSize());
		for(int i = 0; i < s - 1; i++) {
			assertEquals(io.value(), ib.value());
			io.next();
			ib.next();
		}
		assertEquals(io.value(), ib.value());
		assertEquals(io.value(), b.getOffsetToLast());

	}

	public static void compareOffsetIterator(AOffset o, int[] v) {
		if(o.getSize() != v.length) {
			fail("Incorrect result sizes : " + o + " " + Arrays.toString(v));
		}
		if(o.getSize() > 0) {
			AOffsetIterator i = o.getOffsetIterator();
			if(v[0] != i.value())
				fail("incorrect result using : " + o.getClass().getSimpleName() + " expected: " + Arrays.toString(v)
					+ " but was :" + o.toString());
			for(int j = 1; j < v.length; j++) {
				i.next();
				if(v[j] != i.value())
					fail("incorrect result using : " + o.getClass().getSimpleName() + " expected: " + Arrays.toString(v)
						+ " but was :" + o.toString());
			}
		}
	}
}
