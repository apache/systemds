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

package org.apache.sysds.test.component.compress.mapping;

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
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;
import java.util.concurrent.ExecutorService;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.IMapToDataGroup;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToBit;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToCharPByte;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToZero;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class MappingTests {

	protected static final Log LOG = LogFactory.getLog(MappingTests.class.getName());

	protected static final int fictiveMax = MapToCharPByte.max + 3;

	public final int seed;
	public final MAP_TYPE type;
	public final int size;
	private final AMapToData m;
	private final int[] expected;

	final int max;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		for(MAP_TYPE t : MAP_TYPE.values()) {
			tests.add(new Object[] {1, t, 13, false});
			tests.add(new Object[] {3, t, 13, false});
			tests.add(new Object[] {3, t, 63, false});
			tests.add(new Object[] {6, t, 63, false});
			tests.add(new Object[] {4, t, 63, false});
			tests.add(new Object[] {3, t, 64, false});
			tests.add(new Object[] {4, t, 64, false});
			tests.add(new Object[] {3, t, 65, false});
			tests.add(new Object[] {5, t, 64 + 63, false});
			tests.add(new Object[] {5, t, 127, false});
			tests.add(new Object[] {5, t, 128, false});
			tests.add(new Object[] {4, t, 128, false});
			tests.add(new Object[] {3, t, 128, false});
			tests.add(new Object[] {5, t, 129, false});
			tests.add(new Object[] {7, t, 255, false});
			tests.add(new Object[] {8, t, 256, false});
			tests.add(new Object[] {8, t, 257, false});
			tests.add(new Object[] {5, t, 1234, false});
			tests.add(new Object[] {5, t, 13, true});
		}
		return tests;
	}

	public MappingTests(int seed, MAP_TYPE type, int size, boolean fill) {
		CompressedMatrixBlock.debug = true;
		this.seed = seed;
		this.type = type;
		this.size = size;
		this.max = MappingTestUtil.getUpperBoundValue(type);
		this.expected = new int[size];
		m = genMap(MapToFactory.create(size, (int) (Math.min(Integer.MAX_VALUE, (long) max + 1))), expected, max, fill,
			seed);
	}

	public static AMapToData genMap(AMapToData m, int[] expected, int max, boolean fill, int seed) {
		if(max <= 0)
			return m;
		Random vals = new Random(seed);
		int size = m.size();

		int randUpperBound = (int) (Math.min(Integer.MAX_VALUE, (long) max + 1));

		if(fill) {
			int v = vals.nextInt(randUpperBound);
			m.fill(v);
			Arrays.fill(expected, v);
		}

		for(int i = 0; i < size; i++) {
			int v = vals.nextInt(randUpperBound);
			if(fill) {
				if(v > max / 2)
					continue;
				else {
					m.set(i, v);
					expected[i] = v;
				}
			}
			else {
				m.set(i, v);
				expected[i] = v;
			}
		}

		// to make sure that the bit set is actually filled.
		for(int i = 0; i <= max && i < size; i++) {

			m.set(i, i);
			expected[i] = i;
		}
		return m;
	}

	@Test
	public void isEqual() {
		for(int i = 0; i < size; i++)
			if(expected[i] != m.getIndex(i))
				fail("Expected equals " + Arrays.toString(expected) + "\nbut got: " + m);
	}

	@Test
	public void testSerialization() {
		try {
			// Serialize out
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			m.write(fos);

			// Serialize in
			ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
			DataInputStream fis = new DataInputStream(bis);

			AMapToData n = MapToFactory.readIn(fis);
			compare(m, n);
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
	public void equalsTest() {
		AMapToData tmp = MapToFactory.create(m.size(), m.getUnique());
		if(m instanceof MapToZero)
			assertTrue(m.equals(tmp));
		else
			assertFalse(m.equals(tmp));
		tmp.copy(m);
		assertTrue(m.equals(tmp));
	}

	@Test
	public void countRuns() {
		int runs = m.countRuns();
		assertTrue(runs <= m.size());
	}

	@Test
	public void testOnDiskSizeInBytes() {
		try {
			// Serialize out
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			m.write(fos);
			byte[] arr = bos.toByteArray();
			int size = arr.length;
			if(size != m.getExactSizeOnDisk())
				fail(m.toString() + "\n The size is not the same on disk as promised: " + size + "  "
					+ m.getExactSizeOnDisk() + " " + type + " " + m.getType());
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
	public void resize() {
		switch(type) {
			// intensionally not containing breaks.
			case ZERO:
				compare(m.resize(-13), m);
				compare(m.resize(1), m);
			case BIT:
				compare(m.resize(5), m);
			case UBYTE:
				compare(m.resize(200), m);
			case BYTE:
				compare(m.resize(526), m);
			case CHAR:
				compare(m.resize(612451), m);
			case CHAR_BYTE:
				compare(m.resize(10000000), m);
			case INT:
				compare(m.resize(10000001), m);
		}
	}

	@Test
	public void resizeToSameSize() {
		// if we resize to same size return the same object!
		AMapToData m_same = m.resize(m.getUnique());
		assertEquals("Resize did not return the correct same objects", m_same, m);
	}

	protected static void compare(AMapToData a, AMapToData b) {
		final int size = Math.max(a.size(), b.size());
		for(int i = 0; i < size; i++)
			if(a.getIndex(i) != b.getIndex(i))
				fail("Not equal values:\n" + a + "\n" + b);
	}

	@Test
	public void replaceMax() {
		if(m instanceof MapToZero)
			return;
		m.replace(max, 0);

		for(int i = 0; i < size; i++) {
			expected[i] = expected[i] == max ? 0 : expected[i];
			if(expected[i] != m.getIndex(i))
				fail("Expected equals " + Arrays.toString(expected) + "\nbut got: " + m);
		}
	}

	@Test
	public void getCountsNoDefault() {
		try {

			int nVal = m.getUnique();
			if(nVal > 1000)
				return;

			int[] counts = m.getCounts(new int[nVal]);
			int sum = 0;
			for(int v : counts)
				sum += v;
			if(sum != size)
				fail("Incorrect count of values. : " + Arrays.toString(counts) + " " + sum
					+ "  sum is incorrect should be equal to number of rows: " + m.size());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed because of exception");
		}
	}

	@Test
	public void replaceMin() {
		if(m instanceof MapToZero)
			return;
		int max = m.getUpperBoundValue();
		m.replace(0, max);

		for(int i = 0; i < size; i++) {
			expected[i] = expected[i] == 0 ? max : expected[i];
			if(expected[i] != m.getIndex(i))
				fail("Expected the out put not to contain 0 but got: " + m);
		}
	}

	@Test
	public void getUnique() {
		int u = m.getUnique();
		if(m instanceof MapToZero)
			return;

		if((int) (Math.min(Integer.MAX_VALUE, (long) max + 1)) != u)
			fail("incorrect number of unique " + m + "expectedInstances" + max + " got" + u);
	}

	@Test
	public void testInMemorySize() {
		long inMemorySize = m.getInMemorySize();
		long estimatedSize = MapToFactory.estimateInMemorySize(size, (int) (Math.min(Integer.MAX_VALUE, (long) max + 1)));

		if(estimatedSize != inMemorySize)
			fail(" estimated size is not actual size: \nest: " + estimatedSize + " act: " + inMemorySize + "\n"
				+ m.getType() + "  " + type + " " + max + " " + m);
	}

	@Test
	public void testAppend() {
		int nVal = m.getUnique();
		if(nVal > 10000)
			return;
		int[] counts = m.getCounts(new int[nVal]);

		AMapToData m2 = m.append(m);
		assertEquals(m.size() * 2, m2.size());
		assertEquals(m.getUnique(), m2.getUnique());
		int[] counts2 = m2.getCounts(new int[nVal]);

		for(int i = 0; i < nVal; i++)
			assertEquals(counts[i] * 2, counts2[i]);
	}

	@Test
	public void testAppendN() {
		int nVal = m.getUnique();
		if(nVal > 10000)
			return;
		int[] counts = m.getCounts(new int[nVal]);

		try {

			AMapToData m2 = m.appendN(new IMapToDataGroup[] {//
				new Holder(m), new Holder(m), new Holder(m)});
			try {
				assertEquals(m.size() * 3, m2.size());
				assertEquals(m.getUnique(), m2.getUnique());
				int[] counts2 = m2.getCounts(new int[nVal]);

				for(int i = 0; i < nVal; i++)
					assertEquals(counts[i] * 3, counts2[i]);
			}
			catch(AssertionError e) {
				fail(e.getMessage() + "\nFailed appendN with in: \n" + m + "\ncomp:\n" + m2);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed " + e.getMessage());
		}

	}

	@Test
	public void testAppendVSAppendN() {
		final AMapToData m2 = m.append(m).append(m);
		final AMapToData m3 = m.appendN(new IMapToDataGroup[] {//
			new Holder(m), new Holder(m), new Holder(m)});
		compare(m2, m3);
	}

	@Test(expected = NotImplementedException.class)
	public void testAppendNotSame() {
		AMapToData mm;
		switch(type) {
			case INT:
				mm = MapToFactory.create(size, MAP_TYPE.CHAR_BYTE);
				mm.copy(m);
				m.append(mm);
				break;
			default:
				mm = MapToFactory.create(size, MAP_TYPE.INT);
				mm.copy(m);
				m.append(mm);
		}
		LOG.error("Did not throw exception with: " + m);
	}

	@Test
	public void splitReshapeParallel() throws Exception {
		if(m.size() % 2 == 0) {

			ExecutorService pool = CommonThreadPool.get();
			AMapToData[] ret = m.splitReshapeDDCPushDown(2, pool);

			for(int i = 0; i < m.size(); i++) {
				assertEquals(m.getIndex(i), ret[i % 2].getIndex(i / 2));
			}
		}
	}

	@Test
	public void splitReshape2() throws Exception {
		if(m.size() % 2 == 0) {

			AMapToData[] ret = m.splitReshapeDDC(2);

			for(int i = 0; i < m.size(); i++) {
				assertEquals(m.getIndex(i), ret[i % 2].getIndex(i / 2));
			}
		}
	}

	@Test
	public void splitReshape4() throws Exception {
		if(m.size() % 4 == 0) {

			AMapToData[] ret = m.splitReshapeDDC(4);

			for(int i = 0; i < m.size(); i++) {
				assertEquals(m.getIndex(i), ret[i % 4].getIndex(i / 4));
			}
		}
	}

	@Test
	public void getCounts() {
		if(m.getUnique() > 10000)
			return;
		int[] counts = m.getCounts();
		int countZeros = 0;
		for(int i = 0; i < m.size(); i++) {
			if(m.getIndex(i) == 0)
				countZeros++;
		}
		assertEquals(counts[0], countZeros);
	}

	private static class Holder implements IMapToDataGroup {

		AMapToData d;

		protected Holder(AMapToData d) {
			this.d = d;
		}

		@Override
		public AMapToData getMapToData() {
			return d;
		}

	}

	@Test
	public void slice() {
		if(m.size() > 2) {
			AMapToData s = m.slice(1, m.size() - 1);
			for(int i = 0; i < m.size() - 2; i++) {
				assertEquals(m.getIndex(i + 1), s.getIndex(i));
			}
		}
	}

	@Test
	public void setRange() {
		AMapToData tmp = MapToFactory.create(m.size(), m.getUnique());
		tmp.copy(m);

		tmp.set(0, m.size(), 0, new MapToZero(size));
		for(int i = 0; i < m.size(); i++)
			assertEquals(0, tmp.getIndex(i));

		if(m.size() > 11) {
			tmp.copy(m);

			tmp.set(10, m.size(), 0, new MapToZero(size));
			for(int i = 0; i < 10; i++)
				assertEquals(m.getIndex(i), tmp.getIndex(i));
			for(int i = 10; i < m.size(); i++)
				assertEquals(0, tmp.getIndex(i));

			if(m instanceof MapToZero)
				return;
			tmp.copy(m);
			AMapToData tmp2 = new MapToBit(2, size - 10);
			tmp2.fill(1);
			tmp2.set(0, 0);
			tmp.set(10, m.size(), 0, tmp2);
			for(int i = 0; i < 10; i++)
				assertEquals(m.getIndex(i), tmp.getIndex(i));
			assertEquals(0, tmp.getIndex(10));
			for(int i = 11; i < m.size(); i++)
				assertEquals(1, tmp.getIndex(i));

			for(MAP_TYPE t : MAP_TYPE.values()) {
				if(t == MAP_TYPE.ZERO)
					continue;
				tmp.copy(m);
				tmp2 = MapToFactory.resizeForce(tmp2, t);
				tmp.set(10, m.size(), 0, tmp2);
				for(int i = 0; i < 10; i++)
					assertEquals(m.getIndex(i), tmp.getIndex(i));
				assertEquals(0, tmp.getIndex(10));
				for(int i = 11; i < m.size(); i++)
					assertEquals(1, tmp.getIndex(i));
			}
		}
	}
}
