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
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToCharPByte;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
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
			tests.add(new Object[] {3, t, 65, false});
			tests.add(new Object[] {5, t, 64 + 63, false});
			tests.add(new Object[] {5, t, 1234, false});
			tests.add(new Object[] {5, t, 13, true});
		}
		return tests;
	}

	public MappingTests(int seed, MAP_TYPE type, int size, boolean fill) {
		this.seed = seed;
		this.type = type;
		this.size = size;
		this.max = Math.min(MappingTestUtil.getUpperBoundValue(type), fictiveMax) + 1;
		expected = new int[size];
		m = genMap(MapToFactory.create(size, max), expected, max, fill, seed);
	}

	protected static AMapToData genMap(AMapToData m, int[] expected, int max, boolean fill, int seed) {
		if(max <= 1)
			return m;
		Random vals = new Random(seed);
		int size = m.size();
		if(fill) {
			int v = vals.nextInt(max);
			m.fill(v);
			Arrays.fill(expected, v);
		}

		for(int i = 0; i < size; i++) {
			int v = vals.nextInt(max);
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
		m.set(size - 1, max - 1);
		expected[size - 1] = max - 1;
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
		AMapToData m_same = MapToFactory.resize(m, m.getUnique());
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
		m.replace(max-1, 0);

		for(int i = 0; i < size; i++) {
			expected[i] = expected[i] == max - 1 ? 0 : expected[i];
			if(expected[i] != m.getIndex(i))
				fail("Expected equals " + Arrays.toString(expected) + "\nbut got: " + m);
		}
	}

	@Test
	public void getCountsNoDefault() {
		try {

			int nVal = m.getUnique();
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
		int max = m.getUpperBoundValue();
		m.replace(0, max);

		for(int i = 0; i < size; i++) {
			expected[i] = expected[i] == 0 ? max : expected[i];
			if(expected[i] != m.getIndex(i))
				fail("Expected equals " + Arrays.toString(expected) + "\nbut got: " + m);
		}
	}

	@Test
	public void getUnique() {
		int u = m.getUnique();
		if(max != u)
			fail("incorrect number of unique " + m + "\n expected" + max + " got" + u);
	}

	@Test
	public void testInMemorySize() {
		long inMemorySize = m.getInMemorySize();
		long estimatedSize = MapToFactory.estimateInMemorySize(size, max);

		if(estimatedSize != inMemorySize)
			fail(" estimated size is not actual size: \nest: " + estimatedSize + " act: " + inMemorySize + "\n"
				+ m.getType() + "  " + type + " " + max + " " + m);
	}

}
