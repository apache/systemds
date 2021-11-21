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
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class MappingTests {

	protected static final Log LOG = LogFactory.getLog(MappingTests.class.getName());

	public final int seed;
	public final MAP_TYPE type;
	public final int size;
	private final AMapToData m;
	private final int[] expected;

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
		final int max = Math.min(MapToFactory.getUpperBoundValue(type), ((int) Character.MAX_VALUE) + 3);
		expected = new int[size];
		m = genMap(MapToFactory.create(size, max + 1), expected, max, fill, seed);
	}

	protected static AMapToData genMap(AMapToData m, int[] expected, int max, boolean fill, int seed) {
		Random vals = new Random(seed);
		int size = m.size();
		if(fill) {
			int v = max == 1 ? vals.nextInt(2) : vals.nextInt(max);
			m.fill(v);
			Arrays.fill(expected, v);
		}

		for(int i = 0; i < size; i++) {
			int v = max == 1 ? vals.nextInt(2) : vals.nextInt(max);
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
		m.set(size - 1, max);

		expected[size - 1] = max;
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
				fail(m.getClass().getSimpleName() + "\n" + m.toString() + "\n");
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
			case BIT:
				compare(MapToFactory.resize(m, 5), m);
			case BYTE:
				compare(MapToFactory.resize(m, 526), m);
			case CHAR:
				compare(MapToFactory.resize(m, 612451), m);
			case INT:
				compare(MapToFactory.resize(m, 4215215), m);
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
			assertEquals("Not equal values", a.getIndex(i), b.getIndex(i));
	}

	@Test
	public void replaceMax() {
		int max = Math.min(MapToFactory.getUpperBoundValue(type), ((int) Character.MAX_VALUE) + 3);
		m.replace(max, 0);

		for(int i = 0; i < size; i++) {
			expected[i] = expected[i] == max ? 0 : expected[i];
			if(expected[i] != m.getIndex(i))
				fail("Expected equals " + Arrays.toString(expected) + "\nbut got: " + m);
		}
	}

	@Test
	public void getCountsWithDefault() {
		int nVal = m.getUnique();
		int[] counts = m.getCounts(new int[nVal + 1], size + 10);
		if(10 != counts[nVal])
			fail("Incorrect number of unique values:" + m + "\n" + Arrays.toString(counts));

	}

	@Test
	public void getCountsNoDefault() {
		int nVal = m.getUnique();
		m.getCounts(new int[nVal], size);
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
		final int max = Math.min(MapToFactory.getUpperBoundValue(type), ((int) Character.MAX_VALUE) + 3);
		assertEquals(max + 1, u);
	}

	@Test
	public void testInMemorySize() {
		long inMemorySize = m.getInMemorySize();
		long estimatedSize = MapToFactory.estimateInMemorySize(size, MapToFactory.getUpperBoundValue(type));
		assertEquals(inMemorySize, estimatedSize);
	}

}
