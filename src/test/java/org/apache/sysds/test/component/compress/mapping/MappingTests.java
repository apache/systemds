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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class MappingTests {

	public final int seed;
	public final MAP_TYPE type;
	public final int size;

	private AMapToData m;
	private int[] expected;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		for(MAP_TYPE t : MAP_TYPE.values()) {
			tests.add(new Object[] {1, t, 13, false});
			tests.add(new Object[] {3, t, 13, false});
			tests.add(new Object[] {3, t, 63, false});
			tests.add(new Object[] {3, t, 64, false});
			tests.add(new Object[] {3, t, 65, false});
			tests.add(new Object[] {5, t, 1234, false});
			tests.add(new Object[] {5, t, 13, true});
		}
		return tests;
	}

	public MappingTests(int seed, MAP_TYPE type, int size, boolean fill) {
		this.seed = seed;
		this.type = type;
		this.size = size;
		Random vals = new Random(seed);
		final int max = getMax();
		m = MapToFactory.create(size, max);
		expected = new int[size];
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
		m.set(size - 1, 1);
		expected[size - 1] = 1;

	}

	@Test
	public void isEqual() {
		for(int i = 0; i < size; i++) {
			assertEquals("Expected equals " + Arrays.toString(expected) + "\nbut got: " + m, expected[i],
				m.getIndex(i));
		}
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
			assertEquals(m.getClass().getSimpleName() + "\n" + m.toString() + "\n", size, m.getExactSizeOnDisk());
		}
		catch(IOException e) {
			throw new RuntimeException("Error in io", e);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	private static void compare(AMapToData a, AMapToData b) {
		final int size = Math.max(a.size(), b.size());
		for(int i = 0; i < size; i++) {
			assertEquals("Expected equals " + a + "\nbut got: " + b, a.getIndex(i), b.getIndex(i));
		}
	}

	@Test
	public void testInMemorySize() {
		long inMemorySize = m.getInMemorySize();
		long estimatedSize = MapToFactory.estimateInMemorySize(size, getMax());
		assertEquals(inMemorySize, estimatedSize);
	}

	private int getMax() {
		switch(type) {
			case BIT:
				return 1;
			case BYTE:
				return (int) Math.pow(2, 8);
			case CHAR:
				return (int) Math.pow(2, 16);
			default:
				return Integer.MAX_VALUE;
		}
	}

}
