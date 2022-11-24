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

package org.apache.sysds.test.component.frame.array;

import static org.junit.Assert.fail;
import static org.junit.Assume.assumeTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.frame.data.columns.StringArray;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class FrameArrayTests {

	public Array<?> a;
	public StringArray s;
	public FrameArrayType t;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		try {
			for(FrameArrayType t : FrameArrayType.values()) {
				tests.add(new Object[] {create(t, 1, 2), t});
				tests.add(new Object[] {create(t, 10, 52), t});
				tests.add(new Object[] {create(t, 80, 22), t});
			}
			tests.add(new Object[] {ArrayFactory.create(new String[] {"a", "b", "c"}), FrameArrayType.STRING});
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	public FrameArrayTests(Array<?> a, FrameArrayType t) {
		try {

			this.a = a;
			this.t = t;
			this.s = toStringArray(a);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed initializing Frame Test");
		}
	}

	@Test
	public void serialize() {
		compare(a, serializeAndBack(a));
	}

	@Test
	public void testGet() {
		int size = a.size();
		for(int i = 0; i < size; i++)
			assumeTrue(a.get(i).toString().equals(s.get(i)));
	}

	@Test(expected = ArrayIndexOutOfBoundsException.class)
	public void testGetOutOfBoundsUpper() {
		a.get(a.size());
	}

	@Test(expected = ArrayIndexOutOfBoundsException.class)
	public void testGetOutOfBoundsLower() {
		a.get(-1);
	}

	@Test
	public void getSizeEstimateVsReal() {
		assumeTrue(a.getInMemorySize() <= ArrayFactory.getInMemorySize(a.getValueType(), a.size()));
	}

	protected static void compare(Array<?> a, Array<?> b) {
		int size = a.size();
		for(int i = 0; i < size; i++)
			assumeTrue(a.get(i).toString().equals(b.get(i).toString()));
	}

	protected static Array<?> serializeAndBack(Array<?> g) {
		try {
			int nRow = g.size();
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			g.write(fos);
			DataInputStream fis = new DataInputStream(new ByteArrayInputStream(bos.toByteArray()));
			return ArrayFactory.read(fis, nRow);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Error in io", e);
		}
	}

	protected static Array<?> create(FrameArrayType t, int size, int seed) {
		switch(t) {
			case STRING:
				return ArrayFactory.create(generateRandomString(size, seed));
			case BOOLEAN:
				return ArrayFactory.create(generateRandomBoolean(size, seed));
			case INT32:
				return ArrayFactory.create(generateRandomInteger(size, seed));
			case INT64:
				return ArrayFactory.create(generateRandomLong(size, seed));
			case FP32:
				return ArrayFactory.create(generateRandomFloat(size, seed));
			case FP64:
				return ArrayFactory.create(generateRandomDouble(size, seed));
			default:
				throw new DMLRuntimeException("Unsupported value type: " + t);

		}
	}

	protected static StringArray toStringArray(Array<?> a) {
		String[] ret = new String[a.size()];
		for(int i = 0; i < a.size(); i++) {
			ret[i] = a.get(i).toString();
		}
		return ArrayFactory.create(ret);
	}

	public static String[] generateRandomString(int size, int seed) {
		Random r = new Random(seed);
		String[] ret = new String[size];
		for(int i = 0; i < size; i++)
			ret[i] = r.nextInt(99) + "ad " + r.nextInt(99);
		return ret;
	}

	protected static boolean[] generateRandomBoolean(int size, int seed) {
		Random r = new Random(seed);
		boolean[] ret = new boolean[size];
		for(int i = 0; i < size; i++)
			ret[i] = r.nextBoolean();
		return ret;
	}

	protected static int[] generateRandomInteger(int size, int seed) {
		Random r = new Random(seed);
		int[] ret = new int[size];
		for(int i = 0; i < size; i++)
			ret[i] = r.nextInt();
		return ret;
	}

	protected static int[] generateRandomInt8(int size, int seed) {
		Random r = new Random(seed);
		int[] ret = new int[size];
		for(int i = 0; i < size; i++)
			ret[i] = r.nextInt(256) - 127;
		return ret;
	}

	protected static long[] generateRandomLong(int size, int seed) {
		Random r = new Random(seed);
		long[] ret = new long[size];
		for(int i = 0; i < size; i++)
			ret[i] = r.nextLong();
		return ret;
	}

	protected static float[] generateRandomFloat(int size, int seed) {
		Random r = new Random(seed);
		float[] ret = new float[size];
		for(int i = 0; i < size; i++)
			ret[i] = r.nextFloat();
		return ret;
	}

	protected static double[] generateRandomDouble(int size, int seed) {
		Random r = new Random(seed);
		double[] ret = new double[size];
		for(int i = 0; i < size; i++)
			ret[i] = r.nextDouble();
		return ret;
	}
}
