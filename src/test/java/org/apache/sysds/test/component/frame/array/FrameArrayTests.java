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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collection;
import java.util.Random;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.frame.data.columns.BitSetArray;
import org.apache.sysds.runtime.frame.data.columns.BooleanArray;
import org.apache.sysds.runtime.frame.data.columns.StringArray;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class FrameArrayTests {
	private static final Log LOG = LogFactory.getLog(FrameArrayTests.class.getName());

	public Array<?> a;
	public StringArray s;
	public FrameArrayType t;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		try {
			int[] seeds = new int[] {1, 6, 123, 232};
			for(FrameArrayType t : FrameArrayType.values()) {
				for(int s : seeds) {
					tests.add(new Object[] {create(t, 1, s), t});
					tests.add(new Object[] {create(t, 10, s), t});
					tests.add(new Object[] {create(t, 80, s), t});
					tests.add(new Object[] {create(t, 124, s), t});
					tests.add(new Object[] {create(t, 130, s), t});
					tests.add(new Object[] {create(t, 200, s), t});
				}
			}
			// Booleans
			tests.add(new Object[] {ArrayFactory.create(new String[] {"a", "b", "c"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"1", "0", "1"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"1", "0", "null"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"0", "0", "null"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"true", "false", "false"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"True", "False", "False"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"False", "False", "False"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"T", "F", "F"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"t", "f", "f"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"f", "t", "t"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"true", "false", "BLAA"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new float[] {0.0f, 1.0f, 1.0f, 0.0f}), FrameArrayType.FP32});
			tests.add(new Object[] {ArrayFactory.create(new double[] {0.0, 1.0, 1.0, 0.0}), FrameArrayType.FP64});
			tests.add(new Object[] {ArrayFactory.create(new long[] {0, 1, 1, 0, 0, 1}), FrameArrayType.INT64});
			tests.add(new Object[] {ArrayFactory.create(new int[] {0, 1, 1, 0, 0, 1}), FrameArrayType.INT32});
			tests.add(new Object[] {ArrayFactory.create(generateRandom01String(100, 324)), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(generateRandom01String(80, 22)), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(generateRandom01String(32, 221)), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(generateRandomTrueFalseString(32, 221)), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(generateRandomTrueFalseString(80, 221)), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(generateRandomTrueFalseString(150, 221)), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(generateRandomTFString(150, 221)), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(generateRandomTFString(22, 2)), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(generateRandomTFString(142, 4)), FrameArrayType.STRING});

			tests.add(new Object[] {ArrayFactory.create(new char[] {0, 0, 0, 0, 1, 1, 1}), FrameArrayType.CHARACTER});
			tests.add(new Object[] {ArrayFactory.create(new char[] {'t', 't', 'f', 'f', 'T'}), FrameArrayType.CHARACTER});
			tests.add(new Object[] {ArrayFactory.create(new char[] {'0', '2', '3', '4', '9'}), FrameArrayType.CHARACTER});
			tests.add(new Object[] {ArrayFactory.create(generateRandom01chars(150, 221)), FrameArrayType.CHARACTER});
			// Long to int
			tests.add(new Object[] {ArrayFactory.create(new long[] {3214, 424, 13, 22, 111, 134}), FrameArrayType.INT64});

			tests.add(new Object[] {ArrayFactory.create(new double[] {//
				Double.NaN, 424, 13, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, 134}), FrameArrayType.FP64});
			tests.add(new Object[] {ArrayFactory.create(new float[] {//
				Float.NaN, 424, 13, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, 134}), FrameArrayType.FP32});

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
		for(int i = 0; i < size; i++) {
			Object av = a.get(i);
			Object sv = s.get(i);
			if(!(av == null && sv == null))
				assertTrue(av.toString().equals(sv));
		}
	}

	@Test(expected = ArrayIndexOutOfBoundsException.class)
	public void testGetOutOfBoundsUpper() {
		a.get(a.size() + 1);
	}

	@Test(expected = ArrayIndexOutOfBoundsException.class)
	public void testGetOutOfBoundsLower() {
		a.get(-1);
	}

	@Test
	public void getSizeEstimateVsReal() {
		long memSize = a.getInMemorySize();
		long estSize = ArrayFactory.getInMemorySize(a.getValueType(), a.size());
		switch(a.getValueType()) {
			case BOOLEAN:
				if(a instanceof BitSetArray)
					estSize = BitSetArray.estimateInMemorySize(a.size());
				else
					estSize = BooleanArray.estimateInMemorySize(a.size());
			default: // nothing
		}
		if(memSize > estSize)
			fail("Estimated size is not smaller than actual:" + memSize + "  " + estSize + "\n" + a.getValueType() + " "
				+ a.getClass().getSimpleName());

	}

	@Test
	public void changeTypeString() {
		changeType(ValueType.STRING);
	}

	@Test
	public void changeTypeDouble() {
		changeType(ValueType.FP64);
	}

	@Test
	public void changeTypeFloat() {
		changeType(ValueType.FP32);
	}

	@Test
	public void changeTypeInteger() {
		changeType(ValueType.INT32);
	}

	@Test
	public void changeTypeLong() {
		changeType(ValueType.INT64);
	}

	@Test
	public void changeTypeBoolean() {
		changeType(ValueType.BOOLEAN);
	}

	@Test
	public void changeTypeCharacter() {
		changeType(ValueType.CHARACTER);
	}

	public void changeType(ValueType t) {
		try {
			Array<?> r = a.changeType(t);
			assertTrue(r.getValueType() == t);
		}
		catch(DMLRuntimeException e) {
			LOG.debug(e.getMessage());
			// okay since we want exceptions
			// in cases where the the change fail.
			// but we have to have more tests that
			// verify individual common
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testToString() {
		// just verify that there is no crash
		a.toString();
	}

	@Test
	public void getFrameArrayType() {
		if(t == FrameArrayType.BITSET)
			return;
		assertEquals(t, a.getFrameArrayType());
	}

	@Test
	public void testSliceStart() {
		int size = a.size();
		if(size <= 1)
			return;
		Array<?> aa = a.slice(0, a.size() - 2);
		compare(aa, a, 0);
	}

	@Test
	public void testSliceEnd() {
		int size = a.size();
		if(size <= 3)
			return;
		Array<?> aa = a.slice(1, a.size() - 1);
		compare(aa, a, 1);
	}

	@Test
	public void testSliceMiddle() {
		int size = a.size();
		if(size <= 3)
			return;
		Array<?> aa = a.slice(1, a.size() - 2);
		compare(aa, a, 1);
	}

	@Test
	@SuppressWarnings("unused")
	public void get() {
		Object x = null;
		switch(a.getFrameArrayType()) {
			case FP64:
				x = (double[]) a.get();
				return;
			case FP32:
				x = (float[]) a.get();
				return;
			case INT32:
				x = (int[]) a.get();
				return;
			case BOOLEAN:
				x = (boolean[]) a.get();
				return;
			case INT64:
				x = (long[]) a.get();
				return;
			case BITSET:
				x = (BitSet) a.get();
				return;
			case STRING:
				x = (String[]) a.get();
				return;
			case CHARACTER:
				x = (char[]) a.get();
			case OPTIONAL:
				try {
					a.get();
				}
				catch(NotImplementedException e) {
					// all good;
				}
				return;
			default:
				throw new NotImplementedException();
		}
	}

	@Test
	public void testSetRange01() {
		if(a.size() > 2)
			testSetRange(1, a.size() - 1, 0);
	}

	@Test
	public void testSetRange02() {
		if(a.size() > 2)
			testSetRange(0, a.size() - 2, 1);
	}

	@Test
	public void testSetRange03() {
		if(a.size() > 64)
			testSetRange(63, a.size() - 1, 3);
	}

	@Test
	public void testSetRange04() {
		if(a.size() > 64)
			testSetRange(0, a.size() - 64, 3);
	}

	@SuppressWarnings("unchecked")
	public void testSetRange(int start, int end, int off) {
		try {
			Array<?> aa = a.clone();
			switch(a.getValueType()) {
				case FP64:
					((Array<Double>) aa).set(start, end, (Array<Double>) a, off);
					break;
				case FP32:
					((Array<Float>) aa).set(start, end, (Array<Float>) a, off);
					break;
				case INT32:
					((Array<Integer>) aa).set(start, end, (Array<Integer>) a, off);
					break;
				case INT64:
					((Array<Long>) aa).set(start, end, (Array<Long>) a, off);
					break;
				case BOOLEAN:
					((Array<Boolean>) aa).set(start, end, (Array<Boolean>) a, off);
					break;
				case STRING:
					((Array<String>) aa).set(start, end, (Array<String>) a, off);
					break;
				case CHARACTER:
					((Array<Character>) aa).set(start, end, (Array<Character>) a, off);
					break;
				default:
					throw new NotImplementedException();
			}
			compareSetSubRange(aa, a, start, end, off, aa.getValueType());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testSetRange_1() {
		if(a.size() > 10)
			testSetRange(0, 10, 20, 132);
	}

	@SuppressWarnings("unchecked")
	public void testSetRange(int start, int end, int otherSize, int seed) {
		try {
			Array<?> other = create(a.getFrameArrayType(), otherSize, seed);
			try {
				other = other.changeTypeWithNulls(a.getValueType());
			}
			catch(DMLRuntimeException e) {
				return;// all good.
			}

			Array<?> aa = a.clone();
			switch(a.getValueType()) {
				case FP64:
					((Array<Double>) aa).set(start, end, (Array<Double>) other);
					break;
				case FP32:
					((Array<Float>) aa).set(start, end, (Array<Float>) other);
					break;
				case INT32:
					((Array<Integer>) aa).set(start, end, (Array<Integer>) other);
					break;
				case INT64:
					((Array<Long>) aa).set(start, end, (Array<Long>) other);
					break;
				case BOOLEAN:
					((Array<Boolean>) aa).set(start, end, (Array<Boolean>) other);
					break;
				case STRING:
					((Array<String>) aa).set(start, end, (Array<String>) other);
					break;
				case CHARACTER:
					((Array<Character>) aa).set(start, end, (Array<Character>) other);
					break;
				default:
					throw new NotImplementedException();
			}
			compareSetSubRange(aa, other, start, end, 0, aa.getValueType());

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void set() {
		switch(a.getValueType()) {
			case FP64:
				Double vd = 1324.42d;
				((Array<Double>) a).set(0, vd);
				assertEquals(((Array<Double>) a).get(0), vd, 0.0000001);
				return;
			case FP32:
				Float vf = 1324.42f;
				((Array<Float>) a).set(0, vf);
				assertEquals(((Array<Float>) a).get(0), vf, 0.0000001);
				return;
			case INT32:
				Integer vi = 1324;
				((Array<Integer>) a).set(0, vi);
				assertEquals(((Array<Integer>) a).get(0), vi);
				return;
			case INT64:
				Long vl = 1324L;
				((Array<Long>) a).set(0, vl);
				assertEquals(((Array<Long>) a).get(0), vl);
				return;
			case BOOLEAN:
				Boolean vb = true;
				((Array<Boolean>) a).set(0, vb);
				assertEquals(((Array<Boolean>) a).get(0), vb);
				return;
			case STRING:
				String vs = "1324L";
				a.set(0, vs);
				assertEquals(((Array<String>) a).get(0), vs);
				return;
			case CHARACTER:
				Character c = '~';
				((Array<Character>) a).set(0, c);
				assertEquals(((Array<Character>) a).get(0), c);
				return;
			default:
				throw new NotImplementedException();
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setDouble() {
		Double vd = 1.0d;
		a.set(0, vd);
		switch(a.getValueType()) {
			case FP64:
				assertEquals(((Array<Double>) a).get(0), vd, 0.0000001);
				return;
			case FP32:
				assertEquals(((Array<Float>) a).get(0), vd, 0.0000001);
				return;
			case INT32:
				assertEquals(((Array<Integer>) a).get(0), Integer.valueOf((int) (double) vd));
				return;
			case INT64:
				assertEquals(((Array<Long>) a).get(0), Long.valueOf((long) (double) vd));
				return;
			case BOOLEAN:
				assertEquals(((Array<Boolean>) a).get(0), vd == 1.0d);
				return;
			case STRING:
				assertEquals(((Array<String>) a).get(0), Double.toString(vd));
				return;
			case CHARACTER:
				assertEquals((int) ((Array<Character>) a).get(0), 1);
				return;
			default:
				throw new NotImplementedException();
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setDouble_2() {
		Double vd = 0.0d;
		a.set(0, vd);
		switch(a.getValueType()) {
			case FP64:
				assertEquals(((Array<Double>) a).get(0), vd, 0.0000001);
				return;
			case FP32:
				assertEquals(((Array<Float>) a).get(0), vd, 0.0000001);
				return;
			case INT32:
				assertEquals(((Array<Integer>) a).get(0), Integer.valueOf((int) (double) vd));
				return;
			case INT64:
				assertEquals(((Array<Long>) a).get(0), Long.valueOf((long) (double) vd));
				return;
			case BOOLEAN:
				assertEquals(((Array<Boolean>) a).get(0), false);
				return;
			case STRING:
				assertEquals(((Array<String>) a).get(0), Double.toString(vd));
				return;
			case CHARACTER:
				assertEquals(((Array<Character>) a).get(0), Character.valueOf((char) 0));
				return;
			default:
				throw new NotImplementedException();
		}
	}

	@Test
	public void analyzeValueType() {
		ValueType av = a.analyzeValueType().getKey();
		switch(a.getValueType()) {
			case BOOLEAN:
				switch(av) {
					case BOOLEAN:
						return;
					default:
						fail("Invalid type returned from analyze valueType " + av);
				}
			case INT32:
				switch(av) {
					case BOOLEAN:
					case INT32:
					case UINT8:
						return;
					default:
						fail("Invalid type returned from analyze valueType " + av);
				}
			case INT64:
				switch(av) {
					case BOOLEAN:
					case INT32:
					case UINT8:
					case INT64:
						return;
					default:
						fail("Invalid type returned from analyze valueType " + av + " " + a);
				}
			case UINT8:
				switch(av) {
					case BOOLEAN:
					case UINT8:
						return;
					default:
						fail("Invalid type returned from analyze valueType " + av);
				}
			case FP32:
				switch(av) {
					case BOOLEAN:
					case INT32:
					case UINT8:
					case INT64:
					case FP32:
						return;
					default:
						fail("Invalid type returned from analyze valueType " + av);
				}
			case FP64:
				switch(av) {
					case BOOLEAN:
					case INT32:
					case UINT8:
					case INT64:
					case FP32:
					case FP64:
						return;
					default:
						fail("Invalid type returned from analyze valueType " + av);
				}
			case STRING:
				break;// all allowed
			case UNKNOWN:
				fail("Not allowed to be unknown");
			default:
				break;
		}
	}

	@Test
	public void setNull() {
		// should not crash
		a.set(0, (String) null);
	}

	@Test
	public void toByteArray() {
		if(a.getValueType() == ValueType.STRING)
			return;
		// just test that it serialize as byte array with no crashes
		a.getAsByteArray();
	}

	@Test
	public void appendString() {
		Array<?> aa = a.clone();

		switch(a.getValueType()) {
			case BOOLEAN:
				aa.append("0");
				assertEquals((Boolean) aa.get(aa.size() - 1), false);
				aa.append("1");
				assertEquals((Boolean) aa.get(aa.size() - 1), true);
				break;
			case FP32:
				float vf = 3215216.222f;
				String vfs = vf + "";
				aa.append(vfs);
				assertEquals((float) aa.get(aa.size() - 1), vf, 0.00001);

				vf = 32152336.222f;
				vfs = vf + "";
				aa.append(vfs);
				assertEquals((float) aa.get(aa.size() - 1), vf, 0.00001);
				break;
			case FP64:
				double vd = 3215216.222;
				String vds = vd + "";
				aa.append(vds);
				assertEquals((double) aa.get(aa.size() - 1), vd, 0.00001);

				vd = 222.222;
				vds = vd + "";
				aa.append(vds);
				assertEquals((double) aa.get(aa.size() - 1), vd, 0.00001);
				break;
			case INT32:
				int vi = 321521;
				String vis = vi + "";
				aa.append(vis);
				assertEquals((int) aa.get(aa.size() - 1), vi);

				vi = -2321;
				vis = vi + "";
				aa.append(vis);
				assertEquals((int) aa.get(aa.size() - 1), vi);
				break;
			case INT64:
				long vl = 321521;
				String vls = vl + "";
				aa.append(vls);
				assertEquals((long) aa.get(aa.size() - 1), vl);

				vl = -22223;
				vls = vl + "";
				aa.append(vls);
				assertEquals((long) aa.get(aa.size() - 1), vl);
				break;
			case STRING:
				String vs = "ThisIsAMonkeyTestSting";
				aa.append(vs);
				assertEquals((String) aa.get(aa.size() - 1), vs);

				vs = "£$&*%!))";
				aa.append(vs);
				assertEquals((String) aa.get(aa.size() - 1), vs);
				break;
			case UINT8:
				int vi8 = 234;
				String vi8s = vi8 + "";
				aa.append(vi8s);
				assertEquals((int) aa.get(aa.size() - 1), vi8);

				vi8 = 42;
				vi8s = vi8 + "";
				aa.append(vi8s);
				assertEquals((int) aa.get(aa.size() - 1), vi8);
				break;
			case CHARACTER:
				char vc = '@';
				String vci = vc + "";
				aa.append(vci);
				assertEquals((char) aa.get(aa.size() - 1), vc);
				vc = (char) 42;
				vci = vc + "";
				aa.append(vci);
				assertEquals((char) aa.get(aa.size() - 1), vc);
				break;
			case UNKNOWN:
			default:
				throw new DMLRuntimeException("Invalid type");
		}
	}

	@Test
	public void appendNull() {
		Array<?> aa = a.clone();

		aa.append((String) null);
		if(a.getFrameArrayType() == FrameArrayType.OPTIONAL)
			assertEquals((String) aa.get(aa.size() - 1), null);
		else {
			switch(a.getValueType()) {
				case BOOLEAN:
					assertEquals((Boolean) aa.get(aa.size() - 1), false);
					break;
				case FP32:
					assertEquals((float) aa.get(aa.size() - 1), 0.0, 0.00001);
					break;
				case FP64:
					assertEquals((double) aa.get(aa.size() - 1), 0.0, 0.00001);
					break;
				case INT32:
					assertEquals((int) aa.get(aa.size() - 1), 0);
					break;
				case INT64:
					assertEquals((long) aa.get(aa.size() - 1), 0);
					break;
				case STRING:
					assertEquals((String) aa.get(aa.size() - 1), null);
					break;
				case UINT8:
					assertEquals((int) aa.get(aa.size() - 1), 0);
					break;
				case CHARACTER:
					assertEquals((char) aa.get(aa.size() - 1), 0);
					break;
				case UNKNOWN:
				default:
					throw new DMLRuntimeException("Invalid type");
			}
		}
	}

	@Test
	public void append60Null() {
		Array<?> aa = a.clone();

		for(int i = 0; i < 60; i++)
			aa.append((String) null);
		if(a.getFrameArrayType() == FrameArrayType.OPTIONAL)
			assertEquals((String) aa.get(aa.size() - 1), null);
		else {
			switch(a.getValueType()) {
				case BOOLEAN:
					assertEquals((Boolean) aa.get(aa.size() - 1), false);
					break;
				case FP32:
					assertEquals((float) aa.get(aa.size() - 1), 0.0, 0.00001);
					break;
				case FP64:
					assertEquals((double) aa.get(aa.size() - 1), 0.0, 0.00001);
					break;
				case INT32:
					assertEquals((int) aa.get(aa.size() - 1), 0);
					break;
				case INT64:
					assertEquals((long) aa.get(aa.size() - 1), 0);
					break;
				case STRING:
					assertEquals((String) aa.get(aa.size() - 1), null);
					break;
				case UINT8:
					assertEquals((int) aa.get(aa.size() - 1), 0);
					break;
				case CHARACTER:
					assertEquals((char) aa.get(aa.size() - 1), 0);
					break;
				case UNKNOWN:
				default:
					throw new DMLRuntimeException("Invalid type");
			}
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void testSetNzSelf() {
		Array<?> aa = a.clone();
		try {

			switch(a.getValueType()) {
				case BOOLEAN:
					((Array<Boolean>) aa).setNz((Array<Boolean>) a);
					break;
				case FP32:
					((Array<Float>) aa).setNz((Array<Float>) a);
					break;
				case FP64:
					((Array<Double>) aa).setNz((Array<Double>) a);
					break;
				case INT32:
				case UINT8:
					((Array<Integer>) aa).setNz((Array<Integer>) a);
					break;
				case INT64:
					((Array<Long>) aa).setNz((Array<Long>) a);
					break;
				case STRING:
					((Array<String>) aa).setNz((Array<String>) a);
					break;
				case CHARACTER:
					((Array<Character>) aa).setNz((Array<Character>) a);
					break;
				case UNKNOWN:
				default:
					throw new DMLRuntimeException("Invalid type");
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

		compare(aa, a);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void testSetNzString() {
		Array<?> aa = a.clone();
		Array<String> af = (Array<String>) aa.changeType(ValueType.STRING);
		try {

			aa.setFromOtherTypeNz(af);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

		compare(aa, a);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void testSetNzStringWithNull() {
		Array<?> aa = a.clone();
		Array<String> af = (Array<String>) aa.changeTypeWithNulls(ValueType.STRING);
		try {

			aa.setFromOtherTypeNz(af);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

		compare(aa, a);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void testSetFromString() {
		Array<?> aa = a.clone();
		Array<String> af = (Array<String>) aa.changeType(ValueType.STRING);
		try {

			aa.setFromOtherType(0, af.size() - 1, af);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

		compare(aa, a);
	}

	@Test
	public void testSetFromStringWithNull() {
		Array<?> aa = a.clone();
		Array<?> af;
		if(aa.getFrameArrayType() == FrameArrayType.OPTIONAL && aa.getValueType() != ValueType.STRING)
			af = aa.changeTypeWithNulls(ValueType.FP64);
		else
			af = aa.changeTypeWithNulls(ValueType.STRING);

		try {

			aa.setFromOtherType(0, af.size() - 1, af);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

		compare(aa, a);
	}

	@Test
	public void resetTestCase() {
		try {

			Array<?> aa = a.clone();
			aa.reset(10);
			if(aa.getValueType() == ValueType.STRING || aa.getFrameArrayType() == FrameArrayType.OPTIONAL) {
				for(int i = 0; i < 10; i++) {
					assertEquals(null, aa.get(i));
				}
			}
			else {

				String v = aa.get(0).toString();
				for(int i = 1; i < 10; i++) {
					assertEquals(v, aa.get(i).toString());
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testSerializationSize() {
		try {
			// Serialize out
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			a.write(fos);
			long s = (long) fos.size();
			long e = a.getExactSerializedSize();
			assertEquals(s, e);
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
	public void testFindEmpty() {
		boolean[] select = new boolean[a.size()];
		a.findEmpty(select);
		Object d = ArrayFactory.defaultNullValue(a.getValueType());
		final String errStart = a.getClass().getSimpleName() + " ";
		for(int i = 0; i < select.length; i++) {
			if(!select[i])
				assertTrue(errStart + a.get(i), //
					(a.get(i) == null) || a.get(i).equals(d) || a.get(i).equals("0"));
		}
	}

	protected static void compare(Array<?> a, Array<?> b) {
		int size = a.size();
		String err = a.getClass().getSimpleName() + " " + a.getValueType() + " " + b.getClass().getSimpleName() + " "
			+ b.getValueType();
		assertTrue(a.size() == b.size());
		for(int i = 0; i < size; i++) {
			final Object av = a.get(i);
			final Object bv = b.get(i);
			if(!(av == null && bv == null))
				assertTrue(err, av.toString().equals(bv.toString()));
		}
	}

	protected static void compare(Array<?> sub, Array<?> b, int off) {
		int size = sub.size();
		for(int i = 0; i < size; i++) {
			final Object av = sub.get(i);
			final Object bv = b.get(i + off);
			if(!(av == null && bv == null))
				assertTrue(av.toString().equals(bv.toString()));
		}
	}

	protected static void compareSetSubRange(Array<?> out, Array<?> in, int rl, int ru, int off, ValueType vt) {
		for(int i = rl; i <= ru; i++, off++) {
			Object av = out.get(i);
			Object bv = in.get(off);
			if(!(av == null && bv == null)) {
				String v1 = av.toString();
				String v2 = bv.toString();
				assertEquals("i: " + i + " args: " + rl + " " + ru + " " + (off - i) + " " + out.size(), v1, v2);
			}
		}
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
			case BITSET:
				return ArrayFactory.create(generateRandomBitSet(size, seed), size);
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
			case CHARACTER:
				return ArrayFactory.create(generateRandomChar(size, seed));
			case OPTIONAL:
				Random r = new Random(seed);
				switch(r.nextInt(7)) {
					case 0:
						return ArrayFactory.create(generateRandomIntegerOpt(size, seed));
					case 1:
						return ArrayFactory.create(generateRandomLongOpt(size, seed));
					case 2:
						return ArrayFactory.create(generateRandomDoubleOpt(size, seed));
					case 3:
						return ArrayFactory.create(generateRandomFloatOpt(size, seed));
					case 4:
						return ArrayFactory.create(generateRandomCharacterOpt(size, seed));
					default:
						return ArrayFactory.create(generateRandomBooleanOpt(size, seed));
				}
			default:
				throw new DMLRuntimeException("Unsupported value type: " + t);

		}
	}

	protected static StringArray toStringArray(Array<?> a) {
		String[] ret = new String[a.size()];
		for(int i = 0; i < a.size(); i++) {
			Object v = a.get(i);
			ret[i] = v == null ? null : v.toString();
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

	public static String[] generateRandom01String(int size, int seed) {
		Random r = new Random(seed);
		String[] ret = new String[size];
		for(int i = 0; i < size; i++)
			ret[i] = r.nextInt(1) + "";
		return ret;
	}

	public static char[] generateRandom01chars(int size, int seed) {
		Random r = new Random(seed);
		char[] ret = new char[size];
		for(int i = 0; i < size; i++)
			ret[i] = (char) r.nextInt(1);
		return ret;
	}

	public static String[] generateRandomTrueFalseString(int size, int seed) {
		Random r = new Random(seed);
		String[] ret = new String[size];
		for(int i = 0; i < size; i++)
			ret[i] = r.nextInt(1) == 1 ? "true" : "false";
		return ret;
	}

	public static String[] generateRandomTFString(int size, int seed) {
		Random r = new Random(seed);
		String[] ret = new String[size];
		for(int i = 0; i < size; i++)
			ret[i] = r.nextInt(1) == 1 ? "t" : "f";
		return ret;
	}

	protected static boolean[] generateRandomBoolean(int size, int seed) {
		Random r = new Random(seed);
		boolean[] ret = new boolean[size];
		for(int i = 0; i < size; i++)
			ret[i] = r.nextBoolean();
		return ret;
	}

	protected static Boolean[] generateRandomBooleanOpt(int size, int seed) {
		Random r = new Random(seed);
		Boolean[] ret = new Boolean[size];
		for(int i = 0; i < size; i++) {
			if(r.nextBoolean())
				ret[i] = r.nextBoolean();
			else
				ret[i] = null;
		}
		return ret;
	}

	protected static Integer[] generateRandomIntegerOpt(int size, int seed) {
		Random r = new Random(seed);
		Integer[] ret = new Integer[size];
		for(int i = 0; i < size; i++) {
			if(r.nextBoolean())
				ret[i] = r.nextInt();
			else
				ret[i] = null;
		}
		return ret;
	}

	protected static Long[] generateRandomLongOpt(int size, int seed) {
		Random r = new Random(seed);
		Long[] ret = new Long[size];
		for(int i = 0; i < size; i++) {
			if(r.nextBoolean())
				ret[i] = r.nextLong() / 10000L;
			else
				ret[i] = null;
		}
		return ret;
	}

	protected static Float[] generateRandomFloatOpt(int size, int seed) {
		Random r = new Random(seed);
		Float[] ret = new Float[size];
		for(int i = 0; i < size; i++) {
			if(r.nextBoolean())
				ret[i] = r.nextFloat();
			else
				ret[i] = null;
		}
		return ret;
	}

	protected static Character[] generateRandomCharacterOpt(int size, int seed) {
		Random r = new Random(seed);
		Character[] ret = new Character[size];
		for(int i = 0; i < size; i++) {
			if(r.nextBoolean())
				ret[i] = (char) r.nextInt((int) Character.MAX_VALUE);
			else
				ret[i] = null;
		}
		return ret;
	}

	protected static Double[] generateRandomDoubleOpt(int size, int seed) {
		Random r = new Random(seed);
		Double[] ret = new Double[size];
		for(int i = 0; i < size; i++) {
			if(r.nextBoolean())
				ret[i] = r.nextDouble();
			else
				ret[i] = null;
		}
		return ret;
	}

	protected static int[] generateRandomInteger(int size, int seed) {
		Random r = new Random(seed);
		int[] ret = new int[size];
		for(int i = 0; i < size; i++)
			ret[i] = r.nextInt();
		return ret;
	}

	protected static char[] generateRandomChar(int size, int seed) {
		Random r = new Random(seed);
		char[] ret = new char[size];
		for(int i = 0; i < size; i++)
			ret[i] = (char) r.nextInt((int) Character.MAX_VALUE);
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
			ret[i] = r.nextLong() / 10000L;
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

	protected static BitSet generateRandomBitSet(int size, int seed) {
		Random r = new Random(seed);
		int nLongs = size / 64 + 1;
		long[] longs = new long[nLongs];
		for(int i = 0; i < nLongs; i++)
			longs[i] = r.nextLong();

		return BitSet.valueOf(longs);
	}
}
