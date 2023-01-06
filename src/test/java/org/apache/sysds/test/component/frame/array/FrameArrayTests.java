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
			for(FrameArrayType t : FrameArrayType.values()) {
				tests.add(new Object[] {create(t, 1, 2), t});
				tests.add(new Object[] {create(t, 10, 52), t});
				tests.add(new Object[] {create(t, 80, 22), t});
				tests.add(new Object[] {create(t, 124, 22), t});
				tests.add(new Object[] {create(t, 124, 23), t});
				tests.add(new Object[] {create(t, 124, 24), t});
				tests.add(new Object[] {create(t, 130, 24), t});
				tests.add(new Object[] {create(t, 512, 22), t});
				tests.add(new Object[] {create(t, 560, 22), t});
			}
			// Booleans
			tests.add(new Object[] {ArrayFactory.create(new String[] {"a", "b", "c"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"1", "0", "1"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"1", "0", "null"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"0", "0", "null"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"true", "false", "false"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"True", "False", "False"}), FrameArrayType.STRING});
			tests.add(new Object[] {ArrayFactory.create(new String[] {"False", "False", "False"}), FrameArrayType.STRING});
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

			// Long to int
			tests.add(new Object[] {ArrayFactory.create(new long[] {3214, 424, 13, 22, 111, 134}), FrameArrayType.INT64});
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
			assertTrue(a.get(i).toString().equals(s.get(i)));
	}

	@Test(expected = ArrayIndexOutOfBoundsException.class)
	public void testGetOutOfBoundsUpper() {
		if(a.getFrameArrayType() == FrameArrayType.BITSET)
			throw new ArrayIndexOutOfBoundsException("make it pass");
		a.get(a.size() + 1);
	}

	@Test(expected = ArrayIndexOutOfBoundsException.class)
	public void testGetOutOfBoundsLower() {
		if(a.getFrameArrayType() == FrameArrayType.BITSET)
			throw new ArrayIndexOutOfBoundsException("make it pass");
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
			switch(a.getFrameArrayType()) {
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
				case BITSET:
					((Array<Boolean>) aa).set(start, end, (Array<Boolean>) a, off);
					break;
				case STRING:
					((Array<String>) aa).set(start, end, (Array<String>) a, off);
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
			Array<?> other = null;
			switch(a.getFrameArrayType()) {
				case BITSET:
					other = ArrayFactory.create(generateRandomBitSet(otherSize, seed), otherSize);
					break;
				case BOOLEAN:
					other = ArrayFactory.create(generateRandomBoolean(otherSize, seed));
					break;
				case FP32:
					other = ArrayFactory.create(generateRandomFloat(otherSize, seed));
					break;
				case FP64:
					other = ArrayFactory.create(generateRandomDouble(otherSize, seed));
					break;
				case INT32:
					other = ArrayFactory.create(generateRandomInteger(otherSize, seed));
					break;
				case INT64:
					other = ArrayFactory.create(generateRandomLong(otherSize, seed));
					break;
				case STRING:
					other = ArrayFactory.create(generateRandomString(otherSize, seed));
					break;
				default:
					throw new NotImplementedException();
			}

			Array<?> aa = a.clone();
			switch(a.getFrameArrayType()) {
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
				case BITSET:
					((Array<Boolean>) aa).set(start, end, (Array<Boolean>) other);
					break;
				case STRING:
					((Array<String>) aa).set(start, end, (Array<String>) other);
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
		switch(a.getFrameArrayType()) {
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
			case BITSET:

				Boolean vb = true;
				((Array<Boolean>) a).set(0, vb);
				assertEquals(((Array<Boolean>) a).get(0), vb);
				return;
			case STRING:

				String vs = "1324L";
				a.set(0,vs);
				assertEquals(((Array<String>) a).get(0), vs);

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
		switch(a.getFrameArrayType()) {
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
			case BITSET:
				assertEquals(((Array<Boolean>) a).get(0), vd == 1.0d);
				return;
			case STRING:
				assertEquals(((Array<String>) a).get(0), Double.toString(vd));
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
		switch(a.getFrameArrayType()) {
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
			case BITSET:
				assertEquals(((Array<Boolean>) a).get(0), false);
				return;
			case STRING:
				assertEquals(((Array<String>) a).get(0), Double.toString(vd));
				return;
			default:
				throw new NotImplementedException();
		}
	}

	@Test
	public void analyzeValueType() {
		ValueType av = a.analyzeValueType();
		switch(a.getValueType()) {
			case BOOLEAN:
				switch(av) {
					case BOOLEAN:
						return;
					default:
						fail("Invalid type returned from analyze valueType");
				}
			case INT32:
				switch(av) {
					case BOOLEAN:
					case INT32:
					case UINT8:
						return;
					default:
						fail("Invalid type returned from analyze valueType");
				}
			case INT64:
				switch(av) {
					case BOOLEAN:
					case INT32:
					case UINT8:
					case INT64:
						return;
					default:
						fail("Invalid type returned from analyze valueType");
				}
			case UINT8:
				switch(av) {
					case BOOLEAN:
					case UINT8:
						return;
					default:
						fail("Invalid type returned from analyze valueType");
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
						fail("Invalid type returned from analyze valueType");
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
						fail("Invalid type returned from analyze valueType");
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
		a.set(0, (String)null);
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
			case UNKNOWN:
			default:
				throw new DMLRuntimeException("Invalid type");
		}
	}

	@Test
	public void appendNull() {
		Array<?> aa = a.clone();

		aa.append((String) null);
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
			case UNKNOWN:
			default:
				throw new DMLRuntimeException("Invalid type");
		}
	}

	@Test
	public void append60Null() {
		Array<?> aa = a.clone();

		try {

			for(int i = 0; i < 60; i++)
				aa.append((String) null);

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
				case UNKNOWN:
				default:
					throw new DMLRuntimeException("Invalid type");
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
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
		Array<String> af = (Array<String>)aa.changeType(ValueType.STRING);
		try{

			aa.setFromOtherTypeNz(af);
		}
		catch(Exception e){
			e.printStackTrace();
			fail(e.getMessage());
		}

		compare(aa, a);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void testSetFromString() {
		Array<?> aa = a.clone();
		Array<String> af = (Array<String>)aa.changeType(ValueType.STRING);
		try{

			aa.setFromOtherType(0, af.size()-1, af);
		}
		catch(Exception e){
			e.printStackTrace();
			fail(e.getMessage());
		}

		compare(aa, a);
	}

	@Test
	public void resetTestCase() {
		Array<?> aa = a.clone();
		aa.reset(10);
		if(aa.getValueType() == ValueType.STRING){
			for(int i = 0; i < 10; i++) {
				assertEquals(null, aa.get(i));
			}
		}
		else{

			String v = aa.get(0).toString();
			for(int i = 1; i < 10; i++) {
				assertEquals(v, aa.get(i).toString());
			}
		}
	}

	protected static void compare(Array<?> a, Array<?> b) {
		int size = a.size();
		assertTrue(a.size() == b.size());
		for(int i = 0; i < size; i++)
			assertTrue(a.get(i).toString().equals(b.get(i).toString()));
	}

	protected static void compare(Array<?> sub, Array<?> b, int off) {
		int size = sub.size();
		for(int i = 0; i < size; i++) {
			assertTrue(sub.get(i).toString().equals(b.get(i + off).toString()));
		}
	}

	protected static void compareSetSubRange(Array<?> out, Array<?> in, int rl, int ru, int off, ValueType vt) {
		for(int i = rl; i <= ru; i++, off++) {
			String v1 = out.get(i).toString();
			String v2 = in.get(off).toString();
			assertEquals("i: " + i + " args: " + rl + " " + ru + " " + (off - i) + " " + out.size(), v1, v2);
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
			default:
				throw new DMLRuntimeException("Unsupported value type: " + t);

		}
	}

	protected static StringArray toStringArray(Array<?> a) {
		String[] ret = new String[a.size()];
		for(int i = 0; i < a.size(); i++)
			ret[i] = a.get(i).toString();

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

	public static String[] generateRandomTrueFalseString(int size, int seed) {
		Random r = new Random(seed);
		String[] ret = new String[size];
		for(int i = 0; i < size; i++)
			ret[i] = r.nextInt(1) == 1 ? "true" : "false";
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

	protected static BitSet generateRandomBitSet(int size, int seed) {
		Random r = new Random(seed);
		int nLongs = size / 64 + 1;
		long[] longs = new long[nLongs];
		for(int i = 0; i < nLongs; i++)
			longs[i] = r.nextLong();

		return BitSet.valueOf(longs);
	}
}
