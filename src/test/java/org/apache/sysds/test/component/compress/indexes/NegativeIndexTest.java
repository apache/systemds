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

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.ArrayIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex.ColIndexType;
import org.apache.sysds.runtime.compress.colgroup.indexes.RangeIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.SingleIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.TwoIndex;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.junit.Test;

public class NegativeIndexTest {

	@Test(expected = DMLCompressionException.class)
	public void notValidRead() {
		try {

			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			fos.writeByte(ColIndexType.UNKNOWN.ordinal());
			ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
			DataInputStream fis = new DataInputStream(bis);
			ColIndexFactory.read(fis);
		}
		catch(IOException e) {
			e.printStackTrace();
			fail("Wrong type of exception");
		}
	}

	@Test
	public void equalsArray1() {
		assertFalse(new ArrayIndex(new int[] {1, 2, 3}).equals(notRelated()));
	}

	@Test
	public void equalsRange() {
		assertFalse(new RangeIndex(1, 2).equals(notRelated()));
	}

	@Test
	public void equalsTwo() {
		assertFalse(new TwoIndex(1, 2).equals(notRelated()));
	}

	@Test
	public void equalsSingle() {
		assertFalse(new SingleIndex(2).equals(notRelated()));
	}

	@Test
	public void notEqualsArray1() {
		assertFalse(new ArrayIndex(new int[] {1, 2, 3}).equals((Object) new SingleIndex(2)));
	}

	@Test
	public void notEqualsArray2() {
		assertFalse(new ArrayIndex(new int[] {1, 2, 3}).equals((Object) new TwoIndex(1, 2)));
	}

	@Test
	public void equalsArray2() {
		assertTrue(new ArrayIndex(new int[] {1, 2}).equals(new TwoIndex(1, 2)));
	}

	@Test
	public void equalsArray3() {
		assertFalse(new ArrayIndex(new int[] {1, 3}).equals(new TwoIndex(1, 2)));
	}

	@Test
	public void equalsArray4() {
		assertFalse(new ArrayIndex(new int[] {2, 3}).equals(new TwoIndex(1, 2)));
	}

	@Test
	public void equalsArray5() {
		assertFalse(new ArrayIndex(new int[] {0, 2}).equals(new TwoIndex(1, 2)));
	}

	@Test
	public void equalsArray6() {
		assertFalse(new ArrayIndex(new int[] {0, 2}).equals(new RangeIndex(1, 2)));
	}

	@Test
	public void equalsArray7() {
		assertFalse(new ArrayIndex(new int[] {0, 2}).equals(new RangeIndex(1, 3)));
	}

	@Test
	public void equalsArray8() {
		assertTrue(new ArrayIndex(new int[] {1, 2}).equals(new RangeIndex(1, 3)));
	}

	@Test
	public void equalsArray9() {
		assertTrue(new ArrayIndex(new int[] {1, 2, 3}).equals(new RangeIndex(1, 4)));
	}

	@Test
	public void equalsArray10() {
		assertFalse(new ArrayIndex(new int[] {1, 2, 4}).equals(new RangeIndex(1, 4)));
	}

	@Test
	public void equalsArray11() {
		assertFalse(new ArrayIndex(new int[] {0, 2, 4}).equals(new RangeIndex(1, 4)));
	}

	@Test
	public void equalsArray12() {
		assertFalse(new ArrayIndex(new int[] {0, 2, 3}).equals(new RangeIndex(1, 4)));
	}

	@Test
	public void notEqualsRange() {
		assertFalse(new RangeIndex(1, 2).equals((Object) new SingleIndex(2)));
	}

	@Test
	public void notEqualsRange2() {
		assertFalse(new RangeIndex(1, 2).equals(new RangeIndex(0, 2)));
	}

	@Test
	public void notEqualsRange3() {
		assertFalse(new RangeIndex(1, 2).equals(new RangeIndex(1, 3)));
	}

	@Test
	public void notEqualsRange4() {
		assertTrue(new RangeIndex(1, 2).equals(new RangeIndex(1, 2)));
	}

	@Test
	public void notEqualsRange5() {
		assertTrue(new RangeIndex(1, 2).equals(new SingleIndex(1)));
	}

	@Test
	public void notEqualsRange6() {
		assertTrue(new RangeIndex(1, 2).equals(new ArrayIndex(new int[] {1})));
	}

	@Test
	public void notEqualsTwo() {
		assertFalse(new TwoIndex(1, 2).equals((Object) new SingleIndex(2)));
	}

	@Test
	public void equalsTwo2() {
		assertTrue(new TwoIndex(1, 2).equals((Object) new ArrayIndex(new int[] {1, 2})));
	}

	@Test
	public void equalsTwo3() {
		assertFalse(new TwoIndex(1, 2).equals((Object) new ArrayIndex(new int[] {1, 3})));
	}

	@Test
	public void equalsTwo4() {
		assertFalse(new TwoIndex(1, 2).equals((Object) new ArrayIndex(new int[] {0, 2})));
	}

	@Test
	public void equalsTwo5() {
		assertFalse(new TwoIndex(1, 2).equals((Object) new ArrayIndex(new int[] {0})));
	}

	@Test
	public void equalsTwo6() {
		assertFalse(new TwoIndex(1, 2).equals((Object) new ArrayIndex(new int[] {0, 1, 2})));
	}

	@Test
	public void notEqualsSingle() {
		assertFalse(new SingleIndex(2).equals((Object) new TwoIndex(1, 12)));
	}

	@Test
	public void hashCode1() {
		assertTrue(new TwoIndex(1, 2).hashCode() == new ArrayIndex(new int[] {1, 2}).hashCode());
	}

	@Test
	public void hashCode2() {
		assertFalse(new TwoIndex(1, 3).hashCode() == new ArrayIndex(new int[] {1, 2}).hashCode());
	}

	@Test
	public void hashCode3() {
		assertFalse(new TwoIndex(1, 3).hashCode() == new ArrayIndex(new int[] {1, 4}).hashCode());
	}

	@Test
	public void hashCode4() {
		assertFalse(new TwoIndex(1, 2).hashCode() == new ArrayIndex(new int[] {1, 2, 3}).hashCode());
	}

	@Test
	public void hashCode5() {
		assertTrue(new SingleIndex(1).hashCode() == new ArrayIndex(new int[] {1}).hashCode());
	}

	@Test
	public void hashCode6() {
		assertTrue(new SingleIndex(4).hashCode() == new ArrayIndex(new int[] {4}).hashCode());
	}

	@Test
	public void hashCode7() {
		assertTrue(new SingleIndex(1324).hashCode() == new ArrayIndex(new int[] {1324}).hashCode());
	}

	@Test
	public void hashCode8() {
		assertTrue(
			new RangeIndex(0, 10).hashCode() != new ArrayIndex(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}).hashCode());
	}

	@Test
	public void hashCode9() {
		assertTrue(new RangeIndex(0, 4).hashCode() != new ArrayIndex(new int[] {0, 1, 2, 3,}).hashCode());
	}

	@Test
	public void hashCode10() {
		assertTrue(
			new RangeIndex(5555, 5560).hashCode() != new ArrayIndex(new int[] {5555, 5556, 5557, 5558, 5559}).hashCode());
	}

	@Test
	public void hashCode11() {
		assertTrue(new RangeIndex(5000000, 5000005)
			.hashCode() != new ArrayIndex(new int[] {5000000, 5000001, 5000002, 5000003, 5000004}).hashCode());
	}

	private static Object notRelated() {
		return Integer.valueOf(2);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidCreate1() {
		ColIndexFactory.create(new int[0]);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidCreate2() {
		ColIndexFactory.create(new IntArrayList());
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidCreate3() {
		ColIndexFactory.create(new IntArrayList(0));
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidCreate3Add() {
		ColIndexFactory.create(new IntArrayList(10));
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidCreate4() {
		ColIndexFactory.create(1, 1);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidCreate5() {
		ColIndexFactory.create(10000, 10000);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidCreate6() {
		ColIndexFactory.create(10000, 0);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidCreate7() {
		ColIndexFactory.create(1, 0);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidCreate8() {
		ColIndexFactory.create(0);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidCreate9() {
		ColIndexFactory.create(-10);
	}

	@Test(expected = DMLCompressionException.class)
	public void invalidRange() {
		new RangeIndex(10, 4);
	}

	@Test(expected = DMLCompressionException.class)
	public void invalidRange2() {
		new RangeIndex(10, 10);
	}

	@Test(expected = DMLCompressionException.class)
	public void invalidRange3() {
		ColIndexFactory.createI(0, -1, 2);
	}

	@Test(expected = DMLCompressionException.class)
	public void invalidRange4() {
		ColIndexFactory.createI(0, 0, 2);
	}

	@Test(expected = DMLCompressionException.class)
	public void invalidRange5() {
		ColIndexFactory.createI(0, 1, 1);
	}
}
