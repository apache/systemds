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

package org.apache.sysds.test.component.tensor;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.data.*;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.data.DenseBlockLBoolBitset;


public class DenseBlockCountNonZeroTest {
	@Test
	public void testIndexDenseBlock2FP32CountNonZero() {
		DenseBlock db = getDenseBlock2(ValueType.FP32);
		checkFullNnz2(db);
	}

	@Test
	public void testIndexDenseBlock2FP64CountNonZero() {
		DenseBlock db = getDenseBlock2(ValueType.FP64);
		checkFullNnz2(db);
	}

	@Test
	public void testIndexDenseBlock2BoolCountNonZero() {
		DenseBlock db = getDenseBlock2(ValueType.BITSET);
		checkFullNnz2(db);
	}

	@Test
	public void testIndexDenseBlock2TrueBoolCountNonZero() {
		DenseBlock db = getDenseBlock2(ValueType.BOOLEAN);
		checkFullNnz2(db);
	}

	@Test
	public void testIndexDenseBlock2Int32CountNonZero() {
		DenseBlock db = getDenseBlock2(ValueType.INT32);
		checkFullNnz2(db);
	}

	@Test
	public void testIndexDenseBlock2Int64CountNonZero() {
		DenseBlock db = getDenseBlock2(ValueType.INT64);
		checkFullNnz2(db);
	}

	@Test
	public void testIndexDenseBlock2StringCountNonZero() {
		DenseBlock db = getDenseBlock2(ValueType.STRING);
		checkFullNnz2(db);
	}

	@Test
	public void testIndexDenseBlockLarge2FP32CountNonZero() {
		DenseBlock db = getDenseBlockLarge2(ValueType.FP32);
		checkFullNnz2(db);
	}

	@Test
	public void testIndexDenseBlockLarge2FP64CountNonZero() {
		DenseBlock db = getDenseBlockLarge2(ValueType.FP64);
		checkFullNnz2(db);
	}

	@Test
	public void testIndexDenseBlockLarge2BoolCountNonZero() {
		DenseBlock db = getDenseBlockLarge2(ValueType.BITSET);
		checkFullNnz2(db);
	}

	@Test
	public void testIndexDenseBlockLarge2Int32CountNonZero() {
		DenseBlock db = getDenseBlockLarge2(ValueType.INT32);
		checkFullNnz2(db);
	}

	@Test
	public void testIndexDenseBlockLarge2Int64CountNonZero() {
		DenseBlock db = getDenseBlockLarge2(ValueType.INT64);
		checkFullNnz2(db);
	}

	@Test
	public void testIndexDenseBlockLarge2StringCountNonZero() {
		DenseBlock db = getDenseBlockLarge2(ValueType.STRING);
		checkFullNnz2(db);
	}

	@Test
	public void testIndexDenseBlock3FP32CountNonZero() {
		DenseBlock db = getDenseBlock3(ValueType.FP32);
		checkFullNnz3(db);
	}

	@Test
	public void testIndexDenseBlock3FP64CountNonZero() {
		DenseBlock db = getDenseBlock3(ValueType.FP64);
		checkFullNnz3(db);
	}

	@Test
	public void testIndexDenseBlock3BoolCountNonZero() {
		DenseBlock db = getDenseBlock3(ValueType.BITSET);
		checkFullNnz3(db);
	}

	@Test
	public void testIndexDenseBlock3TrueBoolCountNonZero() {
		DenseBlock db = getDenseBlock3(ValueType.BOOLEAN);
		checkFullNnz3(db);
	}

	@Test
	public void testIndexDenseBlock3Int32CountNonZero() {
		DenseBlock db = getDenseBlock3(ValueType.INT32);
		checkFullNnz3(db);
	}

	@Test
	public void testIndexDenseBlock3Int64CountNonZero() {
		DenseBlock db = getDenseBlock3(ValueType.INT64);
		checkFullNnz3(db);
	}

	@Test
	public void testIndexDenseBlock3StringCountNonZero() {
		DenseBlock db = getDenseBlock3(ValueType.STRING);
		checkFullNnz3(db);
	}

	@Test
	public void testIndexDenseBlockLarge3FP32CountNonZero() {
		DenseBlock db = getDenseBlockLarge3(ValueType.FP32);
		checkFullNnz3(db);
	}

	@Test
	public void testIndexDenseBlockLarge3FP64CountNonZero() {
		DenseBlock db = getDenseBlockLarge3(ValueType.FP64);
		checkFullNnz3(db);
	}

	@Test
	public void testIndexDenseBlockLarge3BoolCountNonZero() {
		DenseBlock db = getDenseBlockLarge3(ValueType.BITSET);
		checkFullNnz3(db);
	}

	@Test
	public void testIndexDenseBlockLarge3Int32CountNonZero() {
		DenseBlock db = getDenseBlockLarge3(ValueType.INT32);
		checkFullNnz3(db);
	}

	@Test
	public void testIndexDenseBlockLarge3Int64CountNonZero() {
		DenseBlock db = getDenseBlockLarge3(ValueType.INT64);
		checkFullNnz3(db);
	}

	@Test
	public void testIndexDenseBlockLarge3StringCountNonZero() {
		DenseBlock db = getDenseBlockLarge3(ValueType.STRING);
		checkFullNnz3(db);
	}

	private static DenseBlock getDenseBlock2(ValueType vt) {
		return DenseBlockFactory.createDenseBlock(vt, new int[]{3, 5});
	}

	private static DenseBlock getDenseBlock3(ValueType vt) {
		return DenseBlockFactory.createDenseBlock(vt, new int[]{3, 5, 7});
	}

	private static DenseBlock getDenseBlockLarge2(ValueType vt) {
		int[] dims = {3, 5};
		switch (vt) {
			case FP32:
				return new DenseBlockLFP32(dims);
			case FP64:
				return new DenseBlockLFP64(dims);
			case BITSET:
				return new DenseBlockLBoolBitset(dims);
			case INT32:
				return new DenseBlockLInt32(dims);
			case INT64:
				return new DenseBlockLInt64(dims);
			case STRING:
				return new DenseBlockLString(dims);
			default:
				throw new NotImplementedException();
		}
	}

	private static DenseBlock getDenseBlockLarge3(ValueType vt) {
		int[] dims = {3, 5, 7};
		switch (vt) {
			case FP32:
				return new DenseBlockLFP32(dims);
			case FP64:
				return new DenseBlockLFP64(dims);
			case BITSET:
				return new DenseBlockLBoolBitset(dims);
			case INT32:
				return new DenseBlockLInt32(dims);
			case INT64:
				return new DenseBlockLInt64(dims);
			case STRING:
				return new DenseBlockLString(dims);
			default:
				throw new NotImplementedException();
		}
	}

	private static void checkFullNnz2(DenseBlock db) {
		Assert.assertEquals(0, db.countNonZeros());
		for (int r = 0; r < 3; r++) {
			Assert.assertEquals(0, db.countNonZeros(r));
		}
		Assert.assertEquals(0, db.countNonZeros(0, 2, 3, 5));
		db.set(1, 3, 0, 3, 3);
		Assert.assertEquals((3 - 1) * 3, db.countNonZeros());
		Assert.assertEquals(0, db.countNonZeros(0));
		Assert.assertEquals(3, db.countNonZeros(1));
		Assert.assertEquals(3, db.countNonZeros(1));
		Assert.assertEquals(1, db.countNonZeros(0, 2, 2, 5));
		db.set(1);
		Assert.assertEquals(3 * 5, db.countNonZeros());
		for (int r = 0; r < 3; r++) {
			Assert.assertEquals(5, db.countNonZeros(r));
		}
		Assert.assertEquals(4, db.countNonZeros(0, 2, 3, 5));
	}

	private static void checkFullNnz3(DenseBlock db) {
		Assert.assertEquals(0, db.countNonZeros());
		for (int r = 0; r < 3; r++) {
			Assert.assertEquals(0, db.countNonZeros(r));
		}
		db.set(1);
		Assert.assertEquals(3 * 5 * 7, db.countNonZeros());
		for (int r = 0; r < 3; r++) {
			Assert.assertEquals(5 * 7, db.countNonZeros(r));
		}
	}
}
