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

public class DenseBlockSetRowTest {
	@Test
	public void testDenseBlock2FP32Row() {
		DenseBlock db = getDenseBlock2(ValueType.FP32);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlock2FP64Row() {
		DenseBlock db = getDenseBlock2(ValueType.FP64);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlock2BoolRow() {
		DenseBlock db = getDenseBlock2(ValueType.BOOLEAN);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlock2TrueBoolRow() {
		DenseBlock db = new DenseBlockBoolArray(new int[] {3,5});
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlock2Int32Row() {
		DenseBlock db = getDenseBlock2(ValueType.INT32);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlock2Int64Row() {
		DenseBlock db = getDenseBlock2(ValueType.INT64);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlock2StringRow() {
		DenseBlock db = getDenseBlock2(ValueType.STRING);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlockLarge2FP32Row() {
		DenseBlock db = getDenseBlockLarge2(ValueType.FP32);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlockLarge2FP64Row() {
		DenseBlock db = getDenseBlockLarge2(ValueType.FP64);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlockLarge2BoolRow() {
		DenseBlock db = getDenseBlockLarge2(ValueType.BOOLEAN);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlockLarge2Int32Row() {
		DenseBlock db = getDenseBlockLarge2(ValueType.INT32);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlockLarge2Int64Row() {
		DenseBlock db = getDenseBlockLarge2(ValueType.INT64);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlockLarge2StringRow() {
		DenseBlock db = getDenseBlockLarge2(ValueType.STRING);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlock3FP32Row() {
		DenseBlock db = getDenseBlock3(ValueType.FP32);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlock3FP64Row() {
		DenseBlock db = getDenseBlock3(ValueType.FP64);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlock3BoolRow() {
		DenseBlock db = getDenseBlock3(ValueType.BOOLEAN);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlock3TrueBoolRow() {
		DenseBlock db = new DenseBlockBoolArray( new int[] {3,5,7});
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlock3Int32Row() {
		DenseBlock db = getDenseBlock3(ValueType.INT32);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlock3Int64Row() {
		DenseBlock db = getDenseBlock3(ValueType.INT64);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlock3StringRow() {
		DenseBlock db = getDenseBlock3(ValueType.STRING);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlockLarge3FP32Row() {
		DenseBlock db = getDenseBlockLarge3(ValueType.FP32);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlockLarge3FP64Row() {
		DenseBlock db = getDenseBlockLarge3(ValueType.FP64);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlockLarge3BoolRow() {
		DenseBlock db = getDenseBlockLarge3(ValueType.BOOLEAN);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlockLarge3Int32Row() {
		DenseBlock db = getDenseBlockLarge3(ValueType.INT32);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlockLarge3Int64Row() {
		DenseBlock db = getDenseBlockLarge3(ValueType.INT64);
		checkRow(setRow(db));
	}

	@Test
	public void testDenseBlockLarge3StringRow() {
		DenseBlock db = getDenseBlockLarge3(ValueType.STRING);
		checkRow(setRow(db));
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
			case BOOLEAN:
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
			case BOOLEAN:
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

	private static DenseBlock setRow(DenseBlock db) {
		if (db.numDims() == 3) {
			int dim12 = 5 * 7;
			double[] row = new double[dim12];
			for (int i = 0; i < dim12; i++) {
				row[i] = i;
			}
			db.set(1, row);
		} else { //num dims = 2
			int dim1 = 5;
			double[] row = new double[dim1];
			for (int i = 0; i < dim1; i++) {
				row[i] = i;
			}
			db.set(1, row);
		}
		return db;
	}

	private static void checkRow(DenseBlock db) {
		boolean isBool = (db instanceof DenseBlockBoolBitset) || (db instanceof DenseBlockLBoolBitset) || (db instanceof DenseBlockBoolArray);
		if (db.numDims() == 3) {
			int dim1 = 5, dim2 = 7;
			for (int i = 0; i < dim1; i++)
				for (int j = 0; j < dim2; j++) {
					int pos = i * dim2 + j;
					double expected = isBool && pos != 0 ? 1 : (double) pos;
					Assert.assertEquals(expected, db.get(1, pos), 0);
				}
		} else { //num dims = 2
			int dim1 = 5;
			for (int i = 0; i < dim1; i++) {
				double expected = isBool && i != 0 ? 1 : (double) i;
				Assert.assertEquals(expected, db.get(1, i), 0);
			}
		}
	}
}
