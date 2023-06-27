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


public class DenseBlockSetDenseBlockTest
{
	@Test
	public void testDenseBlock2FP32SetDenseBlock() {
		DenseBlock db = getDenseBlock2(ValueType.FP32);
		DenseBlock dbSet = getDenseBlock2(ValueType.FP32);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlock2FP64SetDenseBlock() {
		DenseBlock db = getDenseBlock2(ValueType.FP64);
		DenseBlock dbSet = getDenseBlock2(ValueType.FP64);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlock2BoolSetDenseBlock() {
		DenseBlock db = getDenseBlock2(ValueType.BITSET);
		DenseBlock dbSet = getDenseBlock2(ValueType.BITSET);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlock2TrueBoolSetDenseBlock() {
		DenseBlock db = getDenseBlock2(ValueType.BOOLEAN);
		DenseBlock dbSet = getDenseBlock2(ValueType.BOOLEAN);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlock2Int32SetDenseBlock() {
		DenseBlock db = getDenseBlock2(ValueType.INT32);
		DenseBlock dbSet = getDenseBlock2(ValueType.INT32);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlock2Int64SetDenseBlock() {
		DenseBlock db = getDenseBlock2(ValueType.INT64);
		DenseBlock dbSet = getDenseBlock2(ValueType.INT64);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlock2StringSetDenseBlock() {
		DenseBlock db = getDenseBlock2(ValueType.STRING);
		DenseBlock dbSet = getDenseBlock2(ValueType.STRING);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 5; j++) {
				dbSet.set(new int[]{i,j}, "test");
			}
		}
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlockLarge2FP32SetDenseBlock() {
		DenseBlock db = getDenseBlockLarge2(ValueType.FP32);
		DenseBlock dbSet = getDenseBlockLarge2(ValueType.FP32);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlockLarge2FP64SetDenseBlock() {
		DenseBlock db = getDenseBlockLarge2(ValueType.FP64);
		DenseBlock dbSet = getDenseBlockLarge2(ValueType.FP64);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlockLarge2BoolSetDenseBlock() {
		DenseBlock db = getDenseBlockLarge2(ValueType.BITSET);
		DenseBlock dbSet = getDenseBlockLarge2(ValueType.BITSET);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlockLarge2Int32SetDenseBlock() {
		DenseBlock db = getDenseBlockLarge2(ValueType.INT32);
		DenseBlock dbSet = getDenseBlockLarge2(ValueType.INT32);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlockLarge2Int64SetDenseBlock() {
		DenseBlock db = getDenseBlockLarge2(ValueType.INT64);
		DenseBlock dbSet = getDenseBlockLarge2(ValueType.INT64);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlockLarge2StringSetDenseBlock() {
		DenseBlock db = getDenseBlockLarge2(ValueType.STRING);
		DenseBlock dbSet = getDenseBlockLarge2(ValueType.STRING);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlock3FP32SetDenseBlock() {
		DenseBlock db = getDenseBlock3(ValueType.FP32);
		DenseBlock dbSet = getDenseBlock3(ValueType.FP32);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlock3FP64SetDenseBlock() {
		DenseBlock db = getDenseBlock3(ValueType.FP64);
		DenseBlock dbSet = getDenseBlock3(ValueType.FP64);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlock3BoolSetDenseBlock() {
		DenseBlock db = getDenseBlock3(ValueType.BITSET);
		DenseBlock dbSet = getDenseBlock3(ValueType.BITSET);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlock3TrueBoolSetDenseBlock() {
		DenseBlock db = getDenseBlock3(ValueType.BOOLEAN);
		DenseBlock dbSet = getDenseBlock3(ValueType.BOOLEAN);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlock3Int32SetDenseBlock() {
		DenseBlock db = getDenseBlock3(ValueType.INT32);
		DenseBlock dbSet = getDenseBlock3(ValueType.INT32);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlock3Int64SetDenseBlock() {
		DenseBlock db = getDenseBlock3(ValueType.INT64);
		DenseBlock dbSet = getDenseBlock3(ValueType.INT64);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlock3StringSetDenseBlock() {
		DenseBlock db = getDenseBlock3(ValueType.STRING);
		DenseBlock dbSet = getDenseBlock3(ValueType.STRING);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlockLarge3FP32SetDenseBlock() {
		DenseBlock db = getDenseBlockLarge3(ValueType.FP32);
		DenseBlock dbSet = getDenseBlockLarge3(ValueType.FP32);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlockLarge3FP64SetDenseBlock() {
		DenseBlock db = getDenseBlockLarge3(ValueType.FP64);
		DenseBlock dbSet = getDenseBlockLarge3(ValueType.FP64);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlockLarge3BoolSetDenseBlock() {
		DenseBlock db = getDenseBlockLarge3(ValueType.BITSET);
		DenseBlock dbSet = getDenseBlockLarge3(ValueType.BITSET);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlockLarge3Int32SetDenseBlock() {
		DenseBlock db = getDenseBlockLarge3(ValueType.INT32);
		DenseBlock dbSet = getDenseBlockLarge3(ValueType.INT32);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlockLarge3Int64SetDenseBlock() {
		DenseBlock db = getDenseBlockLarge3(ValueType.INT64);
		DenseBlock dbSet = getDenseBlockLarge3(ValueType.INT64);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	@Test
	public void testDenseBlockLarge3StringSetDenseBlock() {
		DenseBlock db = getDenseBlockLarge3(ValueType.STRING);
		DenseBlock dbSet = getDenseBlockLarge3(ValueType.STRING);
		dbSet.set(1);
		db.set(dbSet);
		compareDenseBlocks(db, dbSet);
	}

	private static DenseBlock getDenseBlock2(ValueType vt) {
		return DenseBlockFactory.createDenseBlock(vt, new int[] {3,5});
	}
	
	private static DenseBlock getDenseBlock3(ValueType vt) {
		return DenseBlockFactory.createDenseBlock(vt, new int[] {3,5,7});
	}

	private static DenseBlock getDenseBlockLarge2(ValueType vt) {
		int[] dims = {3,5};
		switch (vt) {
			case FP32: return new DenseBlockLFP32(dims);
			case FP64: return new DenseBlockLFP64(dims);
			case BITSET: return new DenseBlockLBoolBitset(dims);
			case INT32: return new DenseBlockLInt32(dims);
			case INT64: return new DenseBlockLInt64(dims);
			case STRING: return new DenseBlockLString(dims);
			default: throw new NotImplementedException();
		}
	}

	private static DenseBlock getDenseBlockLarge3(ValueType vt) {
		int[] dims = {3,5,7};
		switch (vt) {
			case FP32: return new DenseBlockLFP32(dims);
			case FP64: return new DenseBlockLFP64(dims);
			case BITSET: return new DenseBlockLBoolBitset(dims);
			case INT32: return new DenseBlockLInt32(dims);
			case INT64: return new DenseBlockLInt64(dims);
			case STRING: return new DenseBlockLString(dims);
			default: throw new NotImplementedException();
		}
	}

	private static void compareDenseBlocks(DenseBlock left, DenseBlock right) {
		Assert.assertEquals(left.numDims(), right.numDims());
		for (long i = 0; i < left.size(); i++) {
			int[] index = new int[left.numDims()];
			for (int ix = 0; ix < left.numDims() - 1; ix++) {
				Assert.assertEquals(left.getDim(ix), right.getDim(ix));
				index[ix] = (int)((i % left.getDim(ix)) / right.getDim(ix + 1));
			}
			Assert.assertEquals(left.getDim(left.numDims() - 1), right.getDim(left.numDims() - 1));
			index[left.numDims() - 1] = (int)(i % left.getDim(left.numDims() - 1));
			if (left instanceof DenseBlockString || left instanceof DenseBlockLString) {
				Assert.assertEquals(left.getString(index), right.getString(index));
			} else {
				Assert.assertEquals(left.get(index), right.get(index), 0);
			}
		}
	}
}
