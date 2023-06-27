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


public class DenseBlockConstructionTest 
{
	@Test
	public void testMetaDenseBlock2FP32() {
		DenseBlock db = getDenseBlock2(ValueType.FP32);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertEquals(3*5, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}
	
	@Test
	public void testMetaDenseBlock2FP64() {
		DenseBlock db = getDenseBlock2(ValueType.FP64);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertEquals(3*5, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}
	
	@Test
	public void testMetaDenseBlock2Bool() {
		DenseBlock db = getDenseBlock2(ValueType.BITSET);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertTrue(3*5 <= db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlock2TrueBool() {
		DenseBlock db = getDenseBlock2(ValueType.BOOLEAN);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertTrue(3*5 <= db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
				DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlock2Int32() {
		DenseBlock db = getDenseBlock2(ValueType.INT32);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertEquals(3*5, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}
	
	@Test
	public void testMetaDenseBlock2Int64() {
		DenseBlock db = getDenseBlock2(ValueType.INT64);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertEquals(3*5, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlock2String() {
		DenseBlock db = getDenseBlock2(ValueType.STRING);
		Assert.assertEquals(3, db.numRows());
		Assert.assertFalse(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertEquals(3*5, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
				DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlockLarge2FP32() {
		DenseBlock db = getDenseBlockLarge2(ValueType.FP32);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertEquals(3*5, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.LDRB,
				DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlockLarge2FP64() {
		DenseBlock db = getDenseBlockLarge2(ValueType.FP64);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertEquals(3*5, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.LDRB,
				DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlockLarge2Bool() {
		DenseBlock db = getDenseBlockLarge2(ValueType.BITSET);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertEquals(64, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.LDRB,
				DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlockLarge2Int32() {
		DenseBlock db = getDenseBlockLarge2(ValueType.INT32);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertEquals(3*5, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.LDRB,
				DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlockLarge2Int64() {
		DenseBlock db = getDenseBlockLarge2(ValueType.INT64);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertEquals(3*5, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.LDRB,
				DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlockLarge2String() {
		DenseBlock db = getDenseBlockLarge2(ValueType.STRING);
		Assert.assertEquals(3, db.numRows());
		Assert.assertFalse(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertEquals(3*5, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.LDRB,
				DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlock3FP32() {
		DenseBlock db = getDenseBlock3(ValueType.FP32);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertEquals(3*5*7, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}
	
	@Test
	public void testMetaDenseBlock3FP64() {
		DenseBlock db = getDenseBlock3(ValueType.FP64);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertEquals(3*5*7, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}
	
	@Test
	public void testMetaDenseBlock3Bool() {
		DenseBlock db = getDenseBlock3(ValueType.BITSET);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertTrue(3*5*7 <= db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlock3TrueBool() {
		DenseBlock db = getDenseBlock3(ValueType.BOOLEAN);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertTrue(3*5*7 <= db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
				DenseBlockFactory.getDenseBlockType(db));
	}
	
	@Test
	public void testMetaDenseBlock3Int32() {
		DenseBlock db = getDenseBlock3(ValueType.INT32);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertEquals(3*5*7, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}
	
	@Test
	public void testMetaDenseBlock3Int64() {
		DenseBlock db = getDenseBlock3(ValueType.INT64);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertEquals(3*5*7, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlock3String() {
		DenseBlock db = getDenseBlock3(ValueType.STRING);
		Assert.assertEquals(3, db.numRows());
		Assert.assertFalse(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertEquals(3*5*7, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
				DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlockLarge3FP32() {
		DenseBlock db = getDenseBlockLarge3(ValueType.FP32);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertEquals(3*5*7, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.LDRB,
				DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlockLarge3FP64() {
		DenseBlock db = getDenseBlockLarge3(ValueType.FP64);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertEquals(3*5*7, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.LDRB,
				DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlockLarge3Bool() {
		DenseBlock db = getDenseBlockLarge3(ValueType.BITSET);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertEquals(128, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.LDRB,
				DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlockLarge3Int32() {
		DenseBlock db = getDenseBlockLarge3(ValueType.INT32);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertEquals(3*5*7, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.LDRB,
				DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlockLarge3Int64() {
		DenseBlock db = getDenseBlockLarge3(ValueType.INT64);
		Assert.assertEquals(3, db.numRows());
		Assert.assertTrue(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertEquals(3*5*7, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.LDRB,
				DenseBlockFactory.getDenseBlockType(db));
	}

	@Test
	public void testMetaDenseBlockLarge3String() {
		DenseBlock db = getDenseBlockLarge3(ValueType.STRING);
		Assert.assertEquals(3, db.numRows());
		Assert.assertFalse(db.isNumeric());
		Assert.assertTrue(db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertEquals(3*5*7, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.LDRB,
				DenseBlockFactory.getDenseBlockType(db));
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
}
