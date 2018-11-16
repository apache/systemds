/*
 * Copyright 2018 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.test.tensor;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.data.DenseBlock;
import org.tugraz.sysds.runtime.data.DenseBlockFactory;


public class DenseBlockConstructionTest 
{
	@Test
	public void testMetaDenseBlock2FP32() throws Exception {
		DenseBlock db = getDenseBlock2(ValueType.FP32);
		Assert.assertEquals(3, db.numRows());
		Assert.assertEquals(true, db.isNumeric());
		Assert.assertEquals(true, db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertEquals(3*5, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}
	
	@Test
	public void testMetaDenseBlock2FP64() throws Exception {
		DenseBlock db = getDenseBlock2(ValueType.FP64);
		Assert.assertEquals(3, db.numRows());
		Assert.assertEquals(true, db.isNumeric());
		Assert.assertEquals(true, db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertEquals(3*5, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}
	
	@Test
	public void testMetaDenseBlock2Bool() throws Exception {
		DenseBlock db = getDenseBlock2(ValueType.BOOLEAN);
		Assert.assertEquals(3, db.numRows());
		Assert.assertEquals(true, db.isNumeric());
		Assert.assertEquals(true, db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5, db.size());
		Assert.assertTrue(3*5 <= db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}
	
	@Test
	public void testMetaDenseBlock3FP32() throws Exception {
		DenseBlock db = getDenseBlock3(ValueType.FP32);
		Assert.assertEquals(3, db.numRows());
		Assert.assertEquals(true, db.isNumeric());
		Assert.assertEquals(true, db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertEquals(3*5*7, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}
	
	@Test
	public void testMetaDenseBlock3FP64() throws Exception {
		DenseBlock db = getDenseBlock3(ValueType.FP64);
		Assert.assertEquals(3, db.numRows());
		Assert.assertEquals(true, db.isNumeric());
		Assert.assertEquals(true, db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertEquals(3*5*7, db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}
	
	@Test
	public void testMetaDenseBlock3Bool() throws Exception {
		DenseBlock db = getDenseBlock3(ValueType.BOOLEAN);
		Assert.assertEquals(3, db.numRows());
		Assert.assertEquals(true, db.isNumeric());
		Assert.assertEquals(true, db.isContiguous());
		Assert.assertEquals(1, db.numBlocks());
		Assert.assertEquals(3, db.blockSize());
		Assert.assertEquals(3*5*7, db.size());
		Assert.assertTrue(3*5*7 <= db.capacity());
		Assert.assertEquals(0, db.countNonZeros());
		Assert.assertEquals(DenseBlock.Type.DRB,
			DenseBlockFactory.getDenseBlockType(db));
	}
	
	private DenseBlock getDenseBlock2(ValueType vt) {
		return DenseBlockFactory.createDenseBlock(vt, new int[] {3,5});
	}
	
	private DenseBlock getDenseBlock3(ValueType vt) {
		return DenseBlockFactory.createDenseBlock(vt, new int[] {3,5,7});
	}
}
