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

package org.tugraz.sysds.test.component.tensor;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.data.TensorBlock;


public class TensorConstructionTest 
{
	@Test
	public void testMetaDefaultTensor() {
		TensorBlock tb = new TensorBlock();
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(0, tb.getNumCols());
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}
	
	@Test
	public void testMetaValueTensor() {
		TensorBlock tb = new TensorBlock(7.3);
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(1, tb.getNumRows());
		Assert.assertEquals(1, tb.getNumCols());
		Assert.assertEquals(1, tb.getNonZeros());
		Assert.assertFalse(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}
	
	@Test
	public void testMetaTypedTensor() {
		TensorBlock tb = new TensorBlock(ValueType.INT64, new int[]{11,12,13});
		Assert.assertEquals(ValueType.INT64, tb.getValueType());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumCols());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}
	
	@Test
	public void testMetaTypedTensor2() {
		TensorBlock tb = new TensorBlock(ValueType.INT64, new int[]{11,12,13}, false);
		Assert.assertEquals(ValueType.INT64, tb.getValueType());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumCols());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertFalse(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}
	
	@Test
	public void testMetaTypedTensor3() {
		TensorBlock tb = new TensorBlock(ValueType.BOOLEAN, new int[]{11,12}, true);
		Assert.assertEquals(ValueType.BOOLEAN, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumCols());
		Assert.assertEquals(12, tb.getDim(1));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
		Assert.assertTrue(tb.isMatrix());
	}
	
	@Test
	public void testMetaCopyDefaultTensor() {
		TensorBlock tb = new TensorBlock(new TensorBlock());
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(0, tb.getNumCols());
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}
	
	@Test
	public void testMetaCopyValueTensor() {
		TensorBlock tb = new TensorBlock(new TensorBlock(7.3));
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(1, tb.getNumRows());
		Assert.assertEquals(1, tb.getNumCols());
		Assert.assertEquals(1, tb.getNonZeros());
		Assert.assertFalse(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}
	
	@Test
	public void testMetaCopyTypedTensor() {
		TensorBlock tb = new TensorBlock(new TensorBlock(ValueType.INT64, new int[]{11,12,13}));
		Assert.assertEquals(ValueType.INT64, tb.getValueType());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumCols());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}
	
	@Test
	public void testMetaCopyTypedTensor2() {
		TensorBlock tb = new TensorBlock(new TensorBlock(ValueType.INT64, new int[]{11,12,13}, false));
		Assert.assertEquals(ValueType.INT64, tb.getValueType());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumCols());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertFalse(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}
	
	@Test
	public void testMetaCopyTypedTensor3() {
		TensorBlock tb = new TensorBlock(new TensorBlock(ValueType.BOOLEAN, new int[]{11,12}, true));
		Assert.assertEquals(ValueType.BOOLEAN, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumCols());
		Assert.assertEquals(12, tb.getDim(1));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
		Assert.assertTrue(tb.isMatrix());
	}
}
