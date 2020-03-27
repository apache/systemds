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
import org.tugraz.sysds.runtime.data.DataTensorBlock;
import org.tugraz.sysds.runtime.data.BasicTensorBlock;
import org.tugraz.sysds.runtime.data.TensorBlock;

import java.util.Arrays;


public class TensorConstructionTest {
	@Test
	public void testMetaDefaultBasicTensor() {
		BasicTensorBlock tb = new BasicTensorBlock();
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(0, tb.getNumColumns());
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
	}

	@Test
	public void testMetaValueBasicTensor() {
		BasicTensorBlock tb = new BasicTensorBlock(7.3);
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(1, tb.getNumRows());
		Assert.assertEquals(1, tb.getNumColumns());
		Assert.assertEquals(1, tb.getNonZeros());
		Assert.assertFalse(tb.isSparse());
	}

	@Test
	public void testMetaTypedBasicTensor() {
		BasicTensorBlock tb = new BasicTensorBlock(ValueType.INT64, new int[]{11, 12, 13});
		Assert.assertEquals(ValueType.INT64, tb.getValueType());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
	}

	@Test
	public void testMetaTypedBasicTensor2() {
		BasicTensorBlock tb = new BasicTensorBlock(ValueType.INT64, new int[]{11, 12, 13}, false);
		Assert.assertEquals(ValueType.INT64, tb.getValueType());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertFalse(tb.isSparse());
	}

	@Test
	public void testMetaTypedBasicTensor3() {
		BasicTensorBlock tb = new BasicTensorBlock(ValueType.BOOLEAN, new int[]{11, 12}, true);
		Assert.assertEquals(ValueType.BOOLEAN, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(12, tb.getDim(1));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
	}

	@Test
	public void testMetaCopyDefaultBasicTensor() {
		BasicTensorBlock tb = new BasicTensorBlock(new BasicTensorBlock());
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(0, tb.getNumColumns());
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
	}

	@Test
	public void testMetaCopyValueBasicTensor() {
		BasicTensorBlock tb = new BasicTensorBlock(new BasicTensorBlock(7.3));
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(1, tb.getNumRows());
		Assert.assertEquals(1, tb.getNumColumns());
		Assert.assertEquals(1, tb.getNonZeros());
		Assert.assertFalse(tb.isSparse());
	}

	@Test
	public void testMetaCopyTypedBasicTensor() {
		BasicTensorBlock tb = new BasicTensorBlock(new BasicTensorBlock(ValueType.INT64, new int[]{11, 12, 13}));
		Assert.assertEquals(ValueType.INT64, tb.getValueType());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
	}

	@Test
	public void testMetaCopyTypedBasicTensor2() {
		BasicTensorBlock tb = new BasicTensorBlock(new BasicTensorBlock(ValueType.INT64, new int[]{11, 12, 13}, false));
		Assert.assertEquals(ValueType.INT64, tb.getValueType());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertFalse(tb.isSparse());
	}

	@Test
	public void testMetaCopyTypedBasicTensor3() {
		BasicTensorBlock tb = new BasicTensorBlock(new BasicTensorBlock(ValueType.BOOLEAN, new int[]{11, 12}, true));
		Assert.assertEquals(ValueType.BOOLEAN, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(12, tb.getDim(1));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
	}

	@Test
	public void testMetaDefaultDataTensor() {
		DataTensorBlock tb = new DataTensorBlock();
		Assert.assertArrayEquals(new ValueType[0], tb.getSchema());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(0, tb.getNumColumns());
	}

	@Test
	public void testMetaValueDataTensor() {
		DataTensorBlock tb = new DataTensorBlock(7.3);
		Assert.assertArrayEquals(new ValueType[]{ValueType.FP64}, tb.getSchema());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(1, tb.getNumRows());
		Assert.assertEquals(1, tb.getNumColumns());
	}

	@Test
	public void testMetaTypedDataTensor() {
		DataTensorBlock tb = new DataTensorBlock(ValueType.INT64, new int[]{11, 12, 13});
		ValueType[] schema = new ValueType[12];
		Arrays.fill(schema, ValueType.INT64);
		Assert.assertArrayEquals(schema, tb.getSchema());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
	}

	@Test
	public void testMetaSchemaTypedDataTensor() {
		ValueType[] schema = new ValueType[] {ValueType.BOOLEAN, ValueType.INT32, ValueType.STRING};
		DataTensorBlock tb = new DataTensorBlock(schema);
		Assert.assertArrayEquals(schema, tb.getSchema());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(schema.length, tb.getNumColumns());
	}

	@Test
	public void testMetaNColsTypedDataTensor() {
		DataTensorBlock tb = new DataTensorBlock(10, ValueType.BOOLEAN);
		ValueType[] schema = new ValueType[10];
		Arrays.fill(schema, ValueType.BOOLEAN);
		Assert.assertArrayEquals(schema, tb.getSchema());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(10, tb.getNumColumns());
	}

	@Test
	public void testMetaCopyDefaultDataTensor() {
		DataTensorBlock tb = new DataTensorBlock(new DataTensorBlock());
		Assert.assertArrayEquals(new ValueType[0], tb.getSchema());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(0, tb.getNumColumns());
	}

	@Test
	public void testMetaCopyValueDataTensor() {
		DataTensorBlock tb = new DataTensorBlock(new DataTensorBlock(7.3));
		Assert.assertArrayEquals(new ValueType[]{ValueType.FP64}, tb.getSchema());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(1, tb.getNumRows());
		Assert.assertEquals(1, tb.getNumColumns());
	}

	@Test
	public void testMetaCopyTypedDataTensor() {
		ValueType[] schema = {ValueType.INT32, ValueType.INT64, ValueType.BOOLEAN};
		DataTensorBlock tb = new DataTensorBlock(new DataTensorBlock(schema, new int[]{11, 3, 13}));
		Assert.assertArrayEquals(schema, tb.getSchema());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(3, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
	}

	@Test
	public void testMetaCopyTypedDataTensor2() {
		ValueType[] schema = {ValueType.FP64, ValueType.FP32, ValueType.STRING};
		DataTensorBlock tb = new DataTensorBlock(new DataTensorBlock(schema, new int[]{11, 3, 13}));
		Assert.assertArrayEquals(schema, tb.getSchema());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(3, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
	}

	@Test
	public void testMetaCopyTypedDataTensor3() {
		ValueType[] schema = {ValueType.FP64, ValueType.FP32, ValueType.INT64, ValueType.INT32, ValueType.BOOLEAN,
				ValueType.STRING};
		DataTensorBlock tb = new DataTensorBlock(new DataTensorBlock(schema, new int[]{2, schema.length, 2}, new String[][]{
				new String[]{"1.4", "-5.34", "4.5", "-100000.1"},
				new String[]{"1.4", "-5.34", "4.5", "-100000.1"},
				new String[]{"1", "-5", "4", "-100000"},
				new String[]{"1", "-5", "4", "-100000"},
				new String[]{"TRUE", "FALSE", "FALSE", "TRUE"},
				new String[]{"hello", "bye", "me", "foobar"},
		}));
		Assert.assertArrayEquals(schema, tb.getSchema());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(2, tb.getNumRows());
		Assert.assertEquals(schema.length, tb.getNumColumns());
		Assert.assertEquals(2, tb.getDim(2));
	}

	@Test
	public void testMetaDefaultTensorBlock() {
		TensorBlock tb = new TensorBlock();
		Assert.assertArrayEquals(null, tb.getSchema());
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertTrue(tb.isBasic());
		Assert.assertNull(tb.getDataTensor());
		Assert.assertFalse(tb.isAllocated());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(0, tb.getNumColumns());
		Assert.assertFalse(tb.isMatrix());
		Assert.assertFalse(tb.isVector());
	}

	@Test
	public void testMetaHeterogeneousTensorBlock() {
		TensorBlock tb = new TensorBlock(new int[]{1, 1}, false);
		Assert.assertArrayEquals(new ValueType[]{ValueType.FP64}, tb.getSchema());
		Assert.assertNull(tb.getValueType());
		Assert.assertFalse(tb.isBasic());
		Assert.assertNull(tb.getBasicTensor());
		Assert.assertNull(tb.getDataTensor());
		Assert.assertFalse(tb.isAllocated());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(1, tb.getNumRows());
		Assert.assertEquals(1, tb.getNumColumns());
		Assert.assertFalse(tb.isMatrix());
		Assert.assertTrue(tb.isVector());
	}

	@Test
	public void testMetaHomogeneousTensorBlock() {
		TensorBlock tb = new TensorBlock(new int[]{1, 1}, true);
		Assert.assertNull(tb.getSchema());
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertTrue(tb.isBasic());
		Assert.assertNull(tb.getBasicTensor());
		Assert.assertNull(tb.getDataTensor());
		Assert.assertFalse(tb.isAllocated());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(1, tb.getNumRows());
		Assert.assertEquals(1, tb.getNumColumns());
		Assert.assertFalse(tb.isMatrix());
		Assert.assertTrue(tb.isVector());
	}

	@Test
	public void testMetaValueTensorBlock() {
		double val = 7.3;
		TensorBlock tb = new TensorBlock(val);
		Assert.assertArrayEquals(null, tb.getSchema());
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertTrue(tb.isBasic());
		Assert.assertNull(tb.getDataTensor());
		Assert.assertTrue(tb.isAllocated());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(1, tb.getNumRows());
		Assert.assertEquals(1, tb.getNumColumns());
		Assert.assertFalse(tb.isMatrix());
		Assert.assertTrue(tb.isVector());
		Assert.assertEquals(val, tb.get(0, 0), 0);
	}

	@Test
	public void testMetaTypedTensorBlock() {
		TensorBlock tb = new TensorBlock(ValueType.INT64, new int[]{11, 12, 13});
		Assert.assertArrayEquals(null, tb.getSchema());
		Assert.assertEquals(ValueType.INT64, tb.getValueType());
		Assert.assertTrue(tb.isBasic());
		Assert.assertNull(tb.getDataTensor());
		Assert.assertTrue(tb.isAllocated());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertFalse(tb.isMatrix());
		Assert.assertFalse(tb.isVector());
	}

	@Test
	public void testMetaSchemaTypedTensorBlock() {
		ValueType[] schema = new ValueType[] {ValueType.BOOLEAN, ValueType.INT32, ValueType.STRING};
		TensorBlock tb = new TensorBlock(schema, new int[]{4, 3, 2});
		Assert.assertArrayEquals(schema, tb.getSchema());
		Assert.assertNull(tb.getValueType());
		Assert.assertFalse(tb.isBasic());
		Assert.assertNull(tb.getBasicTensor());
		Assert.assertTrue(tb.isAllocated());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(4, tb.getNumRows());
		Assert.assertEquals(3, tb.getNumColumns());
		Assert.assertEquals(2, tb.getDim(2));
		Assert.assertFalse(tb.isMatrix());
		Assert.assertFalse(tb.isVector());
	}

	@Test
	public void testMetaCopyDefaultTensorBlock() {
		TensorBlock tb = new TensorBlock(new TensorBlock());
		Assert.assertArrayEquals(null, tb.getSchema());
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertTrue(tb.isBasic());
		Assert.assertNull(tb.getDataTensor());
		Assert.assertFalse(tb.isAllocated());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(0, tb.getNumColumns());
		Assert.assertFalse(tb.isMatrix());
		Assert.assertFalse(tb.isVector());
	}

	@Test
	public void testMetaCopyValueTensorBlock() {
		double val = 7.3;
		TensorBlock tb = new TensorBlock(new TensorBlock(val));
		Assert.assertArrayEquals(null, tb.getSchema());
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertTrue(tb.isBasic());
		Assert.assertNull(tb.getDataTensor());
		Assert.assertTrue(tb.isAllocated());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(1, tb.getNumRows());
		Assert.assertEquals(1, tb.getNumColumns());
		Assert.assertFalse(tb.isMatrix());
		Assert.assertTrue(tb.isVector());
		Assert.assertEquals(val, tb.get(0, 0), 0);
	}

	@Test
	public void testMetaCopyTypedTensorBlock() {
		TensorBlock tb = new TensorBlock(new TensorBlock(ValueType.INT64, new int[]{11, 12, 13}));
		Assert.assertArrayEquals(null, tb.getSchema());
		Assert.assertEquals(ValueType.INT64, tb.getValueType());
		Assert.assertTrue(tb.isBasic());
		Assert.assertNull(tb.getDataTensor());
		Assert.assertTrue(tb.isAllocated());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertFalse(tb.isMatrix());
		Assert.assertFalse(tb.isVector());
	}

	@Test
	public void testMetaCopySchemaTypedTensorBlock() {
		ValueType[] schema = new ValueType[] {ValueType.BOOLEAN, ValueType.INT32, ValueType.STRING};
		TensorBlock tb = new TensorBlock(new TensorBlock(schema, new int[]{4, 3, 2}));
		Assert.assertArrayEquals(schema, tb.getSchema());
		Assert.assertNull(tb.getValueType());
		Assert.assertFalse(tb.isBasic());
		Assert.assertNull(tb.getBasicTensor());
		Assert.assertTrue(tb.isAllocated());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(4, tb.getNumRows());
		Assert.assertEquals(3, tb.getNumColumns());
		Assert.assertEquals(2, tb.getDim(2));
		Assert.assertFalse(tb.isMatrix());
		Assert.assertFalse(tb.isVector());
	}
}
