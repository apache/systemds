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
import org.tugraz.sysds.runtime.data.HeterogTensor;
import org.tugraz.sysds.runtime.data.HomogTensor;

import java.util.Arrays;


public class TensorConstructionTest {
	@Test
	public void testMetaDefaultHomogTensor() {
		HomogTensor tb = new HomogTensor();
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(0, tb.getNumColumns());
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaValueHomogTensor() {
		HomogTensor tb = new HomogTensor(7.3);
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(1, tb.getNumRows());
		Assert.assertEquals(1, tb.getNumColumns());
		Assert.assertEquals(1, tb.getNonZeros());
		Assert.assertFalse(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaTypedHomogTensor() {
		HomogTensor tb = new HomogTensor(ValueType.INT64, new int[]{11, 12, 13});
		Assert.assertEquals(ValueType.INT64, tb.getValueType());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaTypedHomogTensor2() {
		HomogTensor tb = new HomogTensor(ValueType.INT64, new int[]{11, 12, 13}, false);
		Assert.assertEquals(ValueType.INT64, tb.getValueType());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertFalse(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaTypedHomogTensor3() {
		HomogTensor tb = new HomogTensor(ValueType.BOOLEAN, new int[]{11, 12}, true);
		Assert.assertEquals(ValueType.BOOLEAN, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(12, tb.getDim(1));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
		Assert.assertTrue(tb.isMatrix());
	}

	@Test
	public void testMetaCopyDefaultHomogTensor() {
		HomogTensor tb = new HomogTensor(new HomogTensor());
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(0, tb.getNumColumns());
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaCopyValueHomogTensor() {
		HomogTensor tb = new HomogTensor(new HomogTensor(7.3));
		Assert.assertEquals(ValueType.FP64, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(1, tb.getNumRows());
		Assert.assertEquals(1, tb.getNumColumns());
		Assert.assertEquals(1, tb.getNonZeros());
		Assert.assertFalse(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaCopyTypedHomogTensor() {
		HomogTensor tb = new HomogTensor(new HomogTensor(ValueType.INT64, new int[]{11, 12, 13}));
		Assert.assertEquals(ValueType.INT64, tb.getValueType());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaCopyTypedHomogTensor2() {
		HomogTensor tb = new HomogTensor(new HomogTensor(ValueType.INT64, new int[]{11, 12, 13}, false));
		Assert.assertEquals(ValueType.INT64, tb.getValueType());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertFalse(tb.isSparse());
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaCopyTypedHomogTensor3() {
		HomogTensor tb = new HomogTensor(new HomogTensor(ValueType.BOOLEAN, new int[]{11, 12}, true));
		Assert.assertEquals(ValueType.BOOLEAN, tb.getValueType());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(12, tb.getDim(1));
		Assert.assertEquals(0, tb.getNonZeros());
		Assert.assertTrue(tb.isSparse());
		Assert.assertTrue(tb.isMatrix());
	}

	@Test
	public void testMetaDefaultHeterogTensor() {
		HeterogTensor tb = new HeterogTensor();
		Assert.assertArrayEquals(new ValueType[0], tb.getSchema());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(0, tb.getNumColumns());
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaValueHeterogTensor() {
		HeterogTensor tb = new HeterogTensor(7.3);
		Assert.assertArrayEquals(new ValueType[]{ValueType.FP64}, tb.getSchema());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(1, tb.getNumRows());
		Assert.assertEquals(1, tb.getNumColumns());
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaTypedHeterogTensor() {
		HeterogTensor tb = new HeterogTensor(ValueType.INT64, new int[]{11, 12, 13});
		ValueType[] schema = new ValueType[12];
		Arrays.fill(schema, ValueType.INT64);
		Assert.assertArrayEquals(schema, tb.getSchema());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(12, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaSchemaTypedHeterogTensor() {
		ValueType[] schema = new ValueType[] {ValueType.BOOLEAN, ValueType.INT32, ValueType.STRING};
		HeterogTensor tb = new HeterogTensor(schema);
		Assert.assertArrayEquals(schema, tb.getSchema());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(schema.length, tb.getNumColumns());
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaNColsTypedHeterogTensor() {
		HeterogTensor tb = new HeterogTensor(10, ValueType.BOOLEAN);
		ValueType[] schema = new ValueType[10];
		Arrays.fill(schema, ValueType.BOOLEAN);
		Assert.assertArrayEquals(schema, tb.getSchema());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(10, tb.getNumColumns());
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaCopyDefaultHeterogTensor() {
		HeterogTensor tb = new HeterogTensor(new HeterogTensor());
		Assert.assertArrayEquals(new ValueType[0], tb.getSchema());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(0, tb.getNumRows());
		Assert.assertEquals(0, tb.getNumColumns());
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaCopyValueHeterogTensor() {
		HeterogTensor tb = new HeterogTensor(new HeterogTensor(7.3));
		Assert.assertArrayEquals(new ValueType[]{ValueType.FP64}, tb.getSchema());
		Assert.assertEquals(2, tb.getNumDims());
		Assert.assertEquals(1, tb.getNumRows());
		Assert.assertEquals(1, tb.getNumColumns());
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaCopyTypedHeterogTensor() {
		ValueType[] schema = {ValueType.INT32, ValueType.INT64, ValueType.BOOLEAN};
		HeterogTensor tb = new HeterogTensor(new HeterogTensor(schema, new int[]{11, 3, 13}));
		Assert.assertArrayEquals(schema, tb.getSchema());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(3, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaCopyTypedHeterogTensor2() {
		ValueType[] schema = {ValueType.FP64, ValueType.FP32, ValueType.STRING};
		HeterogTensor tb = new HeterogTensor(new HeterogTensor(schema, new int[]{11, 3, 13}));
		Assert.assertArrayEquals(schema, tb.getSchema());
		Assert.assertEquals(3, tb.getNumDims());
		Assert.assertEquals(11, tb.getNumRows());
		Assert.assertEquals(3, tb.getNumColumns());
		Assert.assertEquals(13, tb.getDim(2));
		Assert.assertFalse(tb.isMatrix());
	}

	@Test
	public void testMetaCopyTypedHeterogTensor3() {
		ValueType[] schema = {ValueType.FP64, ValueType.FP32, ValueType.INT64, ValueType.INT32, ValueType.BOOLEAN,
				ValueType.STRING};
		HeterogTensor tb = new HeterogTensor(new HeterogTensor(schema, new int[]{2, schema.length, 2}, new String[][]{
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
		Assert.assertFalse(tb.isMatrix());
	}
}
