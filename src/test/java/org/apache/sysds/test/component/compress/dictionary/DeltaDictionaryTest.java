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
package org.apache.sysds.test.component.compress.dictionary;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import org.apache.sysds.runtime.compress.colgroup.dictionary.DeltaDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary.DictType;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.junit.Assert;
import org.junit.Test;

public class DeltaDictionaryTest {

	@Test
	public void testScalarOpRightMultiplySingleColumn() {
		double scalar = 2;
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2}, 1);
		ScalarOperator sop = new RightScalarOperator(Multiply.getMultiplyFnObject(), scalar, 1);
		d = d.applyScalarOp(sop);
		double[] expected = new double[] {2, 4};
		Assert.assertArrayEquals(expected, d.getValues(), 0.01);
	}

	@Test
	public void testScalarOpRightMultiplyTwoColumns() {
		double scalar = 2;
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
		ScalarOperator sop = new RightScalarOperator(Multiply.getMultiplyFnObject(), scalar, 1);
		d = d.applyScalarOp(sop);
		double[] expected = new double[] {2, 4, 6, 8};
		Assert.assertArrayEquals(expected, d.getValues(), 0.01);
	}

	@Test
	public void testNegScalarOpRightMultiplyTwoColumns() {
		double scalar = -2;
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
		ScalarOperator sop = new RightScalarOperator(Multiply.getMultiplyFnObject(), scalar, 1);
		d = d.applyScalarOp(sop);
		double[] expected = new double[] {-2, -4, -6, -8};
		Assert.assertArrayEquals(expected, d.getValues(), 0.01);
	}

	@Test
	public void testScalarOpLeftMultiplyTwoColumns() {
		double scalar = 2;
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
		ScalarOperator sop = new LeftScalarOperator(Multiply.getMultiplyFnObject(), scalar, 1);
		d = d.applyScalarOp(sop);
		double[] expected = new double[] {2, 4, 6, 8};
		Assert.assertArrayEquals(expected, d.getValues(), 0.01);
	}

	@Test
	public void testScalarOpRightDivideTwoColumns() {
		double scalar = 0.5;
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
		ScalarOperator sop = new RightScalarOperator(Divide.getDivideFnObject(), scalar, 1);
		d = d.applyScalarOp(sop);
		double[] expected = new double[] {2, 4, 6, 8};
		Assert.assertArrayEquals(expected, d.getValues(), 0.01);
	}


	@Test
	public void testSerializationSingleColumn() throws IOException {
		DeltaDictionary original = new DeltaDictionary(new double[] {1, 2, 3, 4, 5}, 1);
		
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DataOutputStream dos = new DataOutputStream(bos);
		original.write(dos);
		Assert.assertEquals(original.getExactSizeOnDisk(), bos.size());
		
		ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
		DataInputStream dis = new DataInputStream(bis);
		IDictionary deserialized = DictionaryFactory.read(dis);
		
		Assert.assertTrue("Deserialized dictionary should be DeltaDictionary", deserialized instanceof DeltaDictionary);
		DeltaDictionary deltaDict = (DeltaDictionary) deserialized;
		Assert.assertArrayEquals("Values should match after serialization", original.getValues(), deltaDict.getValues(), 0.01);
	}

	@Test
	public void testSerializationTwoColumns() throws IOException {
		DeltaDictionary original = new DeltaDictionary(new double[] {1, 2, 3, 4, 5, 6}, 2);
		
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DataOutputStream dos = new DataOutputStream(bos);
		original.write(dos);
		Assert.assertEquals(original.getExactSizeOnDisk(), bos.size());
		
		ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
		DataInputStream dis = new DataInputStream(bis);
		IDictionary deserialized = DictionaryFactory.read(dis);
		
		Assert.assertTrue("Deserialized dictionary should be DeltaDictionary", deserialized instanceof DeltaDictionary);
		DeltaDictionary deltaDict = (DeltaDictionary) deserialized;
		Assert.assertArrayEquals("Values should match after serialization", original.getValues(), deltaDict.getValues(), 0.01);
	}

	@Test
	public void testGetValue() {
		DeltaDictionary d = new DeltaDictionary(new double[] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 2);
		Assert.assertEquals(1.0, d.getValue(0, 0, 2), 0.01);
		Assert.assertEquals(2.0, d.getValue(0, 1, 2), 0.01);
		Assert.assertEquals(3.0, d.getValue(1, 0, 2), 0.01);
		Assert.assertEquals(4.0, d.getValue(1, 1, 2), 0.01);
		Assert.assertEquals(5.0, d.getValue(2, 0, 2), 0.01);
		Assert.assertEquals(6.0, d.getValue(2, 1, 2), 0.01);
	}

	@Test
	public void testGetValueSingleColumn() {
		DeltaDictionary d = new DeltaDictionary(new double[] {1.0, 2.0, 3.0}, 1);
		Assert.assertEquals(1.0, d.getValue(0, 0, 1), 0.01);
		Assert.assertEquals(2.0, d.getValue(1, 0, 1), 0.01);
		Assert.assertEquals(3.0, d.getValue(2, 0, 1), 0.01);
	}

	@Test
	public void testGetDictType() {
		DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
		Assert.assertEquals(DictType.Delta, d.getDictType());
	}

	@Test
	public void testGetString() {
		DeltaDictionary d = new DeltaDictionary(new double[] {1.0, 2.0, 3.0, 4.0}, 2);
		String result = d.getString(2);
		String expected = "1.0, 2.0\n3.0, 4.0";
		Assert.assertEquals(expected, result);
	}

	@Test
	public void testGetStringSingleColumn() {
		DeltaDictionary d = new DeltaDictionary(new double[] {1.0, 2.0, 3.0}, 1);
		String result = d.getString(1);
		String expected = "1.0\n2.0\n3.0";
		Assert.assertEquals(expected, result);
	}

}
