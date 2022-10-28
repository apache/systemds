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

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class CustomDictionaryTest {

	protected static final Log LOG = LogFactory.getLog(CustomDictionaryTest.class.getName());

	@Test
	public void testContainsValue() {
		Dictionary d = Dictionary.createNoCheck(new double[] {1, 2, 3});
		assertTrue(d.containsValue(1));
		assertTrue(!d.containsValue(-1));
	}

	@Test
	public void testContainsValue_nan() {
		Dictionary d = Dictionary.createNoCheck(new double[] {Double.NaN, 2, 3});
		assertTrue(d.containsValue(Double.NaN));
	}

	@Test
	public void testContainsValue_nan_not() {
		Dictionary d = Dictionary.createNoCheck(new double[] {1, 2, 3});
		assertTrue(!d.containsValue(Double.NaN));
	}

	@Test
	public void testToString() {
		ADictionary d = Dictionary.create(new double[] {1.0, 2.0, 3.3, 4.0, 5.0, 6.0});
		String s = d.toString();
		assertFalse(s.contains("0"));
		assertTrue(s.contains("1"));
		assertTrue(s.contains("2"));
		assertTrue(s.contains("3.3"));
		assertTrue(s.contains("4"));
		assertTrue(s.contains("5"));
		assertTrue(s.contains("6"));
		assertTrue(s.contains(","));
	}

	@Test
	public void testGetString2() {
		ADictionary d = Dictionary.create(new double[] {1.0, 2.0, 3.3, 4.0, 5.0, 6.0});
		String s = d.getString(2);
		assertFalse(s.contains("0"));
		assertTrue(s.contains("1"));
		assertTrue(s.contains("2"));
		assertTrue(s.contains("3.3"));
		assertTrue(s.contains("4"));
		assertTrue(s.contains("5"));
		assertTrue(s.contains("6"));
		assertTrue(s.contains(","));
	}

	@Test
	public void testGetString1() {
		ADictionary d = Dictionary.create(new double[] {1.0, 2.0, 3.3, 4.0, 5.0, 6.0});
		String s = d.getString(1);
		assertFalse(s.contains("0"));
		assertTrue(s.contains("1"));
		assertTrue(s.contains("2"));
		assertTrue(s.contains("3.3"));
		assertTrue(s.contains("4"));
		assertTrue(s.contains("5"));
		assertTrue(s.contains("6"));
		assertTrue(s.contains(","));
	}

	@Test
	public void testGetString3() {
		ADictionary d = Dictionary.create(new double[] {1.0, 2.0, 3.3, 4.0, 5.0, 6.0});
		String s = d.getString(3);
		assertFalse(s.contains("0"));
		assertTrue(s.contains("1"));
		assertTrue(s.contains("2"));
		assertTrue(s.contains("3.3"));
		assertTrue(s.contains("4"));
		assertTrue(s.contains("5"));
		assertTrue(s.contains("6"));
		assertTrue(s.contains(","));
	}

	@Test
	public void isNullIfEmpty() {
		ADictionary d = Dictionary.create(new double[] {0, 0, 0, 0});
		assertNull("This should be null if empty creation", d);
	}

	@Test
	public void isNullIfEmptyMatrixBlock() {
		ADictionary d = MatrixBlockDictionary.create(new MatrixBlock(10, 10, 0.0));
		assertNull("This should be null if empty creation", d);
	}

	@Test(expected = DMLCompressionException.class)
	public void createEmpty() {
		Dictionary.create(new double[] {});
	}

	@Test(expected = DMLCompressionException.class)
	public void createNull() {
		Dictionary.create(null);
	}

	@Test(expected = DMLCompressionException.class)
	public void createNullMatrixBlock() {
		MatrixBlockDictionary.create(null);
	}

	@Test(expected = DMLCompressionException.class)
	public void createZeroRowAndColMatrixBlock() {
		MatrixBlockDictionary.create(new MatrixBlock(0, 0, 10.0));
	}

	@Test(expected = DMLCompressionException.class)
	public void createZeroColMatrixBlock() {
		MatrixBlockDictionary.create(new MatrixBlock(10, 0, 10.0));
	}

	@Test(expected = DMLCompressionException.class)
	public void createZeroRowMatrixBlock() {
		MatrixBlockDictionary.create(new MatrixBlock(0, 10, 10.0));
	}
}
