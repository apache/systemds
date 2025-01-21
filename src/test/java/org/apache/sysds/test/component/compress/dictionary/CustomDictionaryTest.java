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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.bitmap.BitmapEncoder;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IdentityDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayCountHashMap;
import org.apache.sysds.runtime.compress.utils.DoubleCountHashMap;
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

	@Test
	public void bitMapConstructor() {
		MatrixBlock mb = new MatrixBlock(10, 10, 1.0);
		mb.set(5, 5, 2.0);
		mb.set(7, 5, 2.0);
		final ABitmap ubm = BitmapEncoder.extractBitmap(ColIndexFactory.create(10), mb, true, 1, true);
		final double[] defaultTuple = new double[10];

		IDictionary dict = DictionaryFactory.create(ubm, 0, defaultTuple, 1.0, ubm.getNumZeros() > 0);
		assertEquals(dict, Dictionary.create(new double[] {//
			1, 1, 1, 1, 1, 2, 1, 2, 1, 1}));
		assertTrue(Arrays.equals(defaultTuple, new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
	}

	@Test
	public void bitMapConstructor2() {
		MatrixBlock mb = new MatrixBlock(10, 10, 1.0);
		mb.set(5, 5, 2.0);
		mb.set(7, 7, 2.0);
		final ABitmap ubm = BitmapEncoder.extractBitmap(ColIndexFactory.create(10), mb, true, 1, true);
		final double[] defaultTuple = new double[10];

		IDictionary dict = DictionaryFactory.create(ubm, 0, defaultTuple, 1.0, ubm.getNumZeros() > 0);
		assertEquals(dict, Dictionary.create(new double[] {//
			1, 1, 1, 1, 1, 1, 1, 2, 1, 1, //
			1, 1, 1, 1, 1, 2, 1, 1, 1, 1}));
		assertTrue(Arrays.equals(defaultTuple, new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
	}

	@Test
	public void bitMapConstructor3() {
		MatrixBlock mb = new MatrixBlock(10, 10, 1.0);
		mb.set(5, 5, 2.0);
		mb.set(7, 7, 2.0);
		final ABitmap ubm = BitmapEncoder.extractBitmap(ColIndexFactory.create(10), mb, true, 1, true);
		final double[] defaultTuple = new double[10];

		IDictionary dict = DictionaryFactory.create(ubm, 1, defaultTuple, 1.0, ubm.getNumZeros() > 0);
		assertEquals(dict, Dictionary.create(new double[] {//
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //
			1, 1, 1, 1, 1, 2, 1, 1, 1, 1}));
		assertTrue(Arrays.equals(defaultTuple, new double[] {1, 1, 1, 1, 1, 1, 1, 2, 1, 1}));
	}

	@Test
	public void bitMapConstructor4_Sparse() {
		MatrixBlock mb = new MatrixBlock(10, 10, 0.0);
		mb.set(5, 5, 2.0);
		mb.set(7, 7, 2.0);
		mb.set(8, 8, 2.0);
		final ABitmap ubm = BitmapEncoder.extractBitmap(ColIndexFactory.create(10), mb, true, 1, true);
		final double[] defaultTuple = new double[10];

		IDictionary dict = DictionaryFactory.create(ubm, 1, defaultTuple, 0.1, ubm.getNumZeros() > 0);
		assertEquals(dict, Dictionary.create(new double[] {//
			0, 0, 0, 0, 0, 2, 0, 0, 0, 0, //
			0, 0, 0, 0, 0, 0, 0, 0, 2, 0, //
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
		}));
		assertTrue(Arrays.equals(defaultTuple, new double[] {0, 0, 0, 0, 0, 0, 0, 2, 0, 0}));
	}

	@Test
	public void bitMapConstructorVector() {
		MatrixBlock mb = new MatrixBlock(10, 1, 1.0);
		mb.set(5, 1, 2.0);
		mb.set(7, 1, 3.0);
		final ABitmap ubm = BitmapEncoder.extractBitmap(ColIndexFactory.create(1), mb, false, 1, true);
		final double[] defaultTuple = new double[1];

		IDictionary dict = DictionaryFactory.create(ubm, 0, defaultTuple, 1.0, ubm.getNumZeros() > 0);
		assertEquals(dict, Dictionary.create(new double[] {//
			2, 3}));
		assertTrue(Arrays.equals(defaultTuple, new double[] {1}));
	}

	@Test
	public void bitMapConstructorVector2() {
		MatrixBlock mb = new MatrixBlock(10, 1, 1.0);
		mb.set(5, 1, 2.0);
		mb.set(7, 1, 3.0);
		mb.set(8, 1, 3.0);
		final ABitmap ubm = BitmapEncoder.extractBitmap(ColIndexFactory.create(1), mb, false, 1, true);
		final double[] defaultTuple = new double[1];

		IDictionary dict = DictionaryFactory.create(ubm, 0, defaultTuple, 1.0, ubm.getNumZeros() > 0);
		assertEquals(dict, Dictionary.create(new double[] {//
			3, 2}));
		assertTrue(Arrays.equals(defaultTuple, new double[] {1}));
	}

	@Test
	public void bitMapConstructorVector3() {
		MatrixBlock mb = new MatrixBlock(10, 1, 1.0);
		mb.set(5, 1, 2.0);
		mb.set(7, 1, 3.0);
		mb.set(8, 1, 3.0);
		final ABitmap ubm = BitmapEncoder.extractBitmap(ColIndexFactory.create(1), mb, false, 1, true);
		final double[] defaultTuple = new double[1];

		IDictionary dict = DictionaryFactory.create(ubm, 0, defaultTuple, 1.0, true);
		assertEquals(dict, Dictionary.create(new double[] {//
			3, 2, 0.0}));
		assertTrue(Arrays.equals(defaultTuple, new double[] {1}));
	}

	@Test
	public void bitMapConstruct() {
		MatrixBlock mb = new MatrixBlock(10, 1, 1.0);
		mb.set(5, 0, 2.0);
		mb.set(7, 0, 3.0);
		mb.set(8, 0, 3.0);
		final ABitmap ubm = BitmapEncoder.extractBitmap(ColIndexFactory.create(1), mb, false, 1, true);

		IDictionary dict = DictionaryFactory.create(ubm);
		assertEquals(dict, Dictionary.create(new double[] {//
			1, 3, 2}));
	}

	@Test
	public void bitMapConstruct2() {
		MatrixBlock mb = new MatrixBlock(10, 1, 1.0);
		mb.set(5, 0, 2.0);
		mb.set(7, 0, 3.0);
		mb.set(8, 0, 2.0);
		final ABitmap ubm = BitmapEncoder.extractBitmap(ColIndexFactory.create(1), mb, false, 1, true);

		IDictionary dict = DictionaryFactory.create(ubm);
		assertEquals(dict, Dictionary.create(new double[] {//
			1, 2, 3}));
	}

	@Test
	public void bitMapConstruct3() {
		MatrixBlock mb = new MatrixBlock(10, 2, 1.0);
		mb.set(5, 0, 2.0);
		mb.set(7, 0, 3.0);
		mb.set(8, 0, 2.0);
		final ABitmap ubm = BitmapEncoder.extractBitmap(ColIndexFactory.create(2), mb, false, 1, true);

		IDictionary dict = DictionaryFactory.create(ubm);
		assertEquals(dict, Dictionary.create(new double[] {//
			1, 1, //
			2, 1, //
			3, 1,//
		}));
	}

	@Test
	public void bitMapConstruct4Sparse() {
		MatrixBlock mb = new MatrixBlock(10, 5, 0.0);
		mb.set(5, 0, 2.0);
		mb.set(7, 0, 3.0);
		mb.set(8, 0, 2.0);
		final ABitmap ubm = BitmapEncoder.extractBitmap(ColIndexFactory.create(5), mb, false, 1, true);

		IDictionary dict = DictionaryFactory.create(ubm, 0.1);
		assertEquals(dict, Dictionary.create(new double[] {//
			2, 0, 0, 0, 0, //
			3, 0, 0, 0, 0,//
		}));
	}

	@Test
	public void bitMapConstruct4Sparse2() {
		MatrixBlock mb = new MatrixBlock(10, 3, 0.0);
		mb.set(5, 0, 2.0);
		mb.set(7, 0, 3.0);
		mb.set(8, 0, 2.0);
		final ABitmap ubm = BitmapEncoder.extractBitmap(ColIndexFactory.create(3), mb, false, 1, true);

		IDictionary dict = DictionaryFactory.create(ubm, 0.1);
		assertEquals(dict, Dictionary.create(new double[] {//
			2, 0, 0, //
			3, 0, 0,//
		}));
	}

	@Test
	public void getInMemorySize() {
		long s = DictionaryFactory.getInMemorySize(100, 100, 1.0, false);
		long s2 = Dictionary.getInMemorySize(100 * 100);
		assertTrue(s <= s2);
	}

	@Test
	public void getInMemorySize2() {
		long s = DictionaryFactory.getInMemorySize(100, 100, 0.1, false);
		long s2 = MatrixBlockDictionary.getInMemorySize(100, 100, 0.1);
		assertTrue(s <= s2);
	}

	@Test
	public void getInMemorySize3() {
		long s = DictionaryFactory.getInMemorySize(100, 100, 1.0, true);
		long s2 = Dictionary.getInMemorySize(100 * 100);
		assertTrue(s <= s2);
	}

	@Test
	public void getInMemorySize4() {
		long s = DictionaryFactory.getInMemorySize(100, 1, 1.0, true);
		long s2 = Dictionary.getInMemorySize(100);
		assertTrue(s <= s2);
	}

	@Test
	public void getInMemorySize5() {
		long s = DictionaryFactory.getInMemorySize(100, 1, 1.0, false);
		long s2 = Dictionary.getInMemorySize(100);
		assertTrue(s <= s2);
	}

	@Test
	public void createDblArrayCount() {

		DblArrayCountHashMap m = new DblArrayCountHashMap(3);
		m.increment(new DblArray(new double[] {1, 2, 3}));
		m.increment(new DblArray(new double[] {1, 2, 4}));
		m.increment(new DblArray(new double[] {1, 2, 3}));
		IDictionary d = DictionaryFactory.create(m, 3, false, 1.0);

		assertEquals(Dictionary.create(new double[] {//
			1, 2, 3, //
			1, 2, 4,//
		}), d);
	}

	@Test
	public void createDblArrayCount2() {

		DblArrayCountHashMap m = new DblArrayCountHashMap(3);
		m.increment(new DblArray(new double[] {1, 2, 3}));
		m.increment(new DblArray(new double[] {1, 2, 4}));
		m.increment(new DblArray(new double[] {1, 2, 5}));
		IDictionary d = DictionaryFactory.create(m, 3, false, 1.0);

		assertEquals(Dictionary.create(new double[] {//
			1, 2, 3, //
			1, 2, 4, //
			1, 2, 5,//
		}), d);
	}

	@Test
	public void createDblArrayCountSparse() {

		DblArrayCountHashMap m = new DblArrayCountHashMap(3);
		m.increment(new DblArray(new double[] {1, 2, 3}));
		m.increment(new DblArray(new double[] {1, 2, 4}));
		m.increment(new DblArray(new double[] {1, 2, 5}));
		IDictionary d = DictionaryFactory.create(m, 3, false, 0.2);

		assertEquals(Dictionary.create(new double[] {//
			1, 2, 3, //
			1, 2, 4, //
			1, 2, 5,//
		}), d);
	}

	@Test
	public void createDblArrayCountSparse2() {

		DblArrayCountHashMap m = new DblArrayCountHashMap(3);
		m.increment(new DblArray(new double[] {1, 2, 3, 1, 1}));
		m.increment(new DblArray(new double[] {1, 2, 4, 1, 1}));
		m.increment(new DblArray(new double[] {1, 2, 5, 1, 1}));
		IDictionary d = DictionaryFactory.create(m, 5, false, 0.2);

		assertEquals(Dictionary.create(new double[] {//
			1, 2, 3, 1, 1, //
			1, 2, 4, 1, 1, //
			1, 2, 5, 1, 1,//
		}), d);
	}

	@Test
	public void createDblArrayCountSparse3() {

		DblArrayCountHashMap m = new DblArrayCountHashMap(3);
		m.increment(new DblArray(new double[] {0, 2, 3, 0, 0}));
		m.increment(new DblArray(new double[] {0, 2, 4, 0, 0}));
		m.increment(new DblArray(new double[] {1, 2, 5, 1, 1}));
		m.increment(new DblArray(new double[] {1, 2, 5, 1, 1}));
		m.increment(new DblArray(new double[] {1, 2, 5, 1, 1}));
		IDictionary d = DictionaryFactory.create(m, 5, true, 0.2);

		assertEquals(Dictionary.create(new double[] {//
			0, 2, 3, 0, 0, //
			0, 2, 4, 0, 0, //
			1, 2, 5, 1, 1, //
			0, 0, 0, 0, 0}), d);
	}

	@Test
	public void createDoubleCountHashMap() {

		DoubleCountHashMap m = new DoubleCountHashMap(3);
		m.increment(1);
		m.increment(2);
		m.increment(4);
		m.increment(6);
		m.increment(1);
		IDictionary d = DictionaryFactory.create(m);

		assertEquals(Dictionary.create(new double[] {//
			1, 2, 4, 6,}), d);
	}
	public void IdentityDictionaryEquals() {
		IDictionary a = new IdentityDictionary(10);
		IDictionary b = new IdentityDictionary(10);
		assertTrue(a.equals(b));
	}

	@Test
	public void IdentityDictionaryNotEquals() {
		IDictionary a = new IdentityDictionary(10);
		IDictionary b = new IdentityDictionary(11);
		assertFalse(a.equals(b));
	}

	@Test
	public void IdentityDictionaryNotEquals2() {
		IDictionary a = new IdentityDictionary(10);
		IDictionary b = new IdentityDictionary(11, false);
		assertFalse(a.equals(b));
	}

	@Test
	public void IdentityDictionaryEquals2() {
		IDictionary a = new IdentityDictionary(11, false);
		IDictionary b = new IdentityDictionary(11, false);
		assertTrue(a.equals(b));
	}

	@Test
	public void IdentityDictionaryEquals2v() {
		IDictionary a = new IdentityDictionary(11);
		IDictionary b = new IdentityDictionary(11, false);
		assertTrue(a.equals(b));
	}

	@Test
	public void IdentityDictionaryNotEquals3() {
		IDictionary a = new IdentityDictionary(11, true);
		IDictionary b = new IdentityDictionary(11, false);
		assertFalse(a.equals(b));
	}

}
