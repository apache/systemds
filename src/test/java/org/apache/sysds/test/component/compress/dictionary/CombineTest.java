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
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.AColGroupCompressed;
import org.apache.sysds.runtime.compress.colgroup.ADictBasedColGroup;
import org.apache.sysds.runtime.compress.colgroup.ASDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDC;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.utils.HashMapLongInt;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;


public class CombineTest {

	protected static final Log LOG = LogFactory.getLog(CombineTest.class.getName());

	@Test
	public void singleBothSides() {
		try {

			IDictionary a = Dictionary.create(new double[] {1.2});
			IDictionary b = Dictionary.create(new double[] {1.4});

			IDictionary c = DictionaryFactory.combineFullDictionaries(a, 1, b, 1);

			assertEquals(c.getValue(0, 0, 2), 1.2, 0.0);
			assertEquals(c.getValue(0, 1, 2), 1.4, 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void singleBothSidesFilter() {
		try {

			IDictionary a = Dictionary.create(new double[] {1.2});
			IDictionary b = Dictionary.create(new double[] {1.4});
			HashMapLongInt filter = new HashMapLongInt(3);
			filter.putIfAbsent(0, 0);
			IDictionary c = DictionaryFactory.combineFullDictionaries(a, 1, b, 1, filter);

			assertEquals(c.getValue(0, 0, 2), 1.2, 0.0);
			assertEquals(c.getValue(0, 1, 2), 1.4, 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void singleBothSidesFilter2() {
		try {

			IDictionary a = Dictionary.create(new double[] {1.2});
			IDictionary b = Dictionary.create(new double[] {1.4});
			HashMapLongInt filter = new HashMapLongInt(3);
			IDictionary c = DictionaryFactory.combineFullDictionaries(a, 1, b, 1, filter);

			assertEquals(c, null);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void singleOneSideBothSides() {
		try {
			IDictionary a = Dictionary.create(new double[] {1.2, 1.3});
			IDictionary b = Dictionary.create(new double[] {1.4});

			IDictionary c = DictionaryFactory.combineFullDictionaries(a, 1, b, 1);

			assertEquals(c.getValue(0, 0, 2), 1.2, 0.0);
			assertEquals(c.getValue(0, 1, 2), 1.4, 0.0);
			assertEquals(c.getValue(1, 0, 2), 1.3, 0.0);
			assertEquals(c.getValue(1, 1, 2), 1.4, 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void singleOneSideOtherSides() {
		try {
			IDictionary a = Dictionary.create(new double[] {1.2});
			IDictionary b = Dictionary.create(new double[] {1.3, 1.4});

			IDictionary c = DictionaryFactory.combineFullDictionaries(a, 1, b, 1);

			assertEquals(c.getValue(0, 0, 2), 1.2, 0.0);
			assertEquals(c.getValue(0, 1, 2), 1.3, 0.0);
			assertEquals(c.getValue(1, 0, 2), 1.2, 0.0);
			assertEquals(c.getValue(1, 1, 2), 1.4, 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void twoBothSides() {
		try {
			IDictionary a = Dictionary.create(new double[] {1.2, 1.3});
			IDictionary b = Dictionary.create(new double[] {1.4, 1.5});

			IDictionary c = DictionaryFactory.combineFullDictionaries(a, 1, b, 1);

			assertEquals(c.getValue(0, 0, 2), 1.2, 0.0);
			assertEquals(c.getValue(0, 1, 2), 1.4, 0.0);
			assertEquals(c.getValue(1, 0, 2), 1.3, 0.0);
			assertEquals(c.getValue(1, 1, 2), 1.4, 0.0);
			assertEquals(c.getValue(2, 0, 2), 1.2, 0.0);
			assertEquals(c.getValue(2, 1, 2), 1.5, 0.0);
			assertEquals(c.getValue(3, 0, 2), 1.3, 0.0);
			assertEquals(c.getValue(3, 1, 2), 1.5, 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void twoBothSidesFilter() {
		try {
			IDictionary a = Dictionary.create(new double[] {1.2, 1.3});
			IDictionary b = Dictionary.create(new double[] {1.4, 1.5});
			HashMapLongInt filter = new HashMapLongInt(3);
			filter.putIfAbsent(0, 0);

			IDictionary c = DictionaryFactory.combineFullDictionaries(a, 1, b, 1, filter);

			assertEquals(1, c.getNumberOfValues(2));
			assertEquals(c.getValue(0, 0, 2), 1.2, 0.0);
			assertEquals(c.getValue(0, 1, 2), 1.4, 0.0);
			// assertEquals(c.getValue(1, 0, 2), 1.3, 0.0);
			// assertEquals(c.getValue(1, 1, 2), 1.4, 0.0);
			// assertEquals(c.getValue(2, 0, 2), 1.2, 0.0);
			// assertEquals(c.getValue(2, 1, 2), 1.5, 0.0);
			// assertEquals(c.getValue(3, 0, 2), 1.3, 0.0);
			// assertEquals(c.getValue(3, 1, 2), 1.5, 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void twoBothSidesFilter2() {
		try {
			IDictionary a = Dictionary.create(new double[] {1.2, 1.3});
			IDictionary b = Dictionary.create(new double[] {1.4, 1.5});
			HashMapLongInt filter = new HashMapLongInt(3);
			filter.putIfAbsent(3, 0);

			IDictionary c = DictionaryFactory.combineFullDictionaries(a, 1, b, 1, filter);

			assertEquals(1, c.getNumberOfValues(2));
			assertEquals(c.getValue(0, 0, 2), 1.3, 0.0);
			assertEquals(c.getValue(0, 1, 2), 1.5, 0.0);
			// assertEquals(c.getValue(1, 0, 2), 1.3, 0.0);
			// assertEquals(c.getValue(1, 1, 2), 1.4, 0.0);
			// assertEquals(c.getValue(2, 0, 2), 1.2, 0.0);
			// assertEquals(c.getValue(2, 1, 2), 1.5, 0.0);
			// assertEquals(c.getValue(3, 0, 2), 1.3, 0.0);
			// assertEquals(c.getValue(3, 1, 2), 1.5, 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void twoBothSidesFilter3() {
		try {
			IDictionary a = Dictionary.create(new double[] {1.2, 1.3});
			IDictionary b = Dictionary.create(new double[] {1.4, 1.5});
			HashMapLongInt filter = new HashMapLongInt(3);
			filter.putIfAbsent(3, 0);
			filter.putIfAbsent(1, 1);

			IDictionary c = DictionaryFactory.combineFullDictionaries(a, 1, b, 1, filter);

			assertEquals(2, c.getNumberOfValues(2));
			assertEquals(c.getValue(0, 0, 2), 1.3, 0.0);
			assertEquals(c.getValue(0, 1, 2), 1.5, 0.0);
			assertEquals(c.getValue(1, 0, 2), 1.3, 0.0);
			assertEquals(c.getValue(1, 1, 2), 1.4, 0.0);
			// assertEquals(c.getValue(2, 0, 2), 1.2, 0.0);
			// assertEquals(c.getValue(2, 1, 2), 1.5, 0.0);
			// assertEquals(c.getValue(3, 0, 2), 1.3, 0.0);
			// assertEquals(c.getValue(3, 1, 2), 1.5, 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void sparseSparse() {
		try {
			IDictionary a = Dictionary.create(new double[] {3});
			IDictionary b = Dictionary.create(new double[] {4});
			double[] ad = new double[] {0};
			double[] bd = new double[] {0};

			IDictionary c = DictionaryFactory.combineSDCNoFilter(a, ad, b, bd);
			MatrixBlock ret = c.getMBDict(2).getMatrixBlock();

			MatrixBlock exp = new MatrixBlock(4, 2, new double[] {0, 0, 3, 0, 0, 4, 3, 4});
			TestUtils.compareMatricesBitAvgDistance(ret, exp, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void sparseSparse2() {
		try {
			IDictionary a = Dictionary.create(new double[] {3});
			IDictionary b = Dictionary.create(new double[] {4, 4});
			double[] ad = new double[] {0};
			double[] bd = new double[] {0, 0};

			IDictionary c = DictionaryFactory.combineSDCNoFilter(a, ad, b, bd);
			MatrixBlock ret = c.getMBDict(2).getMatrixBlock();

			MatrixBlock exp = new MatrixBlock(4, 3, new double[] {0, 0, 0, 3, 0, 0, 0, 4, 4, 3, 4, 4});
			TestUtils.compareMatricesBitAvgDistance(ret, exp, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void sparseSparse3() {
		try {
			IDictionary a = Dictionary.create(new double[] {3});
			IDictionary b = Dictionary.create(new double[] {4});
			double[] ad = new double[] {1};
			double[] bd = new double[] {2};

			IDictionary c = DictionaryFactory.combineSDCNoFilter(a, ad, b, bd);
			MatrixBlock ret = c.getMBDict(2).getMatrixBlock();

			MatrixBlock exp = new MatrixBlock(4, 2, new double[] {//
				1, 2, //
				3, 2, //
				1, 4, //
				3, 4});
			TestUtils.compareMatricesBitAvgDistance(ret, exp, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void sparseSparse4() {
		try {
			IDictionary a = Dictionary.create(new double[] {3, 2});
			IDictionary b = Dictionary.create(new double[] {4, 4});
			double[] ad = new double[] {0, 1};
			double[] bd = new double[] {0, 2};

			IDictionary c = DictionaryFactory.combineSDCNoFilter(a, ad, b, bd);
			MatrixBlock ret = c.getMBDict(2).getMatrixBlock();

			MatrixBlock exp = new MatrixBlock(4, 4, new double[] {//
				0, 1, 0, 2, //
				3, 2, 0, 2, //
				0, 1, 4, 4, //
				3, 2, 4, 4});
			TestUtils.compareMatricesBitAvgDistance(ret, exp, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void sparseSparse5() {
		try {
			IDictionary a = Dictionary.create(new double[] {3, 2, 7, 8});
			IDictionary b = Dictionary.create(new double[] {4, 4});
			double[] ad = new double[] {0, 1};
			double[] bd = new double[] {0, 2};

			IDictionary c = DictionaryFactory.combineSDCNoFilter(a, ad, b, bd);
			MatrixBlock ret = c.getMBDict(2).getMatrixBlock();

			MatrixBlock exp = new MatrixBlock(6, 4, new double[] {//
				0, 1, 0, 2, //
				3, 2, 0, 2, //
				7, 8, 0, 2, //
				0, 1, 4, 4, //
				3, 2, 4, 4, //
				7, 8, 4, 4,});
			TestUtils.compareMatricesBitAvgDistance(ret, exp, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void sparseSparse6() {
		try {
			IDictionary a = Dictionary.create(new double[] {3, 2, 7, 8});
			IDictionary b = Dictionary.create(new double[] {4, 4, 9, 5});
			double[] ad = new double[] {0, 1};
			double[] bd = new double[] {0, 2};

			IDictionary c = DictionaryFactory.combineSDCNoFilter(a, ad, b, bd);
			MatrixBlock ret = c.getMBDict(2).getMatrixBlock();

			MatrixBlock exp = new MatrixBlock(9, 4, new double[] {//
				0, 1, 0, 2, //
				3, 2, 0, 2, //
				7, 8, 0, 2, //
				0, 1, 4, 4, //
				3, 2, 4, 4, //
				7, 8, 4, 4, //
				0, 1, 9, 5, //
				3, 2, 9, 5, //
				7, 8, 9, 5,});
			TestUtils.compareMatricesBitAvgDistance(ret, exp, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void combineEmpties() {
		assertNull(DictionaryFactory.combineDictionaries(ColGroupEmpty.create(120), ColGroupEmpty.create(22)));
	}

	@Test(expected = NotImplementedException.class)
	public void combineNotImplemented1() {
		ADictBasedColGroup m = mock(ADictBasedColGroup.class);
		when(m.getCompType()).thenReturn(CompressionType.UNCOMPRESSED);
		DictionaryFactory.combineDictionaries(m, m);
	}

	@Test(expected = NotImplementedException.class)
	public void combineNotImplemented2() {
		ADictBasedColGroup m = mock(ADictBasedColGroup.class);
		when(m.getCompType()).thenReturn(CompressionType.UNCOMPRESSED);
		ADictBasedColGroup d = mock(ColGroupDDC.class);
		when(d.getCompType()).thenReturn(CompressionType.DDC);
		DictionaryFactory.combineDictionaries(d, m);
	}

	@Test(expected = NotImplementedException.class)
	public void combineNotImplemented3() {
		ADictBasedColGroup m = mock(ADictBasedColGroup.class);
		when(m.getCompType()).thenReturn(CompressionType.UNCOMPRESSED);
		ADictBasedColGroup d = mock(ColGroupDDC.class);
		when(d.getCompType()).thenReturn(CompressionType.DDC);
		DictionaryFactory.combineDictionaries(m, d);
	}

	@Test(expected = NotImplementedException.class)
	public void combineNotImplemented4() {
		ADictBasedColGroup m = mock(ADictBasedColGroup.class);
		when(m.getCompType()).thenReturn(CompressionType.UNCOMPRESSED);
		ADictBasedColGroup s = mock(ColGroupSDC.class);
		when(s.getCompType()).thenReturn(CompressionType.SDC);
		DictionaryFactory.combineDictionaries(s, m);
	}

	@Test(expected = NotImplementedException.class)
	public void combineNotImplemented5() {
		ADictBasedColGroup m = mock(ADictBasedColGroup.class);
		when(m.getCompType()).thenReturn(CompressionType.UNCOMPRESSED);
		ADictBasedColGroup s = mock(ColGroupSDC.class);
		when(s.getCompType()).thenReturn(CompressionType.SDC);
		DictionaryFactory.combineDictionaries(m, s);
	}

	@Test(expected = NotImplementedException.class)
	public void combineNotImplemented6() {
		AColGroupCompressed m = mock(AColGroupCompressed.class);
		when(m.getCompType()).thenReturn(CompressionType.UNCOMPRESSED);
		AColGroupCompressed s = mock(ColGroupSDC.class);
		when(s.getCompType()).thenReturn(CompressionType.SDC);
		DictionaryFactory.combineDictionaries(s, m);
	}

	@Test(expected = NotImplementedException.class)
	public void combineNotImplemented7() {
		AColGroupCompressed m = mock(AColGroupCompressed.class);
		when(m.getCompType()).thenReturn(CompressionType.UNCOMPRESSED);
		AColGroupCompressed s = mock(AColGroupCompressed.class);
		when(s.getCompType()).thenReturn(CompressionType.UNCOMPRESSED);
		DictionaryFactory.combineDictionaries(s, m);
	}

	@Test(expected = NotImplementedException.class)
	public void combineNotImplemented8() {
		AColGroupCompressed m = mock(AColGroupCompressed.class);
		when(m.getCompType()).thenReturn(CompressionType.UNCOMPRESSED);
		AColGroupCompressed s = mock(ColGroupSDC.class);
		when(s.getCompType()).thenReturn(CompressionType.SDC);
		DictionaryFactory.combineDictionaries(m, s);
	}

	@Test(expected = NotImplementedException.class)
	public void combineNotImplemented9() {
		AColGroupCompressed m = mock(ColGroupConst.class);
		when(m.getCompType()).thenReturn(CompressionType.CONST);
		AColGroupCompressed s = mock(ColGroupSDC.class);
		when(s.getCompType()).thenReturn(CompressionType.SDC);
		DictionaryFactory.combineDictionaries(m, s);
	}

	@Test(expected = NotImplementedException.class)
	public void combineNotImplementedSparse1() {
		ADictBasedColGroup m = mock(ADictBasedColGroup.class);
		when(m.getCompType()).thenReturn(CompressionType.UNCOMPRESSED);
		ADictBasedColGroup s = mock(ColGroupSDC.class);
		when(s.getCompType()).thenReturn(CompressionType.SDC);
		DictionaryFactory.combineDictionariesSparse(m, s);
	}

	@Test(expected = NotImplementedException.class)
	public void combineNotImplementedSparse2() {
		ADictBasedColGroup m = mock(ADictBasedColGroup.class);
		when(m.getCompType()).thenReturn(CompressionType.UNCOMPRESSED);
		ADictBasedColGroup s = mock(ColGroupSDC.class);
		when(s.getCompType()).thenReturn(CompressionType.SDC);
		DictionaryFactory.combineDictionariesSparse(s, m);
	}

	@Test(expected = NotImplementedException.class)
	public void combineNotImplementedSparse3() {
		ADictBasedColGroup m = mock(ColGroupSDC.class);
		when(m.getCompType()).thenReturn(CompressionType.SDC);
		ADictBasedColGroup s = mock(ColGroupDDC.class);
		when(s.getCompType()).thenReturn(CompressionType.DDC);
		DictionaryFactory.combineDictionariesSparse(s, m);
	}

	@Test(expected = NotImplementedException.class)
	public void combineNotImplementedSparse4() {
		ADictBasedColGroup m = mock(ColGroupConst.class);
		when(m.getCompType()).thenReturn(CompressionType.CONST);
		ADictBasedColGroup s = mock(ColGroupDDC.class);
		when(s.getCompType()).thenReturn(CompressionType.DDC);
		DictionaryFactory.combineDictionariesSparse(s, m);
	}

	@Test(expected = NotImplementedException.class)
	public void combineNotImplementedSparse5() {
		ADictBasedColGroup m = mock(ColGroupSDC.class);
		when(m.getCompType()).thenReturn(CompressionType.SDC);
		ADictBasedColGroup s = mock(ColGroupDDC.class);
		when(s.getCompType()).thenReturn(CompressionType.DDC);
		DictionaryFactory.combineDictionariesSparse(s, m);
	}

	@Test(expected = NotImplementedException.class)
	public void combineNotImplementedSparse6() {
		ADictBasedColGroup m = mock(ColGroupConst.class);
		when(m.getCompType()).thenReturn(CompressionType.CONST);
		ADictBasedColGroup s = mock(ColGroupDDC.class);
		when(s.getCompType()).thenReturn(CompressionType.DDC);
		DictionaryFactory.combineDictionariesSparse(m, s);
	}

	@Test
	public void testEmpty() {
		try {
			IDictionary d = Dictionary.create(new double[] {3, 2, 7, 8});
			AColGroup a = ColGroupDDC.create(ColIndexFactory.createI(1, 2), d, MapToFactory.create(10, 2), null);
			ColGroupEmpty b = new ColGroupEmpty(ColIndexFactory.createI(3, 4, 5, 6));

			IDictionary c = DictionaryFactory.combineDictionaries((AColGroupCompressed) a, (AColGroupCompressed) b);
			MatrixBlock ret = c.getMBDict(2).getMatrixBlock();

			MatrixBlock exp = new MatrixBlock(2, 6, new double[] {//
				3, 2, 0, 0, 0, 0, //
				7, 8, 0, 0, 0, 0,});
			TestUtils.compareMatricesBitAvgDistance(ret, exp, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void combineDictionariesSparse1() {
		try {
			IDictionary d = Dictionary.create(new double[] {3, 2, 7, 8});
			AColGroup a = ColGroupSDC.create(ColIndexFactory.createI(1, 2), 500, d, new double[] {1, 2},
				OffsetFactory.createOffset(new int[] {3, 4}), MapToFactory.create(10, 2), null);
			ColGroupEmpty b = new ColGroupEmpty(ColIndexFactory.createI(3, 4, 5, 6));

			IDictionary c = DictionaryFactory.combineDictionariesSparse((AColGroupCompressed) a, (AColGroupCompressed) b);
			MatrixBlock ret = c.getMBDict(2).getMatrixBlock();

			MatrixBlock exp = new MatrixBlock(2, 6, new double[] {//
				3, 2, 0, 0, 0, 0, //
				7, 8, 0, 0, 0, 0,});
			TestUtils.compareMatricesBitAvgDistance(ret, exp, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void combineDictionariesSparse2() {
		try {
			IDictionary d = Dictionary.create(new double[] {//
				3, 2, //
				7, 8});
			AColGroup a = ColGroupSDC.create(ColIndexFactory.createI(1, 2), 500, d, new double[] {1, 2},
				OffsetFactory.createOffset(new int[] {3, 4}), MapToFactory.create(10, 2), null);
			ColGroupEmpty b = new ColGroupEmpty(ColIndexFactory.createI(3, 4, 5, 6));

			IDictionary c = DictionaryFactory.combineDictionariesSparse((AColGroupCompressed) a, (AColGroupCompressed) b);
			MatrixBlock ret = c.getMBDict(2).getMatrixBlock();

			MatrixBlock exp = new MatrixBlock(2, 6, new double[] {//
				3, 2, 0, 0, 0, 0, //
				7, 8, 0, 0, 0, 0,});
			TestUtils.compareMatricesBitAvgDistance(ret, exp, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void combineMockingEmpty() {
		IDictionary ad = Dictionary.create(new double[] {1, 2, 3, 4});
		double[] ade = new double[] {0};
		AColGroupCompressed a = mockSDC(ad, ade, ColIndexFactory.create(1));
		AColGroupCompressed b = mockSDC(ad, ade, ColIndexFactory.create(2));

		HashMapLongInt m = new HashMapLongInt(10);
		IDictionary red = DictionaryFactory.combineDictionaries(a, b, m);

		assertEquals(red.getNumberOfValues(2), 0);
	}

	@Test
	public void combineMockingDefault() {
		try {
			IDictionary ad = Dictionary.create(new double[] {1, 2, 3, 4});
			double[] ade = new double[] {0};
			AColGroupCompressed a = mockSDC(ad, ade, ColIndexFactory.create(1));
			AColGroupCompressed b = mockSDC(ad, ade, ColIndexFactory.create(2));
			HashMapLongInt m = new HashMapLongInt(10);
			m.putIfAbsent(0, 0);
			IDictionary red = DictionaryFactory.combineDictionaries(a, b, m);
			assertEquals(red.getNumberOfValues(2), 1);
			assertEquals(Dictionary.createNoCheck(new double[] {0, 0}), red);
			assertEquals(red, Dictionary.createNoCheck(new double[] {0, 0}));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void combineMockingFirstValue() {
		IDictionary ad = Dictionary.create(new double[] {1, 2, 3, 4});
		double[] ade = new double[] {0};
		AColGroupCompressed a = mockSDC(ad, ade, ColIndexFactory.create(1));
		AColGroupCompressed b = mockSDC(ad, ade, ColIndexFactory.create(2));

		HashMapLongInt m = new HashMapLongInt(10);
		m.putIfAbsent(1, 0);
		IDictionary red = DictionaryFactory.combineDictionaries(a, b, m);

		assertEquals(red.getNumberOfValues(2), 1);
		assertEquals(red, Dictionary.create(new double[] {0, 1}));
	}

	@Test
	public void combineMockingFirstAndDefault() {
		IDictionary ad = Dictionary.create(new double[] {1, 2, 3, 4});
		double[] ade = new double[] {0};
		AColGroupCompressed a = mockSDC(ad, ade, ColIndexFactory.create(1));
		AColGroupCompressed b = mockSDC(ad, ade, ColIndexFactory.create(2));

		HashMapLongInt m = new HashMapLongInt(10);
		m.putIfAbsent(1, 0);
		m.putIfAbsent(0, 1);
		IDictionary red = DictionaryFactory.combineDictionaries(a, b, m);

		assertEquals(red.getNumberOfValues(2), 2);
		assertEquals(red, Dictionary.create(new double[] {0, 1, 0, 0}));
	}

	@Test
	public void combineMockingMixed() {
		IDictionary ad = Dictionary.create(new double[] {1, 2, 3, 4});
		double[] ade = new double[] {0};
		AColGroupCompressed a = mockSDC(ad, ade, ColIndexFactory.create(1));
		AColGroupCompressed b = mockSDC(ad, ade, ColIndexFactory.create(2));

		HashMapLongInt m = new HashMapLongInt(10);
		m.putIfAbsent(1, 0);
		m.putIfAbsent(0, 1);
		m.putIfAbsent(5, 2);
		IDictionary red = DictionaryFactory.combineDictionaries(a, b, m);

		assertEquals(red.getNumberOfValues(2), 3);
		assertEquals(Dictionary.create(new double[] {0, 1, 0, 0, 1, 0}), red);
	}

	@Test
	public void combineMockingMixed2() {
		IDictionary ad = Dictionary.create(new double[] {1, 2, 3, 4});
		double[] ade = new double[] {0};
		AColGroupCompressed a = mockSDC(ad, ade, ColIndexFactory.create(1));
		AColGroupCompressed b = mockSDC(ad, ade, ColIndexFactory.create(2));

		HashMapLongInt m = new HashMapLongInt(10);
		m.putIfAbsent(1, 0);
		m.putIfAbsent(0, 1);
		m.putIfAbsent(10, 2);
		IDictionary red = DictionaryFactory.combineDictionaries(a, b, m);

		assertEquals(red.getNumberOfValues(2), 3);
		assertEquals(Dictionary.create(new double[] {0, 1, 0, 0, 2, 0}), red);
	}

	@Test
	public void combineMockingSparseDenseEmpty() {
		try {

			IDictionary ad = Dictionary.create(new double[] {1, 2, 3, 4});
			double[] ade = new double[] {0};
			AColGroupCompressed a = mockDDC(ad, ColIndexFactory.create(1));
			AColGroupCompressed b = mockSDC(ad, ade, ColIndexFactory.create(2));

			HashMapLongInt m = new HashMapLongInt(10);
			IDictionary red = DictionaryFactory.combineDictionaries(a, b, m);

			assertEquals(0, red.getNumberOfValues(2));
			assertEquals(Dictionary.createNoCheck(new double[] {}), red);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void combineMockingSparseDenseOne() {
		try {

			IDictionary ad = Dictionary.create(new double[] {1, 2, 3, 4});
			double[] ade = new double[] {0};
			AColGroupCompressed a = mockDDC(ad, ColIndexFactory.create(1));
			AColGroupCompressed b = mockSDC(ad, ade, ColIndexFactory.create(2));

			HashMapLongInt m = new HashMapLongInt(10);
			m.putIfAbsent(0, 0);
			IDictionary red = DictionaryFactory.combineDictionaries(a, b, m);
			assertEquals(1, red.getNumberOfValues(2));
			assertEquals(Dictionary.createNoCheck(new double[] {0, 1}), red);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void combineMockingSparseDenseMixed1() {
		try {

			IDictionary ad = Dictionary.create(new double[] {1, 2, 3, 4});
			double[] ade = new double[] {0};
			AColGroupCompressed a = mockDDC(ad, ColIndexFactory.create(1));
			AColGroupCompressed b = mockSDC(ad, ade, ColIndexFactory.create(2));

			HashMapLongInt m = new HashMapLongInt(10);
			m.putIfAbsent(0, 1);
			m.putIfAbsent(1, 0);
			IDictionary red = DictionaryFactory.combineDictionaries(a, b, m);

			assertEquals(2, red.getNumberOfValues(2));
			assertEquals(Dictionary.createNoCheck(new double[] {0, 2, 0, 1}), red);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void combineMockingSparseDenseMixed2() {
		try {

			IDictionary ad = Dictionary.create(new double[] {1, 2, 3, 4});
			double[] ade = new double[] {0};
			AColGroupCompressed a = mockDDC(ad, ColIndexFactory.create(1));
			AColGroupCompressed b = mockSDC(ad, ade, ColIndexFactory.create(2));

			HashMapLongInt m = new HashMapLongInt(10);
			m.putIfAbsent(0, 1);
			m.putIfAbsent(1, 0);
			m.putIfAbsent(4, 2);
			IDictionary red = DictionaryFactory.combineDictionaries(a, b, m);

			assertEquals(3, red.getNumberOfValues(2));
			assertEquals(Dictionary.createNoCheck(new double[] {0, 2, 0, 1, 1, 1}), red);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void combineMockingSparseDenseMixed3() {
		try {

			IDictionary ad = Dictionary.create(new double[] {1, 2, 3, 4});
			double[] ade = new double[] {0};
			AColGroupCompressed a = mockDDC(ad, ColIndexFactory.create(1));
			AColGroupCompressed b = mockSDC(ad, ade, ColIndexFactory.create(2));

			HashMapLongInt m = new HashMapLongInt(10);
			m.putIfAbsent(0, 1);
			m.putIfAbsent(1, 0);
			m.putIfAbsent(5, 2);
			m.putIfAbsent(4, 3);
			IDictionary red = DictionaryFactory.combineDictionaries(a, b, m);

			assertEquals(4, red.getNumberOfValues(2));
			assertEquals(Dictionary.createNoCheck(new double[] {0, 2, 0, 1, 1, 2, 1, 1}), red);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void combineFailCase1() {
		try {

			IDictionary ad = Dictionary.create(new double[] {3, 1, 2});
			IDictionary ab = Dictionary.create(new double[] {2, 3});
			double[] ade = new double[] {1};
			AColGroupCompressed a = mockDDC(ad, ColIndexFactory.create(1));
			AColGroupCompressed b = mockSDC(ab, ade, ColIndexFactory.create(2));

			HashMapLongInt m = new HashMapLongInt(10);
			// 0=8, 1=7, 2=5, 3=0, 4=6, 5=2, 6=4, 7=1, 8=3
			m.putIfAbsent(0, 8);
			m.putIfAbsent(1, 7);
			m.putIfAbsent(2, 5);
			m.putIfAbsent(3, 0);
			m.putIfAbsent(4, 6);
			m.putIfAbsent(5, 2);
			m.putIfAbsent(6, 4);
			m.putIfAbsent(7, 1);
			m.putIfAbsent(8, 3);
			IDictionary red = DictionaryFactory.combineDictionaries(a, b, m);

			assertEquals(9, red.getNumberOfValues(2));
			assertEquals(Dictionary.createNoCheck(//
				new double[] {//
					2, 3, //
					3, 1, //
					2, 2, //
					3, 2, //
					3, 3, //
					1, 2, //
					2, 1, //
					1, 1, //
					1, 3,//
				}), red);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void combineFailCase2() {
		try {

			IDictionary ad = Dictionary.create(new double[] {3, 1, 2});
			IDictionary ab = Dictionary.create(new double[] {2, 3});
			double[] ade = new double[] {1};
			AColGroupCompressed a = mockDDC(ad, ColIndexFactory.createI(1));
			AColGroupCompressed b = mockSDC(ab, ade, ColIndexFactory.createI(2));

			HashMapLongInt m = new HashMapLongInt(10);
			for(int i = 0; i < 9; i++) {
				m.putIfAbsent(i, i);
			}

			IDictionary red = DictionaryFactory.combineDictionaries(a, b, m);

			assertEquals(9, red.getNumberOfValues(2));
			assertEquals(Dictionary.createNoCheck(//
				new double[] {//
					3, 1, //
					1, 1, //
					2, 1, //
					3, 2, //
					1, 2, //
					2, 2, //
					3, 3, //
					1, 3, //
					2, 3,//
				}), red);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testCombineSDC() {
		IDictionary ad = Dictionary.create(new double[] {2, 3});
		IDictionary ab = Dictionary.create(new double[] {1, 2});
		double[] ade = new double[] {1.0};
		double[] abe = new double[] {3.0};
		AColGroupCompressed a = mockSDC(ad, ade, ColIndexFactory.createI(1));
		AColGroupCompressed b = mockSDC(ab, abe, ColIndexFactory.createI(2));
		HashMapLongInt m = new HashMapLongInt(10);
		m.putIfAbsent(0, 8);
		m.putIfAbsent(1, 0);
		m.putIfAbsent(2, 4);
		m.putIfAbsent(3, 7);
		m.putIfAbsent(4, 6);
		m.putIfAbsent(5, 1);
		m.putIfAbsent(6, 5);
		m.putIfAbsent(7, 2);
		m.putIfAbsent(8, 3);

		IDictionary red = DictionaryFactory.combineDictionaries(a, b, m);

		assertEquals(9, red.getNumberOfValues(2));
		assertEquals(Dictionary.createNoCheck(//
			new double[] {//
				2, 3, //
				3, 1, //
				2, 2, //
				3, 2, //
				3, 3, //
				1, 2, //
				2, 1, //
				1, 1, //
				1, 3,//
			}), red);
	}

	@Test
	public void testCombineSDCRange() {
		IDictionary ad = Dictionary.create(new double[] {2, 3});
		IDictionary ab = Dictionary.create(new double[] {1, 2});
		double[] ade = new double[] {1.0};
		double[] abe = new double[] {3.0};
		AColGroupCompressed a = mockSDC(ad, ade, ColIndexFactory.createI(1));
		AColGroupCompressed b = mockSDC(ab, abe, ColIndexFactory.createI(2));
		HashMapLongInt m = new HashMapLongInt(10);
		for(int i = 0; i < 9; i++) {
			m.putIfAbsent(i, i);
		}
		IDictionary red = DictionaryFactory.combineDictionaries(a, b, m);

		assertEquals(9, red.getNumberOfValues(2));
		assertEquals(Dictionary.createNoCheck(//
			new double[] {//
				1, 3, //
				2, 3, //
				3, 3, //
				1, 1, //
				2, 1, //
				3, 1, //
				1, 2, //
				2, 2, //
				3, 2,//
			}), red);
	}

	private ASDC mockSDC(IDictionary ad, double[] def, IColIndex c) {
		ASDC a = mock(ASDC.class);
		when(a.getCompType()).thenReturn(CompressionType.SDC);
		when(a.getDictionary()).thenReturn(ad);
		when(a.getDefaultTuple()).thenReturn(def);
		when(a.getNumCols()).thenReturn(def.length);
		when(a.getColIndices()).thenReturn(c);
		return a;
	}

	private ColGroupDDC mockDDC(IDictionary ad, IColIndex c) {
		ColGroupDDC a = mock(ColGroupDDC.class);
		when(a.getCompType()).thenReturn(CompressionType.DDC);
		when(a.getDictionary()).thenReturn(ad);
		when(a.getNumCols()).thenReturn(c.size());
		when(a.getColIndices()).thenReturn(c);
		return a;
	}
}
