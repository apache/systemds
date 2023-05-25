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

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ADictBasedColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDC;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class CombineTest {

	protected static final Log LOG = LogFactory.getLog(CombineTest.class.getName());

	@Test
	public void singleBothSides() {
		try {

			ADictionary a = Dictionary.create(new double[] {1.2});
			ADictionary b = Dictionary.create(new double[] {1.4});

			ADictionary c = DictionaryFactory.combineFullDictionaries(a, 1, b, 1);

			assertEquals(c.getValue(0, 0, 2), 1.2, 0.0);
			assertEquals(c.getValue(0, 1, 2), 1.4, 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void singleOneSideBothSides() {
		try {
			ADictionary a = Dictionary.create(new double[] {1.2, 1.3});
			ADictionary b = Dictionary.create(new double[] {1.4});

			ADictionary c = DictionaryFactory.combineFullDictionaries(a, 1, b, 1);

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
	public void twoBothSides() {
		try {
			ADictionary a = Dictionary.create(new double[] {1.2, 1.3});
			ADictionary b = Dictionary.create(new double[] {1.4, 1.5});

			ADictionary c = DictionaryFactory.combineFullDictionaries(a, 1, b, 1);

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
	public void sparseSparse() {
		try {
			ADictionary a = Dictionary.create(new double[] {3});
			ADictionary b = Dictionary.create(new double[] {4});
			double[] ad = new double[] {0};
			double[] bd = new double[] {0};

			ADictionary c = DictionaryFactory.combineSDC(a, ad, b, bd);
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
			ADictionary a = Dictionary.create(new double[] {3});
			ADictionary b = Dictionary.create(new double[] {4, 4});
			double[] ad = new double[] {0};
			double[] bd = new double[] {0, 0};

			ADictionary c = DictionaryFactory.combineSDC(a, ad, b, bd);
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
			ADictionary a = Dictionary.create(new double[] {3});
			ADictionary b = Dictionary.create(new double[] {4});
			double[] ad = new double[] {1};
			double[] bd = new double[] {2};

			ADictionary c = DictionaryFactory.combineSDC(a, ad, b, bd);
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
			ADictionary a = Dictionary.create(new double[] {3, 2});
			ADictionary b = Dictionary.create(new double[] {4, 4});
			double[] ad = new double[] {0, 1};
			double[] bd = new double[] {0, 2};

			ADictionary c = DictionaryFactory.combineSDC(a, ad, b, bd);
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
			ADictionary a = Dictionary.create(new double[] {3, 2, 7, 8});
			ADictionary b = Dictionary.create(new double[] {4, 4});
			double[] ad = new double[] {0, 1};
			double[] bd = new double[] {0, 2};

			ADictionary c = DictionaryFactory.combineSDC(a, ad, b, bd);
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
			ADictionary a = Dictionary.create(new double[] {3, 2, 7, 8});
			ADictionary b = Dictionary.create(new double[] {4, 4, 9, 5});
			double[] ad = new double[] {0, 1};
			double[] bd = new double[] {0, 2};

			ADictionary c = DictionaryFactory.combineSDC(a, ad, b, bd);
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
}
