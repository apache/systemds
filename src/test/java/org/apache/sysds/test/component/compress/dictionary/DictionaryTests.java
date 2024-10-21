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
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IdentityDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ArrayIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.RangeIndex;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import scala.util.Random;

@RunWith(value = Parameterized.class)
public class DictionaryTests {

	protected static final Log LOG = LogFactory.getLog(DictionaryTests.class.getName());

	private final int nRow;
	private final int nCol;
	private final IDictionary a;
	private final IDictionary b;

	public DictionaryTests(IDictionary a, IDictionary b, int nRow, int nCol) {
		this.nRow = nRow;
		this.nCol = nCol;
		this.a = a;
		this.b = b;
	}

	@Parameters
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();

		try {
			addAll(tests, new double[] {1, 1, 1, 1, 1}, 1);
			addAll(tests, new double[] {-3, 0.0, 132, 43, 1}, 1);
			addAll(tests, new double[] {1, 2, 3, 4, 5}, 1);
			addAll(tests, new double[] {1, 2, 3, 4, 5, 6}, 2);
			addAll(tests, new double[] {1, 2.2, 3.3, 4.4, 5.5, 6.6}, 3);

			tests.add(new Object[] {new IdentityDictionary(2), Dictionary.create(new double[] {1, 0, 0, 1}), 2, 2});
			tests.add(new Object[] {new IdentityDictionary(2, true), //
				Dictionary.create(new double[] {1, 0, 0, 1, 0, 0}), 3, 2});
			tests.add(new Object[] {new IdentityDictionary(3), //
				Dictionary.create(new double[] {1, 0, 0, 0, 1, 0, 0, 0, 1}), 3, 3});
			tests.add(new Object[] {new IdentityDictionary(3, true), //
				Dictionary.create(new double[] {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}), 4, 3});

			tests.add(new Object[] {new IdentityDictionary(4), //
				Dictionary.create(new double[] {//
					1, 0, 0, 0, //
					0, 1, 0, 0, //
					0, 0, 1, 0, //
					0, 0, 0, 1,//
				}), 4, 4});

			tests.add(new Object[] {new IdentityDictionary(4).sliceOutColumnRange(1, 4, 4), //
				Dictionary.create(new double[] {//
					0, 0, 0, //
					1, 0, 0, //
					0, 1, 0, //
					0, 0, 1,//
				}), 4, 3});
			tests.add(new Object[] {new IdentityDictionary(4, true), //
				Dictionary.create(new double[] {//
					1, 0, 0, 0, //
					0, 1, 0, 0, //
					0, 0, 1, 0, //
					0, 0, 0, 1, //
					0, 0, 0, 0}),
				5, 4});

			tests.add(new Object[] {new IdentityDictionary(4, true).sliceOutColumnRange(1, 4, 4), //
				Dictionary.create(new double[] {//
					0, 0, 0, //
					1, 0, 0, //
					0, 1, 0, //
					0, 0, 1, //
					0, 0, 0}),
				5, 3});
			tests.add(new Object[] {new IdentityDictionary(4, true), //
				Dictionary.create(new double[] {//
					1, 0, 0, 0, //
					0, 1, 0, 0, //
					0, 0, 1, 0, //
					0, 0, 0, 1, //
					0, 0, 0, 0}).getMBDict(4),
				5, 4});

			tests.add(new Object[] {new IdentityDictionary(20, false), //
				MatrixBlockDictionary.create(//
					new MatrixBlock(20, 20, 20L, //
						SparseBlockFactory.createIdentityMatrix(20)),
					false),
				20, 20});

			tests.add(new Object[] {new IdentityDictionary(20, false), //
				Dictionary.create(new double[] {//
					1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, //
				}), //
				20, 20});

			tests.add(new Object[] {new IdentityDictionary(20, true), //
				Dictionary.create(new double[] {//
					1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, //
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
				}), //
				21, 20});

			create(tests, 30, 300, 0.2);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	private static void create(List<Object[]> tests, int rows, int cols, double sparsity) {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(rows, cols, -3, 3, 0.2, 1342);
		mb.recomputeNonZeros();
		MatrixBlock dense = new MatrixBlock();

		dense.copy(mb);
		dense.sparseToDense();
		double[] values = dense.getDenseBlockValues();

		tests.add(new Object[] {//
			Dictionary.create(values), //
			MatrixBlockDictionary.create(mb), //
			rows, cols});

		tests.add(new Object[] {//
			Dictionary.create(values), //
			MatrixBlockDictionary.create(dense), //
			rows, cols});
	}

	private static void addAll(List<Object[]> tests, double[] vals, int cols) {
		tests.add(new Object[] {//
			Dictionary.create(vals), //
			MatrixBlockDictionary.createDictionary(vals, cols, true), //
			vals.length / cols, cols});
	}

	@Test
	public void sum() {
		int[] counts = getCounts(nRow, 1324);
		double as = a.sum(counts, nCol);
		double bs = b.sum(counts, nCol);
		assertEquals(as, bs, 0.0000001);
	}

	@Test
	public void sum2() {
		int[] counts = getCounts(nRow, 124);
		double as = a.sum(counts, nCol);
		double bs = b.sum(counts, nCol);
		assertEquals(as, bs, 0.0000001);
	}

	@Test
	public void sum3() {
		int[] counts = getCounts(nRow, 124444);
		double as = a.sum(counts, nCol);
		double bs = b.sum(counts, nCol);
		assertEquals(as, bs, 0.0000001);
	}

	@Test
	public void getValues() {
		try {
			double[] av = a.getValues();
			double[] bv = b.getValues();
			TestUtils.compareMatricesBitAvgDistance(av, bv, 10, 10, "Not Equivalent values from getValues");
		}
		catch(DMLCompressionException e) {
			// okay since some cases are safeguarded by not allowing extraction of dense values.
		}
	}

	@Test
	public void getDictType() {
		assertNotEquals(a.getDictType(), b.getDictType());
	}

	@Test
	public void getSparsity() {
		assertEquals(a.getSparsity(), b.getSparsity(), 0.001);
	}

	@Test
	public void productZero() {
		product(0.0);
	}

	@Test
	public void productOne() {
		product(1.0);
	}

	@Test
	public void productMore() {
		product(30.0);
	}

	public void product(double retV) {
		// Shared
		final int[] counts = getCounts(nRow, 1324);

		// A
		final double[] aRet = new double[] {retV};
		a.product(aRet, counts, nCol);

		// B
		final double[] bRet = new double[] {retV};
		b.product(bRet, counts, nCol);

		TestUtils.compareMatricesBitAvgDistance(//
			aRet, bRet, 10, 10, "Not Equivalent values from product");
	}

	@Test
	public void productWithReferenceZero() {
		final double[] reference = getReference(nCol, 132, -3, 3);
		productWithReference(0.0, reference);
	}

	@Test
	public void productWithReferenceOne() {
		final double[] reference = getReference(nCol, 132, -3, 3);
		productWithReference(1.0, reference);
	}

	@Test
	public void productWithDoctoredReference() {
		final double[] reference = getReference(nCol, 132, 0.0, 0.0);
		productWithReference(1.0, reference);
	}

	@Test
	public void productWithDoctoredReference2() {
		final double[] reference = getReference(nCol, 132, 1.0, 1.0);
		productWithReference(1.0, reference);
	}

	public void productWithReference(double retV, double[] reference) {
		// Shared
		final int[] counts = getCounts(nRow, 1324);

		// A
		final double[] aRet = new double[] {retV};
		a.productWithReference(aRet, counts, reference, nCol);

		// B
		final double[] bRet = new double[] {retV};
		b.productWithReference(bRet, counts, reference, nCol);

		TestUtils.compareMatricesBitAvgDistance(//
			aRet, bRet, 10, 10, "Not Equivalent values from product");
	}

	@Test
	public void productWithdefZero() {
		final double[] def = getReference(nCol, 132, -3, 3);
		productWithDefault(0.0, def);
	}

	@Test
	public void productWithdefOne() {
		final double[] def = getReference(nCol, 132, -3, 3);
		productWithDefault(1.0, def);
	}

	@Test
	public void productWithDoctoreddef() {
		final double[] def = getReference(nCol, 132, 0.0, 0.0);
		productWithDefault(1.0, def);
	}

	@Test
	public void productWithDoctoreddef2() {
		final double[] def = getReference(nCol, 132, 1.0, 1.0);
		productWithDefault(1.0, def);
	}

	@Test
	public void replace() {
		final Random rand = new Random(13);
		final int r = rand.nextInt(nRow);
		final int c = rand.nextInt(nCol);
		final double v = a.getValue(r, c, nCol);
		final double rep = rand.nextDouble();
		final IDictionary aRep = a.replace(v, rep, nCol);
		final IDictionary bRep = b.replace(v, rep, nCol);
		assertEquals(aRep.getValue(r, c, nCol), rep, 0.0000001);
		assertEquals(bRep.getValue(r, c, nCol), rep, 0.0000001);
	}

	@Test
	public void replaceWitReference() {
		final Random rand = new Random(444);
		final int r = rand.nextInt(nRow);
		final int c = rand.nextInt(nCol);
		final double[] reference = getReference(nCol, 44, 1.0, 1.0);
		final double before = a.getValue(r, c, nCol);
		final double v = before + 1.0;
		final double rep = rand.nextDouble() * 500;
		final IDictionary aRep = a.replaceWithReference(v, rep, reference);
		final IDictionary bRep = b.replaceWithReference(v, rep, reference);
		assertEquals(aRep.getValue(r, c, nCol), bRep.getValue(r, c, nCol), 0.0000001);
		assertNotEquals(before, aRep.getValue(r, c, nCol), 0.00001);
	}

	@Test
	public void rexpandCols() {
		if(nCol == 1) {
			int max = (int) a.aggregate(0, Builtin.getBuiltinFnObject(BuiltinCode.MAX));
			final IDictionary aR = a.rexpandCols(max + 1, true, false, nCol);
			final IDictionary bR = b.rexpandCols(max + 1, true, false, nCol);
			compare(aR, bR, nRow, max + 1);
		}
	}

	@Test(expected = DMLCompressionException.class)
	public void rexpandColsException() {
		if(nCol > 1) {
			int max = (int) a.aggregate(0, Builtin.getBuiltinFnObject(BuiltinCode.MAX));
			b.rexpandCols(max + 1, true, false, nCol);
		}
		else
			throw new DMLCompressionException("to test pase");
	}

	@Test(expected = DMLCompressionException.class)
	public void rexpandColsExceptionOtherOrder() {
		if(nCol > 1) {
			int max = (int) a.aggregate(0, Builtin.getBuiltinFnObject(BuiltinCode.MAX));
			a.rexpandCols(max + 1, true, false, nCol);
		}
		else
			throw new DMLCompressionException("to test pase");
	}

	@Test
	public void rexpandColsWithReference1() {
		rexpandColsWithReference(1);
	}

	@Test
	public void rexpandColsWithReference33() {
		rexpandColsWithReference(33);
	}

	@Test
	public void rexpandColsWithReference_neg23() {
		rexpandColsWithReference(-23);
	}

	@Test
	public void rexpandColsWithReference_neg1() {
		rexpandColsWithReference(-1);
	}

	public void rexpandColsWithReference(int reference) {
		if(nCol == 1) {
			int max = (int) a.aggregate(0, Builtin.getBuiltinFnObject(BuiltinCode.MAX));

			final IDictionary aR = a.rexpandColsWithReference(max + 1, true, false, reference);
			final IDictionary bR = b.rexpandColsWithReference(max + 1, true, false, reference);
			if(aR == null && bR == null)
				return; // valid
			compare(aR, bR, nRow, max + 1);
		}
	}

	@Test
	public void sumSq() {
		int[] counts = getCounts(nRow, 2323);
		double as = a.sumSq(counts, nCol);
		double bs = b.sumSq(counts, nCol);
		assertEquals(as, bs, 0.0001);
	}

	@Test
	public void sumSqWithReference() {
		int[] counts = getCounts(nRow, 2323);
		double[] reference = getReference(nCol, 323, -10, 23);
		double as = a.sumSqWithReference(counts, reference);
		double bs = b.sumSqWithReference(counts, reference);
		assertEquals(as, bs, 0.0001);
	}

	@Test
	public void sliceOutColumnRange() {
		Random r = new Random(2323);
		int s = r.nextInt(nCol);
		int e = r.nextInt(nCol - s) + s + 1;
		IDictionary ad = a.sliceOutColumnRange(s, e, nCol);
		IDictionary bd = b.sliceOutColumnRange(s, e, nCol);
		compare(ad, bd, nRow, e - s);
	}

	@Test
	public void contains1() {
		containsValue(1);
	}

	@Test
	public void contains2() {
		containsValue(2);
	}

	@Test
	public void contains100() {
		containsValue(100);
	}

	@Test
	public void contains0() {
		containsValue(0);
	}

	@Test
	public void contains1p1() {
		containsValue(1.1);
	}

	public void containsValue(double value) {
		assertEquals(a.containsValue(value), b.containsValue(value));
	}

	@Test
	public void contains1WithReference() {
		containsValueWithReference(1, getReference(nCol, 3241, 1.0, 1.0));
	}

	@Test
	public void contains1WithReference2() {
		containsValueWithReference(1, getReference(nCol, 3241, 1.0, 1.32));
	}

	@Test
	public void contains32WithReference2() {
		containsValueWithReference(32, getReference(nCol, 3241, -1.0, 1.32));
	}

	@Test
	public void contains0WithReference1() {
		containsValueWithReference(0, getReference(nCol, 3241, 1.0, 1.0));
	}

	@Test
	public void contains1WithReferenceMinus1() {
		containsValueWithReference(1.0, getReference(nCol, 3241, -1.0, -1.0));
	}

	@Test
	public void equalsEl() {
		try {
			assertEquals(a, b);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void equalsElOp() {
		assertEquals(b, a);
	}

	@Test
	public void opRightMinus() {
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject());
		double[] vals = TestUtils.generateTestVector(nCol, -1, 1, 1.0, 132L);
		opRight(op, vals, ColIndexFactory.create(0, nCol));
	}

	@Test
	public void opRightMinusNoCol() {
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject());
		double[] vals = TestUtils.generateTestVector(nCol, -1, 1, 1.0, 132L);
		opRight(op, vals);
	}

	@Test
	public void opRightMinusZero() {
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject());
		double[] vals = new double[nCol];
		opRight(op, vals, ColIndexFactory.create(0, nCol));
	}

	@Test
	public void opRightDivOne() {
		BinaryOperator op = new BinaryOperator(Divide.getDivideFnObject());
		double[] vals = new double[nCol];
		Arrays.fill(vals, 1);
		opRight(op, vals, ColIndexFactory.create(0, nCol));
	}

	@Test
	public void opRightDiv() {
		BinaryOperator op = new BinaryOperator(Divide.getDivideFnObject());
		double[] vals = TestUtils.generateTestVector(nCol, -1, 1, 1.0, 232L);
		opRight(op, vals, ColIndexFactory.create(0, nCol));
	}

	private void opRight(BinaryOperator op, double[] vals, IColIndex cols) {
		try {

			IDictionary aa = a.binOpRight(op, vals, cols);
			IDictionary bb = b.binOpRight(op, vals, cols);
			compare(aa, bb, nRow, nCol);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private void opRight(BinaryOperator op, double[] vals) {
		IDictionary aa = a.binOpRight(op, vals);
		IDictionary bb = b.binOpRight(op, vals);
		compare(aa, bb, nRow, nCol);
	}

	@Test
	public void testAddToEntry1() {
		double[] ret1 = new double[nCol];
		a.addToEntry(ret1, 0, 0, nCol);
		double[] ret2 = new double[nCol];
		b.addToEntry(ret2, 0, 0, nCol);
		assertTrue(Arrays.equals(ret1, ret2));
	}

	@Test
	public void testAddToEntry2() {
		double[] ret1 = new double[nCol * 2];
		a.addToEntry(ret1, 0, 1, nCol);
		double[] ret2 = new double[nCol * 2];
		b.addToEntry(ret2, 0, 1, nCol);
		assertTrue(Arrays.equals(ret1, ret2));
	}

	@Test
	public void testAddToEntry3() {
		double[] ret1 = new double[nCol * 3];
		a.addToEntry(ret1, 0, 2, nCol);
		double[] ret2 = new double[nCol * 3];
		b.addToEntry(ret2, 0, 2, nCol);
		assertTrue(Arrays.equals(ret1, ret2));
	}

	@Test
	public void testAddToEntry4() {
		if(a.getNumberOfValues(nCol) > 2) {

			double[] ret1 = new double[nCol * 3];
			a.addToEntry(ret1, 2, 2, nCol);
			double[] ret2 = new double[nCol * 3];
			b.addToEntry(ret2, 2, 2, nCol);
			assertTrue(Arrays.equals(ret1, ret2));
		}
	}

	@Test
	public void testAddToEntryRep1() {
		double[] ret1 = new double[nCol];
		a.addToEntry(ret1, 0, 0, nCol, 16);
		double[] ret2 = new double[nCol];
		b.addToEntry(ret2, 0, 0, nCol, 16);
		assertTrue(Arrays.equals(ret1, ret2));
	}

	@Test
	public void testAddToEntryRep2() {
		double[] ret1 = new double[nCol * 2];
		a.addToEntry(ret1, 0, 1, nCol, 3214);
		double[] ret2 = new double[nCol * 2];
		b.addToEntry(ret2, 0, 1, nCol, 3214);
		assertTrue(Arrays.equals(ret1, ret2));
	}

	@Test
	public void testAddToEntryRep3() {
		double[] ret1 = new double[nCol * 3];
		a.addToEntry(ret1, 0, 2, nCol, 222);
		double[] ret2 = new double[nCol * 3];
		b.addToEntry(ret2, 0, 2, nCol, 222);
		assertTrue(Arrays.equals(ret1, ret2));
	}

	@Test
	public void testAddToEntryRep4() {
		if(a.getNumberOfValues(nCol) > 2) {
			double[] ret1 = new double[nCol * 3];
			a.addToEntry(ret1, 2, 2, nCol, 321);
			double[] ret2 = new double[nCol * 3];
			b.addToEntry(ret2, 2, 2, nCol, 321);
			assertTrue(Arrays.equals(ret1, ret2));
		}
	}

	@Test
	public void testAddToEntryVectorized1() {
		try {
			double[] ret1 = new double[nCol * 3];
			a.addToEntryVectorized(ret1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 2, 0, 1, nCol);
			double[] ret2 = new double[nCol * 3];
			b.addToEntryVectorized(ret2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 2, 0, 1, nCol);
			assertTrue(Arrays.equals(ret1, ret2));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void max() {
		aggregate(Builtin.getBuiltinFnObject(BuiltinCode.MAX));
	}

	@Test
	public void min() {
		aggregate(Builtin.getBuiltinFnObject(BuiltinCode.MIN));
	}

	@Test(expected = NotImplementedException.class)
	public void cMax() {
		aggregate(Builtin.getBuiltinFnObject(BuiltinCode.CUMMAX));
		throw new NotImplementedException();
	}

	private void aggregate(Builtin fn) {
		try {
			double aa = a.aggregate(0, fn);
			double bb = b.aggregate(0, fn);
			assertEquals(aa, bb, 0.0);
		}
		catch(NotImplementedException ee) {
			throw ee;
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testAddToEntryVectorized2() {
		try {

			if(a.getNumberOfValues(nCol) > 1) {
				double[] ret1 = new double[nCol * 3];
				a.addToEntryVectorized(ret1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 2, 0, 1, 2, 0, 1, nCol);
				double[] ret2 = new double[nCol * 3];
				b.addToEntryVectorized(ret2, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 2, 0, 1, 2, 0, 1, nCol);
				assertTrue("Error: " + a.getClass().getSimpleName() + " " + b.getClass().getSimpleName(),
					Arrays.equals(ret1, ret2));
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testAddToEntryVectorized3() {
		try {

			if(a.getNumberOfValues(nCol) > 2) {
				double[] ret1 = new double[nCol * 3];
				a.addToEntryVectorized(ret1, 1, 2, 1, 2, 1, 0, 1, 0, 0, 1, 2, 0, 1, 2, 0, 1, nCol);
				double[] ret2 = new double[nCol * 3];
				b.addToEntryVectorized(ret2, 1, 2, 1, 2, 1, 0, 1, 0, 0, 1, 2, 0, 1, 2, 0, 1, nCol);
				assertTrue("Error: " + a.getClass().getSimpleName() + " " + b.getClass().getSimpleName(),
					Arrays.equals(ret1, ret2));
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testAddToEntryVectorized4() {
		try {

			if(a.getNumberOfValues(nCol) > 3) {
				double[] ret1 = new double[nCol * 57];
				a.addToEntryVectorized(ret1, 3, 3, 0, 3, 0, 2, 0, 3, 20, 1, 12, 2, 10, 3, 6, 56, nCol);
				double[] ret2 = new double[nCol * 57];
				b.addToEntryVectorized(ret2, 3, 3, 0, 3, 0, 2, 0, 3, 20, 1, 12, 2, 10, 3, 6, 56, nCol);
				assertTrue("Error: " + a.getClass().getSimpleName() + " " + b.getClass().getSimpleName(),
					Arrays.equals(ret1, ret2));
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	public void containsValueWithReference(double value, double[] reference) {
		assertEquals(//
			a.containsValueWithReference(value, reference), //
			b.containsValueWithReference(value, reference));
	}

	private static void compare(IDictionary a, IDictionary b, int nCol) {
		assertEquals(a.getNumberOfValues(nCol), b.getNumberOfValues(nCol));
		compare(a, b, a.getNumberOfValues(nCol), nCol);
	}

	private static void compare(IDictionary a, IDictionary b, int nRow, int nCol) {
		try {
			if(a == null && b == null)
				return;
			else if(a == null || b == null)
				fail("both outputs should be null if one is: \n" + a + " \n " + b);
			else {
				String errorM = a.getClass().getSimpleName() + " " + b.getClass().getSimpleName();
				for(int i = 0; i < nRow; i++)
					for(int j = 0; j < nCol; j++) {
						double aa = a.getValue(i, j, nCol);
						double bb = b.getValue(i, j, nCol);
						boolean eq = Math.abs(aa - bb) < 0.0001;
						if(!eq) {
							assertEquals(errorM + " cell:<" + i + "," + j + ">", a.getValue(i, j, nCol),
								b.getValue(i, j, nCol), 0.0001);
						}
					}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void preaggValuesFromDense() {
		try {

			final int nv = a.getNumberOfValues(nCol);
			IColIndex idc = ColIndexFactory.create(0, nCol);

			double[] bv = TestUtils.generateTestVector(nCol * nCol, -1, 1, 1.0, 321521L);

			IDictionary aa = a.preaggValuesFromDense(nv, idc, idc, bv, nCol);
			IDictionary bb = b.preaggValuesFromDense(nv, idc, idc, bv, nCol);

			compare(aa, bb, aa.getNumberOfValues(nCol), nCol);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void rightMMPreAggSparse() {
		final int nColsOut = 30;
		MatrixBlock sparse = TestUtils.generateTestMatrixBlock(1000, nColsOut, -10, 10, 0.1, 100);
		sparse = TestUtils.ceil(sparse);
		sparse.denseToSparse(true);
		SparseBlock sb = sparse.getSparseBlock();
		if(sb == null)
			throw new NotImplementedException();

		IColIndex agCols = new RangeIndex(nColsOut);
		IColIndex thisCols = new RangeIndex(0, nCol);

		int nVals = a.getNumberOfValues(nCol);
		try {

			IDictionary aa = a.rightMMPreAggSparse(nVals, sb, thisCols, agCols, nColsOut);
			IDictionary bb = b.rightMMPreAggSparse(nVals, sb, thisCols, agCols, nColsOut);
			compare(aa, bb, nColsOut);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

	}

	@Test
	public void rightMMPreAggSparse2() {
		final int nColsOut = 1000;
		MatrixBlock sparse = TestUtils.generateTestMatrixBlock(1000, nColsOut, -10, 10, 0.01, 100);
		sparse = TestUtils.ceil(sparse);
		sparse.denseToSparse(true);
		SparseBlock sb = sparse.getSparseBlock();
		if(sb == null)
			throw new NotImplementedException();

		IColIndex agCols = new RangeIndex(nColsOut);
		IColIndex thisCols = new RangeIndex(0, nCol);

		int nVals = a.getNumberOfValues(nCol);
		try {

			IDictionary aa = a.rightMMPreAggSparse(nVals, sb, thisCols, agCols, nColsOut);
			IDictionary bb = b.rightMMPreAggSparse(nVals, sb, thisCols, agCols, nColsOut);
			compare(aa, bb, nColsOut);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

	}

	@Test
	public void rightMMPreAggSparseDifferentColumns() {
		final int nColsOut = 3;
		MatrixBlock sparse = TestUtils.generateTestMatrixBlock(1000, 50, -10, 10, 0.1, 100);
		sparse = TestUtils.ceil(sparse);
		sparse.denseToSparse(true);
		SparseBlock sb = sparse.getSparseBlock();
		if(sb == null)
			throw new NotImplementedException();

		IColIndex agCols = new ArrayIndex(new int[] {4, 10, 38});
		IColIndex thisCols = new RangeIndex(0, nCol);

		int nVals = a.getNumberOfValues(nCol);
		try {

			IDictionary aa = a.rightMMPreAggSparse(nVals, sb, thisCols, agCols, 50);
			IDictionary bb = b.rightMMPreAggSparse(nVals, sb, thisCols, agCols, 50);
			compare(aa, bb, nColsOut);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

	}

	@Test
	public void MMDictScalingDense() {
		double[] left = TestUtils.ceil(TestUtils.generateTestVector(a.getNumberOfValues(nCol) * 3, -10, 10, 1.0, 3214));
		IColIndex rowsLeft = ColIndexFactory.createI(1, 2, 3);
		IColIndex colsRight = ColIndexFactory.create(0, nCol);
		int[] scaling = new int[a.getNumberOfValues(nCol)];
		for(int i = 0; i < a.getNumberOfValues(nCol); i++)
			scaling[i] = i + 1;

		try {

			MatrixBlock retA = new MatrixBlock(5, nCol, 0);
			retA.allocateDenseBlock();
			a.MMDictScalingDense(left, rowsLeft, colsRight, retA, scaling);

			MatrixBlock retB = new MatrixBlock(5, nCol, 0);
			retB.allocateDenseBlock();
			b.MMDictScalingDense(left, rowsLeft, colsRight, retB, scaling);

			TestUtils.compareMatricesBitAvgDistance(retA, retB, 10, 10);
		}
		catch(Exception e) {

			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void MMDictScalingDenseOffset() {
		double[] left = TestUtils.generateTestVector(a.getNumberOfValues(nCol) * 3, -10, 10, 1.0, 3214);
		IColIndex rowsLeft = ColIndexFactory.createI(1, 2, 3);
		IColIndex colsRight = ColIndexFactory.create(3, nCol + 3);
		int[] scaling = new int[a.getNumberOfValues(nCol)];
		for(int i = 0; i < a.getNumberOfValues(nCol); i++)
			scaling[i] = i;

		try {

			MatrixBlock retA = new MatrixBlock(5, nCol + 3, 0);
			retA.allocateDenseBlock();
			a.MMDictScalingDense(left, rowsLeft, colsRight, retA, scaling);

			MatrixBlock retB = new MatrixBlock(5, nCol + 3, 0);
			retB.allocateDenseBlock();
			b.MMDictScalingDense(left, rowsLeft, colsRight, retB, scaling);

			TestUtils.compareMatricesBitAvgDistance(retA, retB, 10, 10);
		}
		catch(Exception e) {

			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void MMDictDense() {
		double[] left = TestUtils.ceil(TestUtils.generateTestVector(a.getNumberOfValues(nCol) * 3, -10, 10, 1.0, 3214));
		IColIndex rowsLeft = ColIndexFactory.createI(1, 2, 3);
		IColIndex colsRight = ColIndexFactory.create(0, nCol);

		try {

			MatrixBlock retA = new MatrixBlock(5, nCol, 0);
			retA.allocateDenseBlock();
			a.MMDictDense(left, rowsLeft, colsRight, retA);

			MatrixBlock retB = new MatrixBlock(5, nCol, 0);
			retB.allocateDenseBlock();
			b.MMDictDense(left, rowsLeft, colsRight, retB);

			TestUtils.compareMatricesBitAvgDistance(retA, retB, 10, 10);
		}
		catch(Exception e) {

			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void MMDictDenseOffset() {
		double[] left = TestUtils.generateTestVector(a.getNumberOfValues(nCol) * 3, -10, 10, 1.0, 3214);
		IColIndex rowsLeft = ColIndexFactory.createI(1, 2, 3);
		IColIndex colsRight = ColIndexFactory.create(3, nCol + 3);

		try {

			MatrixBlock retA = new MatrixBlock(5, nCol + 3, 0);
			retA.allocateDenseBlock();
			a.MMDictDense(left, rowsLeft, colsRight, retA);

			MatrixBlock retB = new MatrixBlock(5, nCol + 3, 0);
			retB.allocateDenseBlock();
			b.MMDictDense(left, rowsLeft, colsRight, retB);

			TestUtils.compareMatricesBitAvgDistance(retA, retB, 10, 10);
		}
		catch(Exception e) {

			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void sumAllRowsToDouble() {
		double[] aa = a.sumAllRowsToDouble(nCol);
		double[] bb = b.sumAllRowsToDouble(nCol);
		TestUtils.compareMatrices(aa, bb, 0.001);
	}

	@Test
	public void sumAllRowsToDoubleWithDefault() {
		double[] def = TestUtils.generateTestVector(nCol, 1, 10, 1.0, 3215213);
		double[] aa = a.sumAllRowsToDoubleWithDefault(def);
		double[] bb = b.sumAllRowsToDoubleWithDefault(def);
		TestUtils.compareMatrices(aa, bb, 0.001);
	}

	@Test
	public void sumAllRowsToDoubleWithReference() {
		double[] def = TestUtils.generateTestVector(nCol, 1, 10, 1.0, 3215213);
		double[] aa = a.sumAllRowsToDoubleWithReference(def);
		double[] bb = b.sumAllRowsToDoubleWithReference(def);
		TestUtils.compareMatrices(aa, bb, 0.001);
	}

	@Test
	public void sumAllRowsToDoubleSq() {
		double[] aa = a.sumAllRowsToDoubleSq(nCol);
		double[] bb = b.sumAllRowsToDoubleSq(nCol);
		TestUtils.compareMatrices(aa, bb, 0.001);
	}

	@Test
	public void sumAllRowsToDoubleSqWithDefault() {
		double[] def = TestUtils.generateTestVector(nCol, 1, 10, 1.0, 3215213);
		double[] aa = a.sumAllRowsToDoubleSqWithDefault(def);
		double[] bb = b.sumAllRowsToDoubleSqWithDefault(def);
		TestUtils.compareMatrices(aa, bb, 0.001);
	}

	@Test
	public void sumAllRowsToDoubleSqWithReference() {
		double[] def = TestUtils.generateTestVector(nCol, 1, 10, 1.0, 3215213);
		double[] aa = a.sumAllRowsToDoubleSqWithReference(def);
		double[] bb = b.sumAllRowsToDoubleSqWithReference(def);
		TestUtils.compareMatrices(aa, bb, 0.001);
	}

	@Test
	public void aggColsMin() {
		IColIndex cols = ColIndexFactory.create(2, nCol + 2);
		Builtin m = Builtin.getBuiltinFnObject(BuiltinCode.MIN);

		double[] aa = new double[nCol + 3];
		a.aggregateCols(aa, m, cols);
		double[] bb = new double[nCol + 3];
		b.aggregateCols(bb, m, cols);

		TestUtils.compareMatrices(aa, bb, 0.001);
	}

	@Test
	public void aggColsMax() {
		IColIndex cols = ColIndexFactory.create(2, nCol + 2);
		Builtin m = Builtin.getBuiltinFnObject(BuiltinCode.MAX);

		double[] aa = new double[nCol + 3];
		a.aggregateCols(aa, m, cols);
		double[] bb = new double[nCol + 3];
		b.aggregateCols(bb, m, cols);

		TestUtils.compareMatrices(aa, bb, 0.001);
	}

	@Test
	public void getValue() {
		int nCell = nCol * a.getNumberOfValues(nCol);
		for(int i = 0; i < nCell; i++)
			assertEquals(a.getValue(i), b.getValue(i), 0.0000);
	}

	@Test
	public void colSum() {
		IColIndex cols = ColIndexFactory.create(2, nCol + 2);
		int[] counts = new int[a.getNumberOfValues(nCol)];
		for(int i = 0; i < counts.length; i++) {
			counts[i] = i + 1;
		}

		double[] aa = new double[nCol + 3];
		a.colSum(aa, counts, cols);
		double[] bb = new double[nCol + 3];
		b.colSum(bb, counts, cols);

		TestUtils.compareMatrices(aa, bb, 0.001);
	}

	@Test
	public void colProduct() {
		IColIndex cols = ColIndexFactory.create(2, nCol + 2);
		int[] counts = new int[a.getNumberOfValues(nCol)];
		for(int i = 0; i < counts.length; i++) {
			counts[i] = i + 1;
		}

		double[] aa = new double[nCol + 3];
		a.colProduct(aa, counts, cols);
		double[] bb = new double[nCol + 3];
		b.colProduct(bb, counts, cols);

		TestUtils.compareMatrices(aa, bb, 0.001);
	}

	public void productWithDefault(double retV, double[] def) {
		// Shared
		final int[] counts = getCounts(nRow, 1324);

		// A
		final double[] aRet = new double[] {retV};
		a.productWithDefault(aRet, counts, def, nCol);

		// B
		final double[] bRet = new double[] {retV};
		b.productWithDefault(bRet, counts, def, nCol);

		TestUtils.compareMatricesBitAvgDistance(//
			aRet, bRet, 10, 10, "Not Equivalent values from product");
	}

	private static int[] getCounts(int nRows, int seed) {
		int[] counts = new int[nRows];
		Random r = new Random(seed);
		for(int i = 0; i < nRows; i++)
			counts[i] = r.nextInt(100);
		return counts;
	}

	private static double[] getReference(int nCol, int seed, double min, double max) {
		double[] reference = new double[nCol];
		Random r = new Random(seed);
		double diff = max - min;
		if(diff == 0)
			for(int i = 0; i < nCol; i++)
				reference[i] = max;
		else
			for(int i = 0; i < nCol; i++)
				reference[i] = r.nextDouble() * diff - min;
		return reference;
	}

	@Test
	public void testSerialization() {
		try {
			// Serialize out
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			a.write(fos);

			// Serialize in
			ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
			DataInputStream fis = new DataInputStream(bis);

			IDictionary n = DictionaryFactory.read(fis);

			compare(a, n, nRow, nCol);
		}
		catch(IOException e) {
			throw new RuntimeException("Error in io", e);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void testSerializationB() {
		try {
			// Serialize out
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			b.write(fos);

			// Serialize in
			ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
			DataInputStream fis = new DataInputStream(bis);

			IDictionary n = DictionaryFactory.read(fis);

			compare(b, n, nRow, nCol);
		}
		catch(IOException e) {
			throw new RuntimeException("Error in io", e);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}
}
