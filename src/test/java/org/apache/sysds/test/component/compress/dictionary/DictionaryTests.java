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
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
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
	private final ADictionary a;
	private final ADictionary b;

	public DictionaryTests(ADictionary a, ADictionary b, int nRow, int nCol) {
		this.nRow = nRow;
		this.nCol = nCol;
		this.a = a;
		this.b = b;
	}

	@Parameters
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();

		try {
			addAll(tests, new double[] {1, 2, 3, 4, 5}, 1);
			addAll(tests, new double[] {1, 2, 3, 4, 5, 6}, 2);
			addAll(tests, new double[] {1, 2.2, 3.3, 4.4, 5.5, 6.6}, 3);

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
		for(int i = 0; i < nCol; i++)
			reference[i] = r.nextDouble() * diff - min;
		return reference;
	}
}
