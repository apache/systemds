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

package org.apache.sysds.test.component.compress.colgroup;

import static org.junit.Assert.fail;

import java.util.Arrays;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroupCompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupLinearFunctional;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
public class ColGroupLinearFunctionalTest extends ColGroupLinearFunctionalBase {
	protected static final Log LOG = LogFactory.getLog(ColGroupLinearFunctionalTest.class.getName());

	public ColGroupLinearFunctionalTest(AColGroup base, ColGroupLinearFunctional lin, AColGroup baseLeft,
		AColGroup cgLeft, int nRowLeft, int nColLeft, int nRowRight, int nColRight, ColGroupUncompressed cgRight,
		double tolerance) {
		super(base, lin, baseLeft, cgLeft, nRowLeft, nColLeft, nRowRight, nColRight, cgRight, tolerance);
	}

	@Test
	public void testContainsValue() {
		double[] linValues = getValues(lin);
		double[] baseValues = getValues(base);

		for(int i = 0; i < linValues.length; i++) {
			Assert.assertEquals("Base ColGroup and linear ColGroup must be initialized with the same values", linValues[i],
				baseValues[i], tolerance);
			if(!lin.containsValue(baseValues[i])) {
				// debug
				System.out.println(baseValues[i]);
				System.out.println(i);
				Assert.assertTrue(base.containsValue(baseValues[i]) && lin.containsValue(baseValues[i]));

			}
			Assert.assertTrue(base.containsValue(baseValues[i]) && lin.containsValue(baseValues[i]));
		}
	}

	@Test
	public void testTsmm() {
		int nCol = lin.getNumCols();

		final MatrixBlock resultUncompressed = new MatrixBlock(lin.getNumCols(), nCol, false);
		resultUncompressed.allocateDenseBlock();
		base.tsmm(resultUncompressed, nRow);

		final MatrixBlock resultCompressed = new MatrixBlock(nCol, nCol, false);
		resultCompressed.allocateDenseBlock();
		lin.tsmm(resultCompressed, nRow);

		Assert.assertArrayEquals(resultUncompressed.getDenseBlockValues(), resultCompressed.getDenseBlockValues(),
			tolerance);
	}

	@Test
	public void testRightMultByMatrix() {
		MatrixBlock mbtRight = cgRight.getData();

		AColGroup colGroupResultExpected = base.rightMultByMatrix(mbtRight);
		MatrixBlock resultExpected = ((ColGroupUncompressed) colGroupResultExpected).getData();
		AColGroup colGroupResult = lin.rightMultByMatrix(mbtRight);
		MatrixBlock result = ((ColGroupUncompressed) colGroupResult).getData();

		Assert.assertArrayEquals(resultExpected.getDenseBlockValues(), result.getDenseBlockValues(), tolerance);
	}

	@Test
	public void testLeftMultByAColGroup() {
		if(cgLeft.getCompType() == AColGroup.CompressionType.LinearFunctional)
			leftMultByAColGroup(true);
		else if(cgLeft.getCompType() == AColGroup.CompressionType.UNCOMPRESSED)
			leftMultByAColGroup(false);
		else
			fail("CompressionType not supported for leftMultByAColGrup");
	}

	public void leftMultByAColGroup(boolean compressedLeft) {
		final MatrixBlock result = new MatrixBlock(nRowLeft, nColRight, false);
		final MatrixBlock resultExpected = new MatrixBlock(nRowLeft, nColRight, false);
		result.allocateDenseBlock();
		resultExpected.allocateDenseBlock();

		base.leftMultByAColGroup(baseLeft, resultExpected, nRowLeft);
		lin.leftMultByAColGroup(cgLeft, result,nRowLeft);

		Assert.assertArrayEquals(resultExpected.getDenseBlockValues(), result.getDenseBlockValues(), tolerance);
	}

	@Test
	public void testColSumsSq() {
		double[] colSumsExpected = new double[base.getNumCols()];
		AggregateOperator aop = new AggregateOperator(0, KahanPlusSq.getKahanPlusSqFnObject());
		AggregateUnaryOperator auop = new AggregateUnaryOperator(aop, ReduceRow.getReduceRowFnObject());

		if(base instanceof AColGroupCompressed) {
			AColGroupCompressed baseComp = (AColGroupCompressed) base;
			baseComp.unaryAggregateOperations(auop, colSumsExpected, nRow, 0, nRow, baseComp.preAggRows(auop.aggOp.increOp.fn));
		}
		else if(base instanceof ColGroupUncompressed) {
			MatrixBlock mb = ((ColGroupUncompressed) base).getData();

			for(int j = 0; j < base.getNumCols(); j++) {
				double colSum = 0;
				for(int i = 0; i < nRow; i++) {
					colSum += Math.pow(mb.getDouble(i, j), 2);
				}
				colSumsExpected[j] = colSum;
			}
		}
		else {
			fail("Base ColGroup type does not support colSumSq.");
		}

		double[] colSums = new double[lin.getNumCols()];
		lin.unaryAggregateOperations(auop, colSums, nRow, 0, nRow, lin.preAggRows(auop.aggOp.increOp.fn));

		Assert.assertArrayEquals(colSumsExpected, colSums, tolerance);
	}

	@Test
	public void testProduct() {
		double[] productExpected = new double[] {1};

		AggregateOperator aop = new AggregateOperator(0, Multiply.getMultiplyFnObject());
		AggregateUnaryOperator auop = new AggregateUnaryOperator(aop, ReduceAll.getReduceAllFnObject());

		if(base instanceof AColGroupCompressed) {
			AColGroupCompressed baseComp = (AColGroupCompressed) base;
			baseComp.unaryAggregateOperations(auop, productExpected, nRow, 0, nRow, baseComp.preAggRows(auop.aggOp.increOp.fn));
		}
		else if(base instanceof ColGroupUncompressed) {
			MatrixBlock mb = ((ColGroupUncompressed) base).getData();

			for(int j = 0; j < base.getNumCols(); j++) {
				for(int i = 0; i < nRow; i++) {
					productExpected[0] *= mb.getDouble(i, j);
				}
			}
		}
		else {
			fail("Base ColGroup type does not support colProduct.");
		}

		double[] product = new double[] {1};
		lin.unaryAggregateOperations(auop, product, nRow, 0, nRow, lin.preAggRows(auop.aggOp.increOp.fn));

		// use relative tolerance since products can get very large
		double relTolerance = tolerance * Math.abs(productExpected[0]);
		Assert.assertEquals(productExpected[0], product[0], relTolerance);
	}

	@Test
	public void testMax() {
		Assert.assertEquals(base.getMax(), lin.getMax(), tolerance);
	}

	@Test
	public void testMin() {
		Assert.assertEquals(base.getMin(), lin.getMin(), tolerance);
	}

	@Test
	public void testColProducts() {
		double[] colProductsExpected = new double[base.getNumCols()];

		AggregateOperator aop = new AggregateOperator(0, Multiply.getMultiplyFnObject());
		AggregateUnaryOperator auop = new AggregateUnaryOperator(aop, ReduceRow.getReduceRowFnObject());

		if(base instanceof AColGroupCompressed) {
			AColGroupCompressed baseComp = (AColGroupCompressed) base;
			baseComp.unaryAggregateOperations(auop, colProductsExpected, nRow, 0, nRow, baseComp.preAggRows(auop.aggOp.increOp.fn));
		}
		else if(base instanceof ColGroupUncompressed) {
			MatrixBlock mb = ((ColGroupUncompressed) base).getData();

			for(int j = 0; j < base.getNumCols(); j++) {
				double colProduct = 1;
				for(int i = 0; i < nRow; i++) {
					colProduct *= mb.getDouble(i, j);
				}
				colProductsExpected[j] = colProduct;
			}
		}
		else {
			fail("Base ColGroup type does not support colProduct.");
		}

		double[] colProducts = new double[base.getNumCols()];
		for(int j = 0; j < base.getNumCols(); j++) {
			colProducts[j] = 1;
		}

		lin.unaryAggregateOperations(auop, colProducts, nRow, 0, nRow, lin.preAggRows(auop.aggOp.increOp.fn));

		// use relative tolerance since column products can get very large
		double relTolerance = tolerance * Math.abs(Arrays.stream(colProductsExpected).max().orElse(0));
		Assert.assertArrayEquals(colProductsExpected, colProducts, relTolerance);
	}

	@Test
	public void testSumSq() {
		double[] sumSqExpected = new double[] {0};

		AggregateOperator aop = new AggregateOperator(0, KahanPlusSq.getKahanPlusSqFnObject());
		AggregateUnaryOperator auop = new AggregateUnaryOperator(aop, ReduceAll.getReduceAllFnObject());

		if(base instanceof AColGroupCompressed) {
			AColGroupCompressed baseComp = (AColGroupCompressed) base;
			baseComp.unaryAggregateOperations(auop, sumSqExpected, nRow, 0, nRow, baseComp.preAggRows(auop.aggOp.increOp.fn));
		}
		else if(base instanceof ColGroupUncompressed) {
			MatrixBlock mb = ((ColGroupUncompressed) base).getData();

			for(int j = 0; j < base.getNumCols(); j++) {
				for(int i = 0; i < nRow; i++) {
					sumSqExpected[0] += Math.pow(mb.getDouble(i, j), 2);
				}
			}
		}
		else {
			fail("Base ColGroup type does not support sumSq.");
		}

		double[] sumSq = new double[] {0};
		lin.unaryAggregateOperations(auop, sumSq, nRow, 0, nRow, lin.preAggRows(auop.aggOp.increOp.fn));

		Assert.assertEquals(sumSqExpected[0], sumSq[0], tolerance);
	}

	@Test
	public void testSum() {
		double[] colSums = new double[base.getNumCols()];
		base.computeColSums(colSums, nRow);
		double sumExpected = Arrays.stream(colSums).sum();

		double[] sum = new double[1];
		AggregateOperator aop = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateUnaryOperator auop = new AggregateUnaryOperator(aop, ReduceAll.getReduceAllFnObject());
		lin.unaryAggregateOperations(auop, sum, nRow, 0, nRow, lin.preAggRows(auop.aggOp.increOp.fn));

		Assert.assertEquals(sumExpected, sum[0], tolerance);
	}

	@Test
	public void testRowSums() {
		double[] rowSumsExpected = new double[nRow];

		AggregateOperator aop = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateUnaryOperator auop = new AggregateUnaryOperator(aop, ReduceCol.getReduceColFnObject());

		if(base instanceof AColGroupCompressed) {
			AColGroupCompressed baseComp = (AColGroupCompressed) base;
			baseComp.unaryAggregateOperations(auop, rowSumsExpected, nRow, 0, nRow, baseComp.preAggRows(auop.aggOp.increOp.fn));
		}
		else if(base instanceof ColGroupUncompressed) {
			MatrixBlock mb = ((ColGroupUncompressed) base).getData();

			for(int i = 0; i < nRow; i++) {
				double rowSum = 0;
				for(int j = 0; j < base.getNumCols(); j++) {
					rowSum += mb.getDouble(i, j);
				}
				rowSumsExpected[i] = rowSum;
			}
		}
		else {
			fail("Base ColGroup type does not support rowSum.");
		}

		double[] rowSums = new double[nRow];
		lin.unaryAggregateOperations(auop, rowSums, nRow, 0, nRow, lin.preAggRows(auop.aggOp.increOp.fn));

		Assert.assertArrayEquals(rowSumsExpected, rowSums, tolerance);
	}

	@Test
	public void testColSums() {
		double[] colSumsExpected = new double[base.getNumCols()];
		double[] colSums = new double[base.getNumCols()];
		base.computeColSums(colSumsExpected, nRow);
		lin.computeColSums(colSums, nRow);

		Assert.assertArrayEquals(colSumsExpected, colSums, tolerance);
	}

	@Test
	public void testColumnGroupConstruction() {
		double[][] constColumn = new double[][] {{1, 1, 1, 1, 1}};
		AColGroup cgConst = cgLinCompressed(constColumn, true);
		Assert.assertSame(AColGroup.CompressionType.CONST, cgConst.getCompType());

		double[][] zeroColumn = new double[][] {{0, 0, 0, 0, 0}};
		AColGroup cgEmpty = cgLinCompressed(zeroColumn, true);
		Assert.assertSame(AColGroup.CompressionType.EMPTY, cgEmpty.getCompType());
	}

	@Test
	public void testDecompressToDenseBlock() {
		MatrixBlock ret = new MatrixBlock(nRow, lin.getNumCols(), false);
		ret.allocateDenseBlock();
		lin.decompressToDenseBlock(ret.getDenseBlock(), 0, nRow);

		MatrixBlock expected = new MatrixBlock(nRow, lin.getNumCols(), false);
		expected.allocateDenseBlock();
		base.decompressToDenseBlock(expected.getDenseBlock(), 0, nRow);

		Assert.assertArrayEquals(expected.getDenseBlockValues(), ret.getDenseBlockValues(), tolerance);
	}

}
