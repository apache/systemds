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

package org.apache.sysds.test.component.compress;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Equals;
import org.apache.sysds.runtime.functionobjects.GreaterThan;
import org.apache.sysds.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysds.runtime.functionobjects.LessThan;
import org.apache.sysds.runtime.functionobjects.LessThanEquals;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.Power2;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.functionobjects.Xor;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.TestConstants.MatrixTypology;
import org.apache.sysds.test.component.compress.TestConstants.OverLapping;
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/**
 * This test class is for the tests that are verified in the large tests, but not fully verified in the api.
 */
@RunWith(value = Parameterized.class)
public class ExtendedMatrixTests extends CompressedTestBase {

	protected static CompressionSettingsBuilder[] usedCompressionSettings = new CompressionSettingsBuilder[] {
		csb().setTransposeInput("true"), csb().setTransposeInput("false")};

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		// for(SparsityType st : usedSparsityTypes)
		SparsityType st = SparsityType.FULL;
		ValueType vt = ValueType.RLE_COMPRESSIBLE;
		ValueRange vr = ValueRange.SMALL;
		MatrixTypology mt = MatrixTypology.LARGE;
		OverLapping ov = OverLapping.NONE;

		// empty matrix compression ... (technically not a compressed matrix.)
		tests.add(new Object[] {SparsityType.EMPTY, ValueType.RAND, vr, csb(), mt, ov, 1, null});

		for(CompressionSettingsBuilder cs : usedCompressionSettings)
			tests.add(new Object[] {st, vt, vr, cs, mt, ov, 1, null});

		ov = OverLapping.PLUS_ROW_VECTOR;
		for(CompressionSettingsBuilder cs : usedCompressionSettings)
			tests.add(new Object[] {st, vt, vr, cs, mt, ov, 1, null});

		st = SparsityType.ULTRA_SPARSE;
		mt = MatrixTypology.COL_16;
		CompressionSettingsBuilder sb = csb().setTransposeInput("true");
		tests.add(new Object[] {st, vt, vr, sb, mt, ov, 1, null});

		tests.add(new Object[] {st, vt, vr, sb, mt, ov, 10, null});

		return tests;
	}

	private final MatrixBlock vectorCols;
	private final MatrixBlock matrixRowsCols;

	public ExtendedMatrixTests(SparsityType sparType, ValueType valType, ValueRange valueRange,
		CompressionSettingsBuilder compSettings, MatrixTypology MatrixTypology, OverLapping ov, int parallelism,
		Collection<CompressionType> ct) {
		super(sparType, valType, valueRange, compSettings, MatrixTypology, ov, parallelism, ct, null);

		if(cmb instanceof CompressedMatrixBlock) {

			vectorCols = TestUtils.generateTestMatrixBlock(1, cols, -1.0, 1.5, 1.0, 3);
			matrixRowsCols = TestUtils.generateTestMatrixBlock(rows, cols, -1.0, 1.5, 1.0, 3);
		}
		else {
			vectorCols = null;
			matrixRowsCols = null;
		}

	}

	@Test
	public void testMin() {
		if(!(cmb instanceof CompressedMatrixBlock))
			return;
		double ret1 = cmb.min();
		double ret2 = mb.min();
		if(_cs != null && (_cs.lossy || overlappingType == OverLapping.SQUASH))
			assertTrue(bufferedToString, TestUtils.compareCellValue(ret2, ret1, lossyTolerance, false));
		else if(OverLapping.effectOnOutput(overlappingType))
			assertTrue(bufferedToString, TestUtils.getPercentDistance(ret2, ret1, true) > .99);
		else
			TestUtils.compareScalarBitsJUnit(ret2, ret1, 3, bufferedToString); // Should be exactly same value

	}

	@Test
	public void testMax() {
		if(!(cmb instanceof CompressedMatrixBlock))
			return;
		double ret1 = cmb.max();
		double ret2 = mb.max();
		if(_cs != null && (_cs.lossy || overlappingType == OverLapping.SQUASH))
			assertTrue(bufferedToString, TestUtils.compareCellValue(ret2, ret1, lossyTolerance, false));
		else if(OverLapping.effectOnOutput(overlappingType))
			assertTrue(bufferedToString, TestUtils.getPercentDistance(ret2, ret1, true) > .99);
		else
			TestUtils.compareScalarBitsJUnit(ret2, ret1, 3, bufferedToString); // Should be exactly same value

	}

	@Test
	public void testSum() {
		if(!(cmb instanceof CompressedMatrixBlock))
			return;
		double ret1 = cmb.sum();
		double ret2 = mb.sum();
		if(_cs != null && (_cs.lossy || overlappingType == OverLapping.SQUASH))
			assertTrue(bufferedToString, TestUtils.compareCellValue(ret2, ret1, lossyTolerance, false));
		else if(OverLapping.effectOnOutput(overlappingType))
			assertTrue(bufferedToString, TestUtils.getPercentDistance(ret2, ret1, true) > .99);
		else
			TestUtils.compareScalarBitsJUnit(ret2, ret1, 100, bufferedToString); // Should be exactly same value

	}

	@Test
	public void testSumSq() {
		if(!(cmb instanceof CompressedMatrixBlock))
			return;
		double ret1 = cmb.sumSq();
		double ret2 = mb.sumSq();
		if(_cs != null && (_cs.lossy || overlappingType == OverLapping.SQUASH))
			assertTrue(bufferedToString, TestUtils.compareCellValue(ret2, ret1, lossyTolerance, false));
		else if(OverLapping.effectOnOutput(overlappingType))
			assertTrue(bufferedToString, TestUtils.getPercentDistance(ret2, ret1, true) > .99);
		else
			TestUtils.compareScalarBitsJUnit(ret2, ret1, 128, bufferedToString); // Should be exactly same value
	}

	@Test
	public void testMean() {
		if(!(cmb instanceof CompressedMatrixBlock))
			return;
		double ret1 = cmb.mean();
		double ret2 = mb.mean();
		if(_cs != null && (_cs.lossy || overlappingType == OverLapping.SQUASH))
			assertTrue(bufferedToString, TestUtils.compareCellValue(ret2, ret1, lossyTolerance, false));
		else if(OverLapping.effectOnOutput(overlappingType))
			assertTrue(bufferedToString, TestUtils.getPercentDistance(ret2, ret1, true) > .99);
		else
			TestUtils.compareScalarBitsJUnit(ret2, ret1, 10, bufferedToString); // Should be exactly same value
	}

	@Test
	public void testProd() {
		try {

			if(!(cmb instanceof CompressedMatrixBlock))
				return;
			double ret1 = cmb.prod();
			LOG.error(ret1);
			LOG.error(cmb);
			double ret2 = mb.prod();
			boolean res;
			if(_cs != null && (_cs.lossy || overlappingType == OverLapping.SQUASH))
				res = TestUtils.compareCellValue(ret2, ret1, lossyTolerance, false);
			else if(OverLapping.effectOnOutput(overlappingType))
				res = TestUtils.getPercentDistance(ret2, ret1, true) > .99;
			else {
				TestUtils.compareScalarBitsJUnit(ret2, ret1, 10, bufferedToString); // Should be exactly same value
				res = true;
			}
			if(!res) {
				fail(bufferedToString + "\n" + ret1 + " vs " + ret2);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(bufferedToString + e);
		}

	}

	@Test
	public void testColSum() {
		if(!(cmb instanceof CompressedMatrixBlock))
			return;
		MatrixBlock ret1 = cmb.colSum();
		MatrixBlock ret2 = mb.colSum();
		compareResultMatrices(ret1, ret2, 1);
	}

	@Test
	public void testToString() {
		if(!(cmb instanceof CompressedMatrixBlock) || rows * cols > 10000)
			return;
		String st = cmb.toString();
		assertTrue(st.contains("CompressedMatrixBlock"));
	}

	@Test
	public void testCompressionStatisticsToString() {
		try {
			if(cmbStats != null) {
				String st = cmbStats.toString();
				assertTrue(st.contains("CompressionStatistics"));
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException(e);
		}
	}

	@Test
	public void testScalarRightOpSubtract() {
		double addValue = 15;
		ScalarOperator sop = new RightScalarOperator(Minus.getMinusFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarRightOpLess() {
		double addValue = 0.11;
		ScalarOperator sop = new RightScalarOperator(LessThan.getLessThanFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarRightOpLessThanEqual() {
		double addValue = -50;
		ScalarOperator sop = new RightScalarOperator(LessThanEquals.getLessThanEqualsFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarRightOpGreater() {
		double addValue = 0.11;
		ScalarOperator sop = new RightScalarOperator(GreaterThan.getGreaterThanFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarRightOpEquals() {
		double addValue = 1.0;
		ScalarOperator sop = new RightScalarOperator(Equals.getEqualsFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarRightOpPower2() {
		double addValue = 2;
		ScalarOperator sop = new RightScalarOperator(Power2.getPower2FnObject(), addValue);
		testScalarOperations(sop, lossyTolerance * 2);
	}

	@Test
	public void testScalarOpLeftMultiplyPositive() {
		double mult = 7;
		ScalarOperator sop = new LeftScalarOperator(Multiply.getMultiplyFnObject(), mult, _k);
		testScalarOperations(sop, lossyTolerance * 7);
	}

	@Test
	public void testScalarOpLeftMultiplyNegative() {
		double mult = -7;
		ScalarOperator sop = new LeftScalarOperator(Multiply.getMultiplyFnObject(), mult, _k);
		testScalarOperations(sop, lossyTolerance * 7);
	}

	@Test
	public void testScalarLeftOpAddition() {
		double addValue = 4;
		ScalarOperator sop = new LeftScalarOperator(Plus.getPlusFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.05);
	}

	@Test
	@Ignore // this is apparently rewritten in dml
	public void testScalarLeftOpSubtract() {
		double addValue = 15;
		ScalarOperator sop = new LeftScalarOperator(Minus.getMinusFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarLeftOpLess() {
		double addValue = 0.11;
		ScalarOperator sop = new LeftScalarOperator(LessThan.getLessThanFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarLeftOpLessSmallValue() {
		double addValue = -1000000.11;
		ScalarOperator sop = new LeftScalarOperator(LessThanEquals.getLessThanEqualsFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarLeftOpEqual() {
		double addValue = 1.0;
		ScalarOperator sop = new LeftScalarOperator(Equals.getEqualsFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	@Ignore
	// Currently ignored because of division with zero.
	public void testScalarLeftOpDivide() {
		double addValue = 14.0;
		ScalarOperator sop = new LeftScalarOperator(Divide.getDivideFnObject(), addValue);
		testScalarOperations(sop, (lossyTolerance + 0.1) * 10);
	}

	@Test
	public void testScalarLeftOpLessThanEqualSmallValue() {
		double addValue = -1000000.11;
		ScalarOperator sop = new LeftScalarOperator(LessThanEquals.getLessThanEqualsFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarLeftOpGreaterThanEqualsSmallValue() {
		double addValue = -1001310000.11;
		ScalarOperator sop = new LeftScalarOperator(GreaterThanEquals.getGreaterThanEqualsFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarLeftOpGreaterThanLargeValue() {
		double addValue = 10132400000.11;
		ScalarOperator sop = new LeftScalarOperator(GreaterThan.getGreaterThanFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testBinaryMVMinusROW() {
		ValueFunction vf = Minus.getMinusFnObject();
		testBinaryMV(vf, vectorCols);
	}

	@Test
	public void testBinaryMVXorROW() {
		ValueFunction vf = Xor.getXorFnObject();
		testBinaryMV(vf, vectorCols);
	}

	@Test
	@Ignore
	public void testBinaryMMDivideLeft_Dense() {
		ValueFunction vf = Divide.getDivideFnObject();
		testBinaryMV(vf, matrixRowsCols, false);
	}

	@Test
	@Ignore
	public void testBinaryMMDivideLeft_Sparse() {
		ValueFunction vf = Divide.getDivideFnObject();
		testBinaryMV(vf, matrixRowsCols, false);
	}

	@Test
	@Ignore
	public void testBinaryMMMinusLeft_Dense() {
		ValueFunction vf = Minus.getMinusFnObject();
		testBinaryMV(vf, matrixRowsCols, false);
	}

	@Test
	@Ignore
	public void testBinaryMMMinusLeft_Sparse() {
		ValueFunction vf = Minus.getMinusFnObject();
		testBinaryMV(vf, matrixRowsCols, false);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testSliceInvalid_01() {
		if(!(cmb instanceof CompressedMatrixBlock))
			throw new DMLRuntimeException("Not Compressed Input");
		testSliceWithException(-1, 0, 0, 0);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testSliceInvalid_02() {
		if(!(cmb instanceof CompressedMatrixBlock))
			throw new DMLRuntimeException("Not Compressed Input");
		testSliceWithException(rows, rows, 0, 0);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testSliceInvalid_03() {
		if(!(cmb instanceof CompressedMatrixBlock))
			throw new DMLRuntimeException("Not Compressed Input");
		testSliceWithException(0, 0, cols, cols);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testSliceInvalid_04() {
		if(!(cmb instanceof CompressedMatrixBlock))
			throw new DMLRuntimeException("Not Compressed Input");
		testSliceWithException(0, 0, -1, 0);
	}

	public void testSliceWithException(int rl, int ru, int cl, int cu) {
		try {
			cmb.slice(rl, ru, cl, cu);
		}
		catch(Exception e) {
			throw new DMLRuntimeException(e);
		}
	}

	@Test
	public void unaryIsNATest() {
		unaryOperations(new UnaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.ISNA)));
	}

	@Test
	public void unaryIsINFTest() {
		unaryOperations(new UnaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.ISINF)));
	}

	@Test
	public void unaryABSTest() {
		unaryOperations(new UnaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.ABS)));
	}

	@Test
	public void unaryEXPTest() {
		unaryOperations(new UnaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.EXP)));
	}

	@Test
	public void testMatrixVectorMult01() {
		testMatrixVectorMult(1.0, 1.1);
	}

	@Test
	public void testMatrixVectorMult02() {
		testMatrixVectorMult(0.7, 1.0);
	}

	@Test
	public void testMatrixVectorMult04() {
		testMatrixVectorMult(1.0, 5.0);
	}

	@Test
	public void testLeftMatrixMatrixMultMedium() {
		MatrixBlock matrix = TestUtils.generateTestMatrixBlock(50, rows, 0.9, 1.5, 1.0, 3);
		testLeftMatrixMatrix(matrix);
	}

	@Test
	public void testCompactEmptyBlock() {
		if(cmb instanceof CompressedMatrixBlock) {
			cmb.compactEmptyBlock();
			if(cmb.isEmpty()) {
				CompressedMatrixBlock cm = (CompressedMatrixBlock) cmb;
				assertTrue(null == cm.getSoftReferenceToDecompressed());
			}
		}
	}

}
