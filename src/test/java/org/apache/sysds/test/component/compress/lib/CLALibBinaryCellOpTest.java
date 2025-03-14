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

package org.apache.sysds.test.component.compress.lib;

import static org.junit.Assert.fail;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.lib.CLALibBinaryCellOp;
import org.apache.sysds.runtime.functionobjects.And;
import org.apache.sysds.runtime.functionobjects.BitwAnd;
import org.apache.sysds.runtime.functionobjects.BitwOr;
import org.apache.sysds.runtime.functionobjects.BitwShiftL;
import org.apache.sysds.runtime.functionobjects.BitwShiftR;
import org.apache.sysds.runtime.functionobjects.BitwXor;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Equals;
import org.apache.sysds.runtime.functionobjects.GreaterThan;
import org.apache.sysds.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysds.runtime.functionobjects.IntegerDivide;
import org.apache.sysds.runtime.functionobjects.LessThan;
import org.apache.sysds.runtime.functionobjects.LessThanEquals;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Minus1Multiply;
import org.apache.sysds.runtime.functionobjects.MinusMultiply;
import org.apache.sysds.runtime.functionobjects.MinusNz;
import org.apache.sysds.runtime.functionobjects.Modulus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.NotEquals;
import org.apache.sysds.runtime.functionobjects.Or;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.PlusMultiply;
import org.apache.sysds.runtime.functionobjects.Power;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.functionobjects.Xor;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class CLALibBinaryCellOpTest {
	protected static final Log LOG = LogFactory.getLog(CombineGroupsTest.class.getName());

	public final static ValueFunction[] vf = {//
		(Plus.getPlusFnObject()), //
		(Minus.getMinusFnObject()), //
		Divide.getDivideFnObject(), //
		(Or.getOrFnObject()), //
		(LessThan.getLessThanFnObject()), //
		(LessThanEquals.getLessThanEqualsFnObject()), //
		(GreaterThan.getGreaterThanFnObject()), //
		(GreaterThanEquals.getGreaterThanEqualsFnObject()), //
		(Multiply.getMultiplyFnObject()), //
		(Modulus.getFnObject()), //
		(IntegerDivide.getFnObject()), //
		(Equals.getEqualsFnObject()), //
		(NotEquals.getNotEqualsFnObject()), //
		(And.getAndFnObject()), //
		(Xor.getXorFnObject()), //
		(BitwAnd.getBitwAndFnObject()), //
		(BitwOr.getBitwOrFnObject()), //
		(BitwXor.getBitwXorFnObject()), //
		(BitwShiftL.getBitwShiftLFnObject()), //
		(BitwShiftR.getBitwShiftRFnObject()), //
		// TODO: power fails currently in some cases
		//(Power.getPowerFnObject()), //
		(MinusNz.getMinusNzFnObject()), //
		(new PlusMultiply(32)), //
		(new PlusMultiply(2)), //
		(new PlusMultiply(0)), //
		(new MinusMultiply(32)), //
		Minus1Multiply.getMinus1MultiplyFnObject(),

		// // Builtin
		(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.MIN)), //
		(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.MAX)), //
		(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.LOG)), //
		(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.LOG_NZ)), //
		(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.MAXINDEX)), //
		(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.MININDEX)), //
		(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.CUMMAX)), //
		(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.CUMMIN)),//
		};

	private final MatrixBlock mb;
	private final CompressedMatrixBlock cmb;

	private final BinaryOperator op;
	private final MatrixBlock mb2;
	private final MatrixBlock mcv2;
	private final MatrixBlock mrv2;
	private final MatrixBlock mScalar2;
	private final CompressedMatrixBlock cmb2;
	private final CompressedMatrixBlock cmb2v2;

	public CLALibBinaryCellOpTest(BinaryOperator op, String s, MatrixBlock mb, CompressedMatrixBlock cmb,
		MatrixBlock mb2, MatrixBlock mcv2, MatrixBlock mrv2, MatrixBlock mScalar2, CompressedMatrixBlock cmb2,
		CompressedMatrixBlock cmb2v2) {
		this.op = op;
		this.mb = mb;
		this.cmb = cmb;
		this.mb2 = mb2;
		this.mcv2 = mcv2;
		this.mrv2 = mrv2;
		this.mScalar2 = mScalar2;
		this.cmb2 = cmb2;
		this.cmb2v2 = cmb2v2;
		Thread.currentThread().setName("main_test_" + Thread.currentThread().getId());

	}

	@Parameters(name = "{0}_{1}")
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();
		MatrixBlock mb;
		CompressedMatrixBlock cmb;

		try {
			mb = TestUtils.generateTestMatrixBlock(200, 50, -10, 10, 1.0, 32);
			mb = TestUtils.round(mb);
			cmb = (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(mb, 1).getLeft();
			genTests(tests, mb, cmb, "Normal");

			MatrixBlock mb2 = new MatrixBlock(10, 10, 132.0);
			CompressedMatrixBlock cmb2 = CompressedMatrixBlockFactory.createConstant(10, 10, 132.0);
			genTests(tests, mb2, cmb2, "Const");

			List<AColGroup> gs = new ArrayList<>();
			gs.add(ColGroupConst.create(ColIndexFactory.create(10), 100.0));
			gs.add(ColGroupConst.create(ColIndexFactory.create(10), 32.0));
			CompressedMatrixBlock cmb3 = new CompressedMatrixBlock(10, 10, 100, true, gs);
			genTests(tests, mb2, cmb3, "OverlappingConst");

			mb = TestUtils.generateTestMatrixBlock(200, 16, -10, 10, 0.04, 32);
			mb = TestUtils.round(mb);
			cmb = (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(mb, 1).getLeft();
			genTests(tests, mb, cmb, "sparse");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	private static void genTests(List<Object[]> tests, MatrixBlock mb, MatrixBlock cmb, String version) {

		MatrixBlock tmp;

		tmp = TestUtils.generateTestMatrixBlock(cmb.getNumRows(), cmb.getNumColumns(), -10, 10, 0.9, 132);
		MatrixBlock mb2 = TestUtils.round(tmp);

		tmp = TestUtils.generateTestMatrixBlock(cmb.getNumRows(), cmb.getNumColumns(), -10, 10, 0.05, 132);
		MatrixBlock mb2_sparse = TestUtils.round(tmp);
		MatrixBlock mb2_empty = new MatrixBlock(cmb.getNumRows(), cmb.getNumColumns(), 0.0);

		tmp = TestUtils.generateTestMatrixBlock(1, cmb.getNumColumns(), -10, 10, 0.9, 32);
		MatrixBlock mrv2 = TestUtils.round(tmp);

		tmp = TestUtils.generateTestMatrixBlock(1, cmb.getNumColumns(), -10, 10, 0.05, 32);
		MatrixBlock mrv2_sparse = TestUtils.round(tmp);
		MatrixBlock mrv2_empty = new MatrixBlock(1, cmb.getNumColumns(), 0.0);

		tmp = TestUtils.generateTestMatrixBlock(cmb.getNumRows(), 1, -10, 10, 0.9, 32);
		MatrixBlock mcv2 = TestUtils.round(tmp);

		tmp = TestUtils.generateTestMatrixBlock(cmb.getNumRows(), 1, -10, 10, 0.05, 32);
		MatrixBlock mcv2_sparse = TestUtils.round(tmp);
		MatrixBlock mcv2_empty = new MatrixBlock(cmb.getNumRows(), 1, 0.0);

		tmp = TestUtils.generateTestMatrixBlock(1, 1, 5, 10, 1.0, 32);
		MatrixBlock mScalar2 = TestUtils.round(tmp);
		tmp = TestUtils.generateTestMatrixBlock(cmb.getNumRows(), cmb.getNumColumns(), 8, 9, 1.0, 32);
		tmp = TestUtils.round(tmp);
		MatrixBlock cmb2 = CompressedMatrixBlockFactory.compress(tmp, 1).getLeft();
		if(!(cmb2 instanceof CompressedMatrixBlock))
			cmb2 = CompressedMatrixBlockFactory.createConstant(cmb.getNumRows(), cmb.getNumColumns(), 2);

		tmp = TestUtils.generateTestMatrixBlock(cmb.getNumRows(), cmb.getNumColumns(), 0, 30, 1.0, 32);
		tmp = TestUtils.round(tmp);
		MatrixBlock cmb2v2 = CompressedMatrixBlockFactory.compress(tmp, 1).getLeft();

		if(!(cmb2v2 instanceof CompressedMatrixBlock))
			cmb2v2 = CompressedMatrixBlockFactory.createConstant(cmb.getNumRows(), cmb.getNumColumns(), 3214.0);

		for(ValueFunction v : vf) {
			tests.add(new Object[] {new BinaryOperator(v), version + "_dense", //
				mb, cmb, mb2, mcv2, mrv2, mScalar2, cmb2, cmb2v2});
			tests.add(new Object[] {new BinaryOperator(v, 2), version + "_dense", //
				mb, cmb, mb2, mcv2, mrv2, mScalar2, cmb2, cmb2v2});
			tests.add(new Object[] {new BinaryOperator(v), version + "_sparse", //
				mb, cmb, mb2_sparse, mcv2_sparse, mrv2_sparse, null, null, null});
			tests.add(new Object[] {new BinaryOperator(v, 2), version + "_sparse", //
				mb, cmb, mb2_sparse, mcv2_sparse, mrv2_sparse, null, null, null});
			tests.add(new Object[] {new BinaryOperator(v), version + "_empty", //
				mb, cmb, mb2_empty, mcv2_empty, mrv2_empty, null, null, null});
			tests.add(new Object[] {new BinaryOperator(v, 2), version + "_empty", //
				mb, cmb, mb2_empty, mcv2_empty, mrv2_empty, null, null, null});
		}
	}

	@Test
	public void binRightMM() {
		try {
			exec(op, mb, cmb, mb2);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void binRightMM_noCache() {
		try {
			CompressedMatrixBlock spy = spy(cmb);
			when(spy.getCachedDecompressed()).thenReturn(null);
			exec(op, mb, spy, mb2);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void binRightMM_noCache_overlapping() {
		try {
			CompressedMatrixBlock spy = spy(cmb);
			when(spy.getCachedDecompressed()).thenReturn(null);
			when(spy.isOverlapping()).thenReturn(true);
			exec(op, mb, spy, mb2);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void binRightM_CMB() {
		try {
			exec(op, mb, cmb, cmb2);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void binRightM_CMB_V2() {
		try {
			exec(op, mb, cmb, cmb2v2);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void binRightM_CMB_noCache() {
		try {
			if(cmb2 == null)
				return;
			CompressedMatrixBlock spy = spy(cmb2);
			when(spy.getCachedDecompressed()).thenReturn(null);
			exec(op, mb, cmb, spy);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void binRightM_CMB_noCache_overlapping() {
		try {
			if(cmb2 == null)
				return;
			CompressedMatrixBlock spy = spy(cmb2);
			when(spy.getCachedDecompressed()).thenReturn(null);
			when(spy.isOverlapping()).thenReturn(true);
			exec(op, mb, cmb, spy);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void binRightM_CMB_V2_noCache() {
		try {

			if(cmb2v2 == null)
				return;
			CompressedMatrixBlock spy = spy(cmb2v2);
			when(spy.getCachedDecompressed()).thenReturn(null);
			exec(op, mb, cmb, spy);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void binRightM_CMB_V2_noCache_Overlapping() {
		try {

			if(cmb2v2 == null)
				return;
			CompressedMatrixBlock spy = spy(cmb2v2);
			when(spy.getCachedDecompressed()).thenReturn(null);
			when(spy.isOverlapping()).thenReturn(true);
			exec(op, mb, cmb, spy);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void binRightMrV() {
		try {

			exec(op, mb, cmb, mrv2);
			if(op.fn instanceof Power) // make sure that we cover the dense positive case of power
				exec(op, mb, cmb, TestUtils.floor(TestUtils.generateTestMatrixBlock(1, mb.getNumColumns(), 1, 10, 1.0, 13)));

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void binRightMrV_noCache() {
		try {
			CompressedMatrixBlock spy = spy(cmb);
			when(spy.getCachedDecompressed()).thenReturn(null);
			exec(op, mb, spy, mrv2);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void binRightMrV_noCache_overlapping() {
		try {

			CompressedMatrixBlock spy = spy(cmb);
			when(spy.getCachedDecompressed()).thenReturn(null);
			when(spy.isOverlapping()).thenReturn(true);
			exec(op, mb, spy, mrv2);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void binRightMcV() {
		try {

			exec(op, mb, cmb, mcv2);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void binRightMcV_noCache() {
		try {

			CompressedMatrixBlock spy = spy(cmb);
			when(spy.getCachedDecompressed()).thenReturn(null);
			exec(op, mb, spy, mcv2);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void binRightMcV_noCache_overlapping() {
		try {

			CompressedMatrixBlock spy = spy(cmb);
			when(spy.getCachedDecompressed()).thenReturn(null);
			when(spy.isOverlapping()).thenReturn(true);
			exec(op, mb, spy, mcv2);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test(expected = Exception.class)
	public void binRightMS() {
		if(mScalar2 == null)
			throw new RuntimeException();
		exec(op, mb, cmb, mScalar2);
	}

	@Test
	public void binLeftMM() {
		try {
			execL(op, mb, cmb, mb2);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test(expected = Exception.class)
	public void binLeftMrV() {
		if(mrv2 == null)
			throw new RuntimeException();
		execL(op, mb, cmb, mrv2);
	}

	@Test(expected = Exception.class)
	public void binLeftMrV_noCache() {
		CompressedMatrixBlock spy = spy(cmb);
		when(spy.getCachedDecompressed()).thenReturn(null);
		execL(op, mb, spy, mrv2);
	}

	@Test(expected = Exception.class)
	public void binLeftMrV_noCache_overlapping() {
		CompressedMatrixBlock spy = spy(cmb);
		when(spy.getCachedDecompressed()).thenReturn(null);
		when(spy.isOverlapping()).thenReturn(true);
		execL(op, mb, spy, mrv2);
	}

	@Test(expected = Exception.class)
	public void binLeftMcV() {
		if(mcv2 == null)
			throw new RuntimeException();
		execL(op, mb, cmb, mcv2);
	}

	@Test(expected = Exception.class)
	public void binLeftMcV_noCache() {
		CompressedMatrixBlock spy = spy(cmb);
		when(spy.getCachedDecompressed()).thenReturn(null);
		execL(op, mb, spy, mcv2);
	}

	@Test(expected = Exception.class)
	public void binLeftMS() {
		if(mScalar2 == null)
			throw new RuntimeException();
		execL(op, mb, cmb, mScalar2);
	}

	private static void exec(BinaryOperator op, MatrixBlock mb1, CompressedMatrixBlock cmb1, MatrixBlock mb2) {
		if(mb2 != null) {

			MatrixBlock cRet = null;
			MatrixBlock uRet = null;
			try{
				cRet = CLALibBinaryCellOp.binaryOperationsRight(op, cmb1, mb2);
				uRet = LibMatrixBincell.bincellOp(mb1, CompressedMatrixBlock.getUncompressed(mb2), null, op);
				compare(op, cRet, uRet);
			}
			catch(AssertionError e ){
				fail(e.getMessage());
				throw new RuntimeException(e);
			}
		}
	}

	private static void execL(BinaryOperator op, MatrixBlock mb1, CompressedMatrixBlock cmb1, MatrixBlock mb2) {
		if(mb2 != null) {
			MatrixBlock cRet = null;
			try {
				cRet = CLALibBinaryCellOp.binaryOperationsLeft(op, cmb1, mb2);
			}
			catch(Exception e) {
				// e.printStackTrace();
				throw e;
			}
			MatrixBlock uRet = LibMatrixBincell.bincellOp(mb2, mb1, null, op);
			if(op.fn instanceof Plus || op.fn instanceof Multiply) {
				// not verifying correctness for all cases since bincellOp left is not supported.
				compare(op, cRet, uRet);
			}
		}
	}

	private static void compare(BinaryOperator op, MatrixBlock cRet, MatrixBlock uRet) {
		if(cRet.containsValue(Double.NaN)) // CLA is not consistent on NaN vs Infinite handling because Nan + Inf = Nan
			TestUtils.compareMatrices(uRet, cRet, 0.0, "", true);
		else
			TestUtils.compareMatricesBitAvgDistance(uRet, cRet, 0, 0, op.toString());
	}

}
