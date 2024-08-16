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

package org.apache.sysds.test.component.matrix.binary;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.And;
import org.apache.sysds.runtime.functionobjects.BitwAnd;
import org.apache.sysds.runtime.functionobjects.BitwOr;
import org.apache.sysds.runtime.functionobjects.BitwShiftL;
import org.apache.sysds.runtime.functionobjects.BitwShiftR;
import org.apache.sysds.runtime.functionobjects.BitwXor;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.Equals;
import org.apache.sysds.runtime.functionobjects.GreaterThan;
import org.apache.sysds.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysds.runtime.functionobjects.IntegerDivide;
import org.apache.sysds.runtime.functionobjects.LessThan;
import org.apache.sysds.runtime.functionobjects.LessThanEquals;
import org.apache.sysds.runtime.functionobjects.Minus;
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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.mockito.Mockito;

@RunWith(value = Parameterized.class)
public class BinaryOpTest {

	protected static final Log LOG = LogFactory.getLog(BinaryOpTest.class.getName());

	private final MatrixBlock a;
	private final MatrixBlock b;
	private final int k;
	private final ValueFunction op;

	public BinaryOpTest(MatrixBlock a, MatrixBlock b, int k, String name, ValueFunction op) {
		this.a = a;
		this.b = b;
		this.k = k;
		this.op = op;
		a.recomputeNonZeros();
		b.recomputeNonZeros();
		// important that the thread is called something containing main!
		Thread.currentThread().setName("main_test_" + Thread.currentThread().getId());
	}

	@Parameters(name = "{3}")
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();
		try {

			ValueFunction[] vf = {//
				(Plus.getPlusFnObject()), //
				(Minus.getMinusFnObject()), //
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
				(Power.getPowerFnObject()), //
				(MinusNz.getMinusNzFnObject()), //
				(new PlusMultiply(32)), (new PlusMultiply(0)), (new MinusMultiply(32)),
				// Builtin
				(Builtin.getBuiltinFnObject(BuiltinCode.MIN)), //
				(Builtin.getBuiltinFnObject(BuiltinCode.MAX)), //
				(Builtin.getBuiltinFnObject(BuiltinCode.LOG)), //
				(Builtin.getBuiltinFnObject(BuiltinCode.LOG_NZ)), //
				(Builtin.getBuiltinFnObject(BuiltinCode.MAXINDEX)), //
				(Builtin.getBuiltinFnObject(BuiltinCode.MININDEX)), //
				(Builtin.getBuiltinFnObject(BuiltinCode.CUMMAX)), //
				(Builtin.getBuiltinFnObject(BuiltinCode.CUMMIN)),//
			};
			double[] sparsities = new double[] {0.0, 0.01, 0.1, 0.42, 1.0};
			// int[] sizes = new int[] {10, 100, 300};
			int[] sizes = new int[] {100};
			for(int s : sizes) {
				for(double rs : sparsities) {
					final MatrixBlock b = TestUtils.floor(TestUtils.generateTestMatrixBlock(s, s, 0, 10, rs, 3));
					final MatrixBlock b_rv = TestUtils.floor(TestUtils.generateTestMatrixBlock(1, s, 0, 10, rs, 4));
					final MatrixBlock b_cv = TestUtils.floor(TestUtils.generateTestMatrixBlock(s, 1, 0, 10, rs, 5));
					for(double ls : sparsities) {
						final MatrixBlock a = TestUtils.floor(TestUtils.generateTestMatrixBlock(s, s, 0, 10, ls, 2));
						final MatrixBlock a_dense;
						if(ls < 0.4 && a.isInSparseFormat()) {
							a_dense = new MatrixBlock();
							// a_dense.copy(a, false);
						}
						else
							a_dense = null;
						final MatrixBlock a_cv = TestUtils.floor(TestUtils.generateTestMatrixBlock(s, 1, 0, 10, ls, 2));
						for(ValueFunction v : vf) {
							tests.add(new Object[] {a, b, 1, name("%s-st-MM-%4.4f-%4.4f-%d", s, ls, rs, v), v});
							tests.add(new Object[] {a, b, 16, name("%s-mt-MM-%4.4f-%4.4f-%d", s, ls, rs, v), v});
							// if(a_dense != null) {
							// 	tests.add(new Object[] {a_dense, b, 1, name("%s-st-MM-%4.4f-%4.4f-%d", s, ls, rs, v), v});
							// 	tests.add(new Object[] {a_dense, b, 16, name("%s-mt-MM-%4.4f-%4.4f-%d", s, ls, rs, v), v});
							// }
							tests.add(new Object[] {a, b_rv, 1, name("%s-st-MrV-%4.4f-%4.4f-%d", s, ls, rs, v), v});
							tests.add(new Object[] {a, b_rv, 16, name("%s-mt-MrV-%4.4f-%4.4f-%d", s, ls, rs, v), v});
							tests.add(new Object[] {a, b_cv, 1, name("%s-st-McV-%4.4f-%4.4f-%d", s, ls, rs, v), v});
							tests.add(new Object[] {a, b_cv, 16, name("%s-mt-McV-%4.4f-%4.4f-%d", s, ls, rs, v), v});
							tests.add(new Object[] {a_cv, b_rv, 1, name("%s-st-VoV-%4.4f-%4.4f-%d", s, ls, rs, v), v});
							tests.add(new Object[] {a_cv, b_rv, 16, name("%s-mt-VoV-%4.4f-%4.4f-%d", s, ls, rs, v), v});
						}
					}
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	private static String name(String base, int s, double ls, double rs, ValueFunction v) {
		return String.format(base, v, ls, rs, s);
	}

	@Test
	public void test() {
		try {
			MatrixBlock spy = spy(new MatrixBlock());
			when(spy.getLength()).thenReturn(16L * 1024 + 1);
			MatrixBlock c = a.binaryOperations(new BinaryOperator(op, k), b, spy);
			Mockito.reset(spy);
			verify(a, b, c, op);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private void verify(MatrixBlock a, MatrixBlock b, MatrixBlock r, ValueFunction op) {
		final int ra = a.getNumRows();
		final int ca = a.getNumColumns();
		final int rb = b.getNumRows();
		final int cb = b.getNumColumns();
		final boolean emptyR = r.isEmpty();

		if((a.getSparsity() < 1 || b.getSparsity() < 1) && op instanceof Builtin &&
			((Builtin) op).bFunc == BuiltinCode.LOG)
			op = Builtin.getBuiltinFnObject(BuiltinCode.LOG_NZ);

		// side stepping the Spy object
		if(!emptyR)
			r.sparseToDense();

		DenseBlock rdb = r.getDenseBlock();
		long nnz = 0;

		// remove indirection of MatrixBlock because we mock a Matrix Block.
		// and mocking makes the MatrixBlock interface extremely slow.
		DenseBlock adb = a.getDenseBlock();
		DenseBlock bdb = b.getDenseBlock();
		SparseBlock asb = a.getSparseBlock();
		SparseBlock bsb = b.getSparseBlock();

		if(ra == rb && ca == cb) { // MM
			for(int i = 0; i < ra; i++) {
				for(int j = 0; j < ca; j++) {
					double in1 = get(adb, asb, i, j);
					double in2 = get(bdb, bsb, i, j);
					nnz = eval(op, emptyR, rdb, nnz, i, j, in1, in2);
				}
			}
		}
		else if(ra == rb && cb == 1) { // McV
			for(int i = 0; i < ra; i++) {
				double in2 = get(bdb, bsb, i, 0);
				for(int j = 0; j < ca; j++) {
					double in1 = get(adb, asb, i, j);
					nnz = eval(op, emptyR, rdb, nnz, i, j, in1, in2);
				}
			}
		}
		else if(rb == 1 && ca == cb) { // MrV
			for(int i = 0; i < ra; i++) {
				for(int j = 0; j < ca; j++) {
					double in1 = get(adb, asb, i, j);
					double in2 = get(bdb, bsb, 0, j);
					nnz = eval(op, emptyR, rdb, nnz, i, j, in1, in2);
				}
			}
		}
		else if(ca == 1 && rb == 1) { // outer VV
			for(int i = 0; i < ra; i++) {
				double in1 = get(adb, asb, i, 0);
				for(int j = 0; j < cb; j++) {
					double in2 = get(bdb, bsb, 0, j);
					nnz = eval(op, emptyR, rdb, nnz, i, j, in1, in2);
				}
			}
		}
		else
			throw new RuntimeException();
		assertEquals(nnz, r.getNonZeros());
	}

	private double get(DenseBlock adb, SparseBlock asb, int i, int j) {
		return adb == null ? asb == null ? 0.0 : asb.get(i, j) : adb.get(i, j);
	}

	private long eval(ValueFunction op, final boolean emptyR, DenseBlock rdb, long nnz, int i, int j, double in1,
		double in2) {
		double v = op.execute(in1, in2);
		nnz += v != 0.0 ? 1 : 0;
		double v2 = emptyR ? 0.0 : rdb.get(i, j);
		if(!Util.eq(v, v2)) {
			fail(String.format("%d,%d cell not equal: expected %4.2f vs got %4.2f : inputs %2.2f and %2.1f op:%s", i, j, v, v2, in1, in2, op.toString()));
		}
		return nnz;
	}

}
