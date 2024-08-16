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

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.functionobjects.And;
import org.apache.sysds.runtime.functionobjects.BitwAnd;
import org.apache.sysds.runtime.functionobjects.BitwOr;
import org.apache.sysds.runtime.functionobjects.BitwShiftL;
import org.apache.sysds.runtime.functionobjects.BitwShiftR;
import org.apache.sysds.runtime.functionobjects.BitwXor;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Equals;
import org.apache.sysds.runtime.functionobjects.GreaterThan;
import org.apache.sysds.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysds.runtime.functionobjects.IntegerDivide;
import org.apache.sysds.runtime.functionobjects.LessThan;
import org.apache.sysds.runtime.functionobjects.LessThanEquals;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.MinusNz;
import org.apache.sysds.runtime.functionobjects.Modulus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.NotEquals;
import org.apache.sysds.runtime.functionobjects.Or;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.Power;
import org.apache.sysds.runtime.functionobjects.Xor;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class BinaryOperationInPlaceTestParameterized {
	protected static final Log LOG = LogFactory.getLog(BinaryOperationInPlaceTestParameterized.class.getName());

	private final MatrixBlock left;
	private final MatrixBlock right;
	private final BinaryOperator op;

	public BinaryOperationInPlaceTestParameterized(MatrixBlock left, MatrixBlock right, BinaryOperator op) {
		this.left = new MatrixBlock();
		this.right = right;
		if((left.getSparsity() < 1 || right.getSparsity() < 1) && op.fn instanceof Builtin &&
			((Builtin) op.fn).bFunc == BuiltinCode.LOG) {
			op = new BinaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.LOG_NZ), op.getNumThreads());
		}

		this.op = op;

		this.left.copy(left);
	}

	@Parameters
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();

		try {
			double[] sparsities = new double[] {0.0, 0.001, 0.1, 0.5, 1.0};

			BinaryOperator[] operators = new BinaryOperator[] {//
				new BinaryOperator(Plus.getPlusFnObject()), //
				new BinaryOperator(Minus.getMinusFnObject()), //
				new BinaryOperator(Or.getOrFnObject()), //
				new BinaryOperator(LessThan.getLessThanFnObject()), //
				new BinaryOperator(LessThanEquals.getLessThanEqualsFnObject()), //
				new BinaryOperator(GreaterThan.getGreaterThanFnObject()), //
				new BinaryOperator(GreaterThanEquals.getGreaterThanEqualsFnObject()), //
				new BinaryOperator(Multiply.getMultiplyFnObject()), //
				new BinaryOperator(Modulus.getFnObject()), //
				new BinaryOperator(IntegerDivide.getFnObject()), //
				new BinaryOperator(Equals.getEqualsFnObject()), //
				new BinaryOperator(NotEquals.getNotEqualsFnObject()), //
				new BinaryOperator(And.getAndFnObject()), //
				new BinaryOperator(Xor.getXorFnObject()), //
				new BinaryOperator(BitwAnd.getBitwAndFnObject()), //
				new BinaryOperator(BitwOr.getBitwOrFnObject()), //
				new BinaryOperator(BitwXor.getBitwXorFnObject()), //
				new BinaryOperator(BitwShiftL.getBitwShiftLFnObject()), //
				new BinaryOperator(BitwShiftR.getBitwShiftRFnObject()), //
				new BinaryOperator(Power.getPowerFnObject()), //
				new BinaryOperator(MinusNz.getMinusNzFnObject()), //
				// Builtin
				new BinaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.MIN)), //
				new BinaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.MAX)), //
				new BinaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.LOG)), //
				new BinaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.LOG_NZ)), //
				new BinaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.MAXINDEX)), //
				new BinaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.MININDEX)), //
				new BinaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.CUMMAX)), //
				new BinaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.CUMMIN)),//
			};

			for(double rightSparsity : sparsities) {
				MatrixBlock right = TestUtils.floor(TestUtils.generateTestMatrixBlock(100, 100, -10, 10, rightSparsity, 2));
				MatrixBlock rightV = TestUtils.floor(TestUtils.generateTestMatrixBlock(1, 100, -10, 10, rightSparsity, 2));
				for(double leftSparsity : sparsities) {
					MatrixBlock left = TestUtils.floor(TestUtils.generateTestMatrixBlock(100, 100, -10, 10, leftSparsity, 2));
					for(BinaryOperator op : operators) {
						tests.add(new Object[] {left, right, op});
						tests.add(new Object[] {left, rightV, op});
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

	@Test
	public void testInplace() {
		try {
			final int lrb = left.getNumRows();
			final int lcb = left.getNumColumns();
			final int rrb = right.getNumRows();
			final int rcb = right.getNumColumns();

			final double lspb = left.getSparsity();
			final double rspb = right.getSparsity();

			final MatrixBlock ret1 = left.binaryOperations(op, right);

			long nnzExpected = ret1.getNonZeros();

			assertEquals(lrb, left.getNumRows());
			assertEquals(lcb, left.getNumColumns());
			assertEquals(rrb, right.getNumRows());
			assertEquals(rcb, right.getNumColumns());

			left.binaryOperationsInPlace(op, right);

			long nnzActual = left.getNonZeros();

			assertEquals(lrb, left.getNumRows());
			assertEquals(lcb, left.getNumColumns());
			assertEquals(rrb, right.getNumRows());
			assertEquals(rcb, right.getNumColumns());
			assertEquals("nnz should be equivalent on inplace operations: " + op.toString(),nnzExpected, nnzActual);
			TestUtils.compareMatricesBitAvgDistance(ret1, left, 0, 0, "Result is incorrect for inplace \n" + op + "  "
				+ lspb + " " + rspb + " (" + lrb + "," + lcb + ")" + " (" + rrb + "," + rcb + ")");
		}
		catch(DMLRuntimeException e) {
			if(e.getMessage().contains("Invalid row safety of in place row operation: ")) {
				if(op.fn instanceof Divide || //
					op.fn instanceof Plus || //
					op.fn instanceof Minus || //
					op.fn instanceof Or)
					return;
			}
			e.printStackTrace();
			fail(e.getMessage());
		}
		catch(NotImplementedException e) {
			// TODO fix the not implemented instances.
			if(e.getMessage().contains("Not made sparse vector in place"))
				return;
			e.printStackTrace();
			fail(e.getMessage());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
