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

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.functionobjects.IfElse;
import org.apache.sysds.runtime.functionobjects.MinusMultiply;
import org.apache.sysds.runtime.functionobjects.PlusMultiply;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class CLALibTernaryOpTest {

	final TernaryOperator op;

	public CLALibTernaryOpTest(TernaryOperator op) {
		this.op = op;
	}

	@Parameters(name = "{0}")
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();

		try {
			tests.add(new Object[] {new TernaryOperator(PlusMultiply.getFnObject(), 1)});
			tests.add(new Object[] {new TernaryOperator(MinusMultiply.getFnObject(), 1)});
			tests.add(new Object[] {new TernaryOperator(IfElse.getFnObject(), 1)});
			tests.add(new Object[] {new TernaryOperator(PlusMultiply.getFnObject(), 2)});
			tests.add(new Object[] {new TernaryOperator(MinusMultiply.getFnObject(), 2)});
			tests.add(new Object[] {new TernaryOperator(IfElse.getFnObject(), 2)});
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	@Test
	public void normal() {
		eval(10, 10, 10, 10, 10, 10);
	}

	@Test
	public void scalar2And3() {
		eval(10, 10, 1, 1, 1, 1);
	}

	@Test
	public void scalar2() {
		eval(10, 10, 1, 1, 10, 10);
	}

	@Test
	public void scalar3() {
		eval(10, 10, 10, 10, 1, 1);
	}

	@Test
	public void scalar1() {
		eval(1, 1, 10, 10, 10, 10);
	}

	@Test
	public void scalar1And3() {
		eval(1, 1, 10, 10, 1, 1);
	}

	@Test
	public void scalar1_2and3() {
		eval(1, 1, 1, 1, 1, 1);
	}

	@Test
	public void empty1_2and3() {
		eval(1, 1, 1, 1, 1, 1, 0.0, 0.0, 0.0);
	}

	@Test
	public void empty2and3() {
		eval(10, 10, 1, 1, 1, 1, 1.0, 0.0, 0.0);
	}

	@Test
	public void empty2and3V2() {
		eval(10, 10, 10, 10, 1, 1, 1.0, 0.0, 0.0);
	}

	@Test
	public void empty2and3V3() {
		eval(10, 10, 10, 10, 10, 10, 1.0, 0.0, 0.0);
	}

	@Test
	public void empty1and3() {
		eval(10, 1, 10, 1, 10, 1, 0.0, 1.02, 0.0);
	}

	@Test
	public void empty1and2() {
		eval(1, 10, 1, 10, 1, 10, 0.0, 0.0, 1.32);
	}

	private void eval(int a1, int a2, int b1, int b2, int c1, int c2) {
		eval(a1, a2, b1, b2, c1, c2, 1.2, 3.1, 2.2);
	}

	private void eval(int a1, int a2, int b1, int b2, int c1, int c2, double ad, double bd, double cd) {
		try {

			CompressedMatrixBlock cmb = CompressedMatrixBlockFactory.createConstant(a1, a2, ad);
			MatrixBlock mb = new MatrixBlock(a1, a2, ad);
			CompressedMatrixBlock cmb2 = CompressedMatrixBlockFactory.createConstant(b1, b2, bd);
			MatrixBlock m2 = new MatrixBlock(b1, b2, bd);
			CompressedMatrixBlock cmb3 = CompressedMatrixBlockFactory.createConstant(c1, c2, cd);
			MatrixBlock m3 = new MatrixBlock(c1, c2, cd);

			MatrixBlock uRet;
			uRet = mb.ternaryOperations(op, m2, m3);
			// input compressed
			compare(op, cmb.ternaryOperations(op, m2, m3), uRet);
			// all compressed
			compare(op, cmb.ternaryOperations(op, cmb2, cmb3), uRet);
			// two compressed
			compare(op, cmb.ternaryOperations(op, cmb2, m3), uRet);
			compare(op, cmb.ternaryOperations(op, m2, cmb3), uRet);
			compare(op, mb.ternaryOperations(op, cmb2, cmb3), uRet);
			// cone compressed
			compare(op, mb.ternaryOperations(op, cmb2, m3), uRet);
			compare(op, mb.ternaryOperations(op, m2, cmb3), uRet);

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private static void compare(TernaryOperator op, MatrixBlock cRet, MatrixBlock uRet) {
		TestUtils.compareMatricesBitAvgDistance(uRet, cRet, 0, 0, op.toString());
	}
}
