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

package org.apache.sysds.test.component.matrix;

import static org.junit.Assert.assertEquals;

import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.junit.Test;

public class CumsumprodTest {
	@Test
	public void testCumsumprodDense() {
		MatrixBlock A = MatrixBlock.randOperations(100, 2, 0.9, 0, 10, "uniform", 7);
		UnaryOperator uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucumk+*"), 1, false);
		MatrixBlock B = A.unaryOperations(uop, new MatrixBlock());
		assertEquals(100, B.getNumRows());
	}
	
	@Test
	public void testCumsumprodSparseMCSR() {
		MatrixBlock A = MatrixBlock.randOperations(1000, 2, 0.05, 0, 10, "uniform", 7);
		A = new MatrixBlock(A, SparseBlock.Type.MCSR, true);
		UnaryOperator uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucumk+*"), 1, false);
		MatrixBlock B = A.unaryOperations(uop, new MatrixBlock());
		assertEquals(1000, B.getNumRows());
	}
	
	@Test
	public void testCumsumprodSparseCSR() {
		MatrixBlock A = MatrixBlock.randOperations(1000, 2, 0.05, 0, 10, "uniform", 7);
		A = new MatrixBlock(A, SparseBlock.Type.CSR, true);
		UnaryOperator uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucumk+*"), 1, false);
		MatrixBlock B = A.unaryOperations(uop, new MatrixBlock());
		assertEquals(1000, B.getNumRows());
	}
	
	@Test
	public void testCumsumprodSparseCOO() {
		MatrixBlock A = MatrixBlock.randOperations(1000, 2, 0.05, 0, 10, "uniform", 7);
		A = new MatrixBlock(A, SparseBlock.Type.COO, true);
		UnaryOperator uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucumk+*"), 1, false);
		MatrixBlock B = A.unaryOperations(uop, new MatrixBlock());
		assertEquals(1000, B.getNumRows());
	}

	@Test
	public void testCumsumprodEmpty() {
		MatrixBlock A = MatrixBlock.randOperations(1000, 2, 0.00, 0, 10, "uniform", 7);
		A = new MatrixBlock(A, SparseBlock.Type.MCSR, true);
		UnaryOperator uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucumk+*"), 1, false);
		MatrixBlock B = A.unaryOperations(uop, new MatrixBlock());
		assertEquals(1000, B.getNumRows());
	}
}
