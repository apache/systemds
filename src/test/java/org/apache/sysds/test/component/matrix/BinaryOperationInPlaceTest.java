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

import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class BinaryOperationInPlaceTest {
	@Test
	public void testPlus() {
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 1);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 2);
		execute(m1,m2);
	}

	@Test
	public void testPlus_emptyInplace() {
		MatrixBlock m1 = new MatrixBlock(10,10,false);
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 2);
		execute(m1,m2);
	}

	@Test 
	public void testPlus_emptyOther(){
		MatrixBlock m1 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 1);
		MatrixBlock m2 = new MatrixBlock(10,10,false);
		execute(m1,m2);
	}

	@Test 
	public void testPlus_emptyInplace_butAllocatedDense() {
		MatrixBlock m1 = new MatrixBlock(10,10,false);
		m1.allocateDenseBlock();
		MatrixBlock m2 = TestUtils.generateTestMatrixBlock(10, 10, 0, 10, 1.0, 2);
		execute(m1,m2);
	}

	private void execute(MatrixBlock m1, MatrixBlock m2){
		BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject());
		MatrixBlock ret1 = m1.binaryOperations(op, m2);
		m1.binaryOperationsInPlace(op, m2);

		TestUtils.compareMatricesBitAvgDistance(ret1, m1, 0, 0, "Result is incorrect for inplace op");
	}
}
