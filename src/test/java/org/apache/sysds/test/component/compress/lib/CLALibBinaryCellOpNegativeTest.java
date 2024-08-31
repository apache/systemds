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

import static org.junit.Assert.assertTrue;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.lib.CLALibBinaryCellOp;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell.BinaryAccessType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.junit.Test;

public class CLALibBinaryCellOpNegativeTest {
	protected static final Log LOG = LogFactory.getLog(CLALibBinaryCellOpNegativeTest.class.getName());

	@Test(expected = Exception.class)
	public void rightFailingTest() {
		CLALibBinaryCellOp.binaryOperationsRight(null, null, null);
	}

	@Test(expected = Exception.class)
	public void leftFailingTest() {
		CLALibBinaryCellOp.binaryOperationsRight(null, null, null);
	}

	@Test(expected = Exception.class)
	public void notColVector() {
		BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject(), 2);
		CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(10, 10, 1.2);
		MatrixBlock m2 = new MatrixBlock(10, 2, 1.2);
		BinaryAccessType atype = LibMatrixBincell.getBinaryAccessTypeExtended(c, m2);
		assertTrue(atype == BinaryAccessType.INVALID);
		CLALibBinaryCellOp.binaryOperationsRight(op, c, m2);
	}

	@Test(expected = Exception.class)
	public void notRowVector() {
		BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject(), 2);
		CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(10, 10, 1.2);
		MatrixBlock m2 = new MatrixBlock(2, 10, 1.2);
		BinaryAccessType atype = LibMatrixBincell.getBinaryAccessTypeExtended(c, m2);
		assertTrue(atype == BinaryAccessType.INVALID);
		CLALibBinaryCellOp.binaryOperationsRight(op, c, m2);
	}

	@Test(expected = Exception.class)
	public void notMatrixVector() {
		BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject(), 2);
		CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(10, 10, 1.2);
		MatrixBlock m2 = new MatrixBlock(10, 11, 1.2);
		BinaryAccessType atype = LibMatrixBincell.getBinaryAccessTypeExtended(c, m2);
		assertTrue(atype == BinaryAccessType.INVALID);
		CLALibBinaryCellOp.binaryOperationsRight(op, c, m2);
	}

}
