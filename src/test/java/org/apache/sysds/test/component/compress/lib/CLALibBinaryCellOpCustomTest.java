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

import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionStatistics;
import org.apache.sysds.runtime.compress.lib.CLALibBinaryCellOp;
import org.apache.sysds.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysds.runtime.functionobjects.LessThanEquals;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class CLALibBinaryCellOpCustomTest {

	@Test
	public void notColVector() {
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject(), 2);
		CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(10, 10, 1.2);
		MatrixBlock c2 = new MatrixBlock(1, 10, 2.5);
		CompressedMatrixBlock spy = spy(c);
		when(spy.isOverlapping()).thenReturn(true);
		MatrixBlock cRet = CLALibBinaryCellOp.binaryOperationsRight(op, spy, c2);

		TestUtils.compareMatricesBitAvgDistance(new MatrixBlock(10, 10, -1.3), cRet, 0, 0, op.toString());
		MatrixBlock cRet2 = CLALibBinaryCellOp.binaryOperationsLeft(op, spy, c2);
		TestUtils.compareMatricesBitAvgDistance(new MatrixBlock(10, 10, 1.3), cRet2, 0, 0, op.toString());
	}

	@Test
	public void twoHotEncodedOutput() {
		BinaryOperator op = new BinaryOperator(LessThanEquals.getLessThanEqualsFnObject(), 2);
		BinaryOperator op2 = new BinaryOperator(LessThanEquals.getLessThanEqualsFnObject());
		BinaryOperator opLeft = new BinaryOperator(GreaterThanEquals.getGreaterThanEqualsFnObject(), 2);
		BinaryOperator opLeft2 = new BinaryOperator(GreaterThanEquals.getGreaterThanEqualsFnObject());

		MatrixBlock cDense = new MatrixBlock(30, 30, 2.0);
		for (int i = 0; i < 30; i++) {
			cDense.set(i,0, 1);
		}
		cDense.set(0,1, 1);
		Pair<MatrixBlock, CompressionStatistics> pair = CompressedMatrixBlockFactory.compress(cDense, 1);
		CompressedMatrixBlock c = (CompressedMatrixBlock) pair.getKey();
		MatrixBlock c2 = new MatrixBlock(30, 1, 1.0);
		CompressedMatrixBlock spy = spy(c);
		when(spy.getCachedDecompressed()).thenReturn(null);

		MatrixBlock cRet = CLALibBinaryCellOp.binaryOperationsRight(op, spy, c2);
		MatrixBlock cRet2 = CLALibBinaryCellOp.binaryOperationsRight(op2, spy, c2);
		TestUtils.compareMatricesBitAvgDistance(cRet, cRet2, 0, 0, op.toString());

		MatrixBlock cRetleft = CLALibBinaryCellOp.binaryOperationsLeft(opLeft, spy, c2);
		MatrixBlock cRetleft2 = CLALibBinaryCellOp.binaryOperationsLeft(opLeft2, spy, c2);
		TestUtils.compareMatricesBitAvgDistance(cRetleft, cRetleft2, 0, 0, op.toString());

		TestUtils.compareMatricesBitAvgDistance(cRet, cRetleft, 0, 0, op.toString());
	}

	@Test
	public void notColVectorEmptyReturn() {
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject(), 2);
		CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(10, 10, 2.5);
		MatrixBlock c2 = new MatrixBlock(1, 10, 2.5);
		CompressedMatrixBlock spy = spy(c);
		when(spy.isOverlapping()).thenReturn(true);
		MatrixBlock cRet = CLALibBinaryCellOp.binaryOperationsRight(op, spy, c2);

		TestUtils.compareMatricesBitAvgDistance(new MatrixBlock(10, 10, true), cRet, 0, 0, op.toString());
		MatrixBlock cRet2 = CLALibBinaryCellOp.binaryOperationsLeft(op, spy, c2);
		TestUtils.compareMatricesBitAvgDistance(new MatrixBlock(10, 10, true), cRet2, 0, 0, op.toString());
	}

	@Test
	public void notRowVectorEmptyReturn() {
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject(), 2);
		CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(10, 10, 2.5);
		MatrixBlock c2 = new MatrixBlock(10, 1, 2.5);
		CompressedMatrixBlock spy = spy(c);
		when(spy.isOverlapping()).thenReturn(true);
		MatrixBlock cRet = CLALibBinaryCellOp.binaryOperationsRight(op, spy, c2);

		TestUtils.compareMatricesBitAvgDistance(new MatrixBlock(10, 10, true), cRet, 0, 0, op.toString());
		MatrixBlock cRet2 = CLALibBinaryCellOp.binaryOperationsLeft(op, spy, c2);
		TestUtils.compareMatricesBitAvgDistance(new MatrixBlock(10, 10, true), cRet2, 0, 0, op.toString());
	}

	@Test
	public void OVV() {
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject(), 2);
		CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(10, 1, 2.5);
		MatrixBlock c2 = new MatrixBlock(1, 10, 2.5);
		CompressedMatrixBlock spy = spy(c);
		when(spy.isOverlapping()).thenReturn(true);
		MatrixBlock cRet = CLALibBinaryCellOp.binaryOperationsRight(op, spy, c2);

		TestUtils.compareMatricesBitAvgDistance(new MatrixBlock(10, 10, true), cRet, 0, 0, op.toString());
	}

	@Test
	public void OVV2() {
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject(), 2);
		CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(10, 1, 2.5);
		MatrixBlock c2 = new MatrixBlock(1, 10, 324.0);
		CompressedMatrixBlock spy = spy(c);
		when(spy.isOverlapping()).thenReturn(true);
		MatrixBlock cRet = CLALibBinaryCellOp.binaryOperationsRight(op, spy, c2);

		TestUtils.compareMatricesBitAvgDistance(new MatrixBlock(10, 10, 2.5 - 324.0), cRet, 0, 0, op.toString());
	}

	@Test
	public void overwriteToCompressedOnSecondCompressed() {
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject(), 2);
		CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(10, 10, 2.5);
		MatrixBlock c2 = new MatrixBlock(10, 10, 324.0);
		CompressedMatrixBlock spy = spy(c);
		when(spy.isOverlapping()).thenReturn(true);
		MatrixBlock cRet = c2.binaryOperations(op, spy);

		TestUtils.compareMatricesBitAvgDistance(new MatrixBlock(10, 10, 324.0 - 2.5), cRet, 0, 0, op.toString());
	}

}
