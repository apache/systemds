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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.TestConstants.MatrixTypology;
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;
import org.junit.Test;

public abstract class AbstractCompressedUnaryTests extends CompressedTestBase {

	public AbstractCompressedUnaryTests(SparsityType sparType, ValueType valType, ValueRange valRange,
		CompressionSettings compSettings, MatrixTypology matrixTypology) {
		super(sparType, valType, valRange, compSettings, matrixTypology);
	}

	enum AggType {
		ROWSUMS, COLSUMS, SUM, ROWSUMSSQ, COLSUMSSQ, SUMSQ, ROWMAXS, COLMAXS, MAX, ROWMINS, COLMINS, MIN,
	}

	@Test
	public void testUnaryOperator_ROWSUMS() {
		testUnaryOperators(AggType.ROWSUMS);
	}

	@Test
	public void testUnaryOperator_COLSUMS() {
		testUnaryOperators(AggType.COLSUMS);
	}

	@Test
	public void testUnaryOperator_SUM() {
		testUnaryOperators(AggType.SUM);
	}

	@Test
	public void testUnaryOperator_ROWSUMSSQ() {
		testUnaryOperators(AggType.ROWSUMSSQ);
	}

	@Test
	public void testUnaryOperator_COLSUMSSQ() {
		testUnaryOperators(AggType.COLSUMSSQ);
	}

	@Test
	public void testUnaryOperator_SUMSQ() {
		testUnaryOperators(AggType.SUMSQ);
	}

	@Test
	public void testUnaryOperator_ROWMAXS() {
		testUnaryOperators(AggType.ROWMAXS);
	}

	@Test
	public void testUnaryOperator_COLMAXS() {
		testUnaryOperators(AggType.COLMAXS);
	}

	@Test
	public void testUnaryOperator_MAX() {
		testUnaryOperators(AggType.MAX);
	}

	@Test
	public void testUnaryOperator_ROWMINS() {
		testUnaryOperators(AggType.MAX);
	}

	@Test
	public void testUnaryOperator_COLMINS() {
		testUnaryOperators(AggType.MAX);
	}

	@Test
	public void testUnaryOperator_MIN() {
		testUnaryOperators(AggType.MAX);
	}

	protected AggregateUnaryOperator getUnaryOperator(AggType aggType, int k) {
		switch(aggType) {
			case SUM:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uak+", k);
			case ROWSUMS:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uark+", k);
			case COLSUMS:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uack+", k);
			case SUMSQ:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uasqk+", k);
			case ROWSUMSSQ:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uarsqk+", k);
			case COLSUMSSQ:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uacsqk+", k);
			case MAX:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uamax", k);
			case ROWMAXS:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uarmax", k);
			case COLMAXS:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uacmax", k);
			case MIN:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uamin", k);
			case ROWMINS:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uarmin", k);
			case COLMINS:
				return InstructionUtils.parseBasicAggregateUnaryOperator("uacmin", k);
			default:
				throw new NotImplementedException("Not Supported Aggregate Unary operator in test");
		}
	}

	public abstract void testUnaryOperators(AggType aggType);

	public void testUnaryOperators(AggType aggType, AggregateUnaryOperator auop) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test
			// matrix-vector uncompressed
			MatrixBlock ret1 = mb.aggregateUnaryOperations(auop, new MatrixBlock(), Math.max(rows, cols), null, true);
			// matrix-vector compressed
			MatrixBlock ret2 = cmb.aggregateUnaryOperations(auop, new MatrixBlock(), Math.max(rows, cols), null, true);

			// compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
			int dim1 = (aggType == AggType.ROWSUMS || aggType == AggType.ROWSUMSSQ || aggType == AggType.ROWMINS ||
				aggType == AggType.ROWMINS) ? rows : 1;
			int dim2 = (aggType == AggType.COLSUMS || aggType == AggType.COLSUMSSQ || aggType == AggType.COLMAXS ||
				aggType == AggType.COLMINS) ? cols : 1;

			TestUtils.compareMatricesBitAvgDistance(d1, d2, dim1, dim2, 2048, 20, compressionSettings.toString());

		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}
}