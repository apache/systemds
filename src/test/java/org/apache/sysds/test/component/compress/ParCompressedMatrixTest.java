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

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.TestConstants.MatrixTypology;
import org.apache.sysds.test.component.compress.TestConstants.OverLapping;
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
public class ParCompressedMatrixTest extends AbstractCompressedUnaryTests {

	public ParCompressedMatrixTest(SparsityType sparType, ValueType valType, ValueRange valRange,
		CompressionSettings compressionSettings, MatrixTypology matrixTypology, OverLapping ov) {
		super(sparType, valType, valRange, compressionSettings, matrixTypology, ov,
			InfrastructureAnalyzer.getLocalParallelism());
	}

	@Override
	public void testUnaryOperators(AggType aggType) {
		AggregateUnaryOperator auop = super.getUnaryOperator(aggType, _k);
		testUnaryOperators(aggType, auop);
	}

	@Test
	public void testLeftMatrixMatrixMultMediumSparse2() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			MatrixBlock matrix = DataConverter
				.convertToMatrixBlock(TestUtils.generateTestMatrix(132, rows, 0.9, 1.5, .1, 3));

			// Make Operator
			AggregateBinaryOperator abop = InstructionUtils.getMatMultOperator(_k);

			// vector-matrix uncompressed
			MatrixBlock ret1 = mb.aggregateBinaryOperations(matrix, mb, new MatrixBlock(), abop);

			// vector-matrix compressed
			MatrixBlock ret2 = cmb.aggregateBinaryOperations(matrix, cmb, new MatrixBlock(), abop);

			// compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
			if(compressionSettings.lossy) {
				TestUtils.compareMatricesPercentageDistance(d1, d2, 0.25, 0.83, this.toString());
			}
			else {
				if(rows > 65000)
					TestUtils.compareMatricesPercentageDistance(d1, d2, 0.50, 0.99, this.toString());
				else if(overlappingType == OverLapping.MATRIX_MULT_NEGATIVE ||
					overlappingType == OverLapping.MATRIX_PLUS || overlappingType == OverLapping.MATRIX ||
					overlappingType == OverLapping.COL)
					TestUtils.compareMatricesBitAvgDistance(d1, d2, 50000, 1000, this.toString());
				else
					TestUtils.compareMatricesBitAvgDistance(d1, d2, 15000, 500, this.toString());

			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

}
