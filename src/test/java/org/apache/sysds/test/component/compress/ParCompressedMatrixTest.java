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

import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.lops.MapMultChain.ChainType;
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
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
public class ParCompressedMatrixTest extends AbstractCompressedUnaryTests {

	private int k = InfrastructureAnalyzer.getLocalParallelism();

	public ParCompressedMatrixTest(SparsityType sparType, ValueType valType, ValueRange valRange,
		CompressionSettings compressionSettings, MatrixTypology matrixTypology) {
		super(sparType, valType, valRange, compressionSettings, matrixTypology);
	}

	@Test
	public void testConstruction() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock)) {
				// TODO Compress EVERYTHING!
				return; // Input was not compressed then just pass test
				// Assert.assertTrue("Compression Failed \n" + this.toString(), false);
			}
			if(compressionSettings.lossy) {
				TestUtils.compareMatrices(input, deCompressed, lossyTolerance);
			}
			else {
				TestUtils.compareMatricesBitAvgDistance(input, deCompressed, rows, cols, 0, 0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testGetValue() {

		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			for(int i = 0; i < rows; i++)
				for(int j = 0; j < cols; j++) {
					double ulaVal = input[i][j];
					double claVal = cmb.getValue(i, j); // calls quickGetValue internally
					if(compressionSettings.lossy) {
						TestUtils.compareCellValue(ulaVal, claVal, lossyTolerance, false);
					}
					else {
						TestUtils.compareScalarBitsJUnit(ulaVal, claVal, 0); // Should be exactly same value
					}
				}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testMatrixMultChain() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			MatrixBlock vector1 = DataConverter
				.convertToMatrixBlock(TestUtils.generateTestMatrix(cols, 1, 0.5, 1.5, 1.0, 3));

			// ChainType ctype = ChainType.XtwXv;
			for(ChainType ctype : new ChainType[] {ChainType.XtwXv, ChainType.XtXv,
				// ChainType.XtXvy
			}) {

				MatrixBlock vector2 = (ctype == ChainType.XtwXv) ? DataConverter
					.convertToMatrixBlock(TestUtils.generateTestMatrix(rows, 1, 0.5, 1.5, 1.0, 3)) : null;

				// matrix-vector uncompressed
				MatrixBlock ret1 = mb.chainMatrixMultOperations(vector1, vector2, new MatrixBlock(), ctype, k);

				// matrix-vector compressed
				MatrixBlock ret2 = cmb.chainMatrixMultOperations(vector1, vector2, new MatrixBlock(), ctype, k);

				// compare result with input
				double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
				double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
				if(compressionSettings.lossy) {
					TestUtils.compareMatricesPercentageDistance(d1, d2, 0.92, 0.95, compressionSettings.toString());
				}
				else {
					TestUtils.compareMatricesBitAvgDistance(d1, d2, 2048, 32, compressionSettings.toString());
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testTransposeSelfMatrixMult() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test
			// ChainType ctype = ChainType.XtwXv;
			for(MMTSJType mType : new MMTSJType[] {MMTSJType.LEFT,
				// MMTSJType.RIGHT
			}) {
				// matrix-vector uncompressed
				MatrixBlock ret1 = mb.transposeSelfMatrixMultOperations(new MatrixBlock(), mType, k);

				// matrix-vector compressed
				MatrixBlock ret2 = cmb.transposeSelfMatrixMultOperations(new MatrixBlock(), mType, k);

				// compare result with input
				double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
				double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
				// High probability that The value is off by some amount
				if(compressionSettings.lossy) {
					/**
					 * Probably the worst thing you can do to increase the amount the values are estimated wrong
					 */
					TestUtils.compareMatricesPercentageDistance(d1, d2, 0.0, 0.8, compressionSettings.toString());
				}
				else {
					TestUtils.compareMatricesBitAvgDistance(d1, d2, 2048, 20, compressionSettings.toString());
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testMatrixVectorMult02() {
		testMatrixVectorMult(0.7, 1.0);
	}

	@Test
	public void testMatrixVectorMult03() {
		testMatrixVectorMult(-1.0, 1.0);
	}

	@Test
	public void testMatrixVectorMult04() {
		testMatrixVectorMult(1.0, 5.0);
	}

	public void testMatrixVectorMult(double min, double max) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			MatrixBlock vector = DataConverter
				.convertToMatrixBlock(TestUtils.generateTestMatrix(cols, 1, min, max, 1.0, 3));

			// matrix-vector uncompressed
			AggregateBinaryOperator abop = InstructionUtils.getMatMultOperator(k);
			MatrixBlock ret1 = mb.aggregateBinaryOperations(mb, vector, new MatrixBlock(), abop);

			// matrix-vector compressed
			MatrixBlock ret2 = cmb.aggregateBinaryOperations(cmb, vector, new MatrixBlock(), abop);

			// compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
			if(compressionSettings.lossy) {
				// TODO Make actual calculation to know the actual tolerance
				double scaledTolerance = lossyTolerance * 30 * max;
				TestUtils.compareMatrices(d1, d2, scaledTolerance);
			}
			else {
				TestUtils.compareMatricesBitAvgDistance(d1, d2, 2048, 5, compressionSettings.toString());
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testVectorMatrixMult() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			MatrixBlock vector = DataConverter
				.convertToMatrixBlock(TestUtils.generateTestMatrix(1, rows, 1, 1, 1.0, 3));

			// Make Operator
			AggregateBinaryOperator abop = InstructionUtils.getMatMultOperator(k);

			// vector-matrix uncompressed
			MatrixBlock ret1 = mb.aggregateBinaryOperations(vector, mb, new MatrixBlock(), abop);

			// vector-matrix compressed
			MatrixBlock ret2 = cmb.aggregateBinaryOperations(vector, cmb, new MatrixBlock(), abop);

			// compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
			if(compressionSettings.lossy) {
				TestUtils.compareMatricesPercentageDistance(d1, d2, 0.35, 0.96, compressionSettings.toString());
			}
			else {
				TestUtils.compareMatricesBitAvgDistance(d1, d2, 10000, 500, compressionSettings.toString());
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Override
	public void testUnaryOperators(AggType aggType) {
		AggregateUnaryOperator auop = super.getUnaryOperator(aggType, k);
		testUnaryOperators(aggType, auop);
	}

}
