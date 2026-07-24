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
package org.apache.sysds.test.functions.federated.io;

import org.apache.sysds.runtime.controlprogram.federated.compression.CompressedMatrix;
import org.apache.sysds.runtime.controlprogram.federated.compression.CompressionType;
import org.apache.sysds.runtime.controlprogram.federated.compression.Quantization.ProbabilisticQuantizationCompressor;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class ProbabilisticQuantizationCompressorTest {

	@Test
	public void testQuantizationBasicProperties() throws Exception {
		MatrixBlock input = createRandomMatrix(10, 20);
		ProbabilisticQuantizationCompressor compressor = new ProbabilisticQuantizationCompressor(4);
		CompressedMatrix compressed = compressor.compress(input);
		MatrixBlock result = compressor.decompress(compressed);

		assertEquals(CompressionType.PROBABILISTIC_QUANTIZATION, compressed.getType());
		assertEquals(10, result.getNumRows());
		assertEquals(20, result.getNumColumns());
		assertEquals(8.0, compressed.getCompressionRatio(), 1e-10);

		double origMin = Double.MAX_VALUE;
		double origMax = -Double.MAX_VALUE;
		for(int i = 0; i < 10; i++) {
			for(int j = 0; j < 20; j++) {
				double v = input.get(i, j);
				if(v < origMin)
					origMin = v;
				if(v > origMax)
					origMax = v;
			}
		}
		for(int i = 0; i < 10; i++)
			for(int j = 0; j < 20; j++) {
				double v = result.get(i, j);
				assertTrue(v >= origMin - 1e-9);
				assertTrue(v <= origMax + 1e-9);
			}
	}

	@Test
	public void testFewerBitsGivesHigherRatio() throws Exception {
		MatrixBlock input = createRandomMatrix(20, 20);

		double ratio2bit = new ProbabilisticQuantizationCompressor(2).compress(input).getCompressionRatio();
		double ratio8bit = new ProbabilisticQuantizationCompressor(8).compress(input).getCompressionRatio();

		assertTrue(ratio2bit > ratio8bit);
	}

	@Test
	public void testUnbiasednessOverManyRuns() throws Exception {
		MatrixBlock input = new MatrixBlock(1, 1, false);
		input.allocateDenseBlock();
		input.set(0, 0, 5.0);
		input.examSparsity();

		double sum = 0.0;
		int runs = 1000;
		for(int r = 0; r < runs; r++) {
			ProbabilisticQuantizationCompressor compressor = new ProbabilisticQuantizationCompressor(4);
			MatrixBlock result = compressor.decompress(compressor.compress(input));
			sum += result.get(0, 0);
		}
		assertEquals(5.0, sum / runs, 0.5);
	}

	@Test
	public void testConstantMatrix() throws Exception {
		MatrixBlock input = new MatrixBlock(3, 3, false);
		input.allocateDenseBlock();
		for(int i = 0; i < 3; i++)
			for(int j = 0; j < 3; j++)
				input.set(i, j, 7.0);
		input.examSparsity();

		ProbabilisticQuantizationCompressor compressor = new ProbabilisticQuantizationCompressor(4);
		MatrixBlock result = compressor.decompress(compressor.compress(input));

		assertNotNull(result);
		assertEquals(3, result.getNumRows());
		assertEquals(3, result.getNumColumns());
	}

	@Test(expected = IllegalArgumentException.class)
	public void testInvalidBitsThrowsException() {
		new ProbabilisticQuantizationCompressor(3);
	}

	@Test
	public void testCompressionRatio2Bit() throws Exception {
		MatrixBlock input = createRandomMatrix(10, 10);
		assertEquals(16.0, new ProbabilisticQuantizationCompressor(2).compress(input).getCompressionRatio(), 1e-10);
	}

	@Test
	public void testCompressionRatio8Bit() throws Exception {
		MatrixBlock input = createRandomMatrix(10, 10);
		assertEquals(4.0, new ProbabilisticQuantizationCompressor(8).compress(input).getCompressionRatio(), 1e-10);
	}

	private MatrixBlock createRandomMatrix(int rows, int cols) {
		MatrixBlock m = new MatrixBlock(rows, cols, false);
		m.allocateDenseBlock();
		java.util.Random rng = new java.util.Random(42);
		for(int i = 0; i < rows; i++)
			for(int j = 0; j < cols; j++)
				m.set(i, j, rng.nextGaussian() * 10);
		m.examSparsity();
		return m;
	}
}
