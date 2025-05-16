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

package org.apache.sysds.performance.matrix;

import java.util.Arrays;

import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCOO;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockDCSR;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;

public class MMSparsityPerformance {

	private final int _rl;
	private final int _cl;

	private final int warmupRuns;
	private final int repetitions;
	private final int resolution;
	private final float resolutionDivisor;
	private final float maxSparsity;

	public MMSparsityPerformance() {
		this(1024, 1024, 15, 50, 18, .4f, 2f);
	}

	public MMSparsityPerformance(int rl, int cl, int warmupRuns, int repetitions,
		int resolution, float maxSparsity, float stepDivisor)
	{
		_rl = rl;
		_cl = cl;
		this.warmupRuns = warmupRuns;
		this.repetitions = repetitions;
		this.resolution = resolution;
		this.resolutionDivisor = stepDivisor;
		this.maxSparsity = maxSparsity;
	}

	private float[] sparsityProvider() {
		float[] sparsities = new float[resolution];
		float currentValue = maxSparsity;

		for (int i = 0; i < resolution; i++) {
			sparsities[i] = currentValue;
			currentValue /= resolutionDivisor;
		}

		return sparsities;
	}

	private static String printAsPythonList(float[] list) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");

		for (float el : list)
			sb.append(el + ",");

		if (list.length > 0)
			sb.deleteCharAt(sb.length() - 1);

		sb.append("]");
		return sb.toString();
	}

	private static String printAsPythonList(double[] list) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");

		for (double el : list)
			sb.append(el + ", ");

		if (list.length > 0){
			sb.deleteCharAt(sb.length() - 1);
			sb.deleteCharAt(sb.length() - 1);
		}

		sb.append("]");
		return sb.toString();
	}

	public void testSparseFormat(SparseBlock.Type btype1, SparseBlock.Type btype2) {
		float[] sparsities = this.sparsityProvider();
		double[] avgNanosPerSparsity = new double[sparsities.length];
		long[] results = new long[repetitions];
		for (int sparsityIndex = 0; sparsityIndex < sparsities.length; sparsityIndex++) {
			// Warmup the JVM
			for (int i = 0; i < warmupRuns; i++)
				runSparsityEstimateTest(btype1, btype2, sparsities[sparsityIndex]);

			for (int i = 0; i < repetitions; i++)
				results[i] = runSparsityEstimateTest(btype1, btype2, sparsities[sparsityIndex]);

			avgNanosPerSparsity[sparsityIndex] = Arrays.stream(results).average().getAsDouble();
		}

		System.out.println("sparsities" + (btype1 == null ? "Dense" : btype1.name()) + " = " + printAsPythonList(sparsities));
		System.out.println("avgNanos" + (btype2 == null ? "Dense" : btype2.name()) + " =  " + printAsPythonList(avgNanosPerSparsity));
	}

	@SuppressWarnings("null")
	private long runSparsityEstimateTest(SparseBlock.Type btype1, SparseBlock.Type btype2, float sparsity) {
		double[][] A = TestUtils.generateTestMatrix(_rl, _cl, -10, 10, sparsity, 7654321);
		double[][] B = TestUtils.generateTestMatrix(_rl, _cl, -10, 10, sparsity, 7654322);

		MatrixBlock mbtmp1 = DataConverter.convertToMatrixBlock(A);
		MatrixBlock mbtmp2 = DataConverter.convertToMatrixBlock(B);

		MatrixBlock m1;
		MatrixBlock m2;

		if (btype1 == null && btype2 == null) {
			if (mbtmp1.isInSparseFormat())
				mbtmp1.sparseToDense();
			if (mbtmp2.isInSparseFormat())
				mbtmp2.sparseToDense();

			m1 = mbtmp1;
			m2 = mbtmp2;
		} else {
			// Ensure that these are sparse blocks
			if (!mbtmp1.isInSparseFormat())
				mbtmp1.denseToSparse(true);
			if (!mbtmp2.isInSparseFormat())
				mbtmp2.denseToSparse(true);

			SparseBlock srtmp1 = mbtmp1.getSparseBlock();
			SparseBlock srtmp2 = mbtmp2.getSparseBlock();
			SparseBlock sblock1;
			SparseBlock sblock2;

			switch (btype1) {
				case MCSR:
					sblock1 = new SparseBlockMCSR(srtmp1);
					break;
				case CSR:
					sblock1 = new SparseBlockCSR(srtmp1);
					break;
				case COO:
					sblock1 = new SparseBlockCOO(srtmp1);
					break;
				case DCSR:
					sblock1 = new SparseBlockDCSR(srtmp1);
					break;
				default:
					throw new IllegalArgumentException("Unknown sparse block type");
			}

			switch (btype2) {
				case MCSR:
					sblock2 = new SparseBlockMCSR(srtmp2);
					break;
				case CSR:
					sblock2 = new SparseBlockCSR(srtmp2);
					break;
				case COO:
					sblock2 = new SparseBlockCOO(srtmp2);
					break;
				case DCSR:
					sblock2 = new SparseBlockDCSR(srtmp2);
					break;
				default:
					throw new IllegalArgumentException("Unknown sparse block type");
			}

			m1 = new MatrixBlock(_rl, _cl, sblock1.size(), sblock1);
			m2 = new MatrixBlock(_rl, _cl, sblock2.size(), sblock2);
		}

		long nanos = System.nanoTime();

		MatrixBlock m3 = m1.aggregateBinaryOperations(m1, m2, InstructionUtils.getMatMultOperator(1));
		m3.sum(); // forced execution
		
		return System.nanoTime() - nanos;
	}
}
