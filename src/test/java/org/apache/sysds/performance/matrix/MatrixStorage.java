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

import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;

public class MatrixStorage {

	private final int resolution;
	private final float resolutionDivisor;
	private final float maxSparsity;

	public MatrixStorage() {
		this(18, 2f, .2f);
	}

	public MatrixStorage(int resolution, float resolutionDivisor, float maxSparsity) {
		this.resolution = resolution;
		this.resolutionDivisor = resolutionDivisor;
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

	private int[][] dimsProvider(int rl, int maxCl, int minCl, int resolution) {
		int[][] dims = new int[2][resolution];
		for (int i = 0; i < resolution; i++) {
			dims[0][i] = rl;
			dims[1][i] = (int)(minCl + i * ((maxCl-minCl)/((float)resolution-1)));
		}

		return dims;
	}

	int[][] balancedDimsProvider(int numEntries, float[] ratio, float qMax) {
		int resolution = ratio.length;
		int[][] dims = new int[2][resolution];

		for (int i = 0; i < resolution; i++) {
			ratio[i] = -qMax + 2 * qMax * (i / ((float)resolution-1));
			if (ratio[i] < 0) {
				// Then columns are bigger than rows
				// r * c = numEntries
				// r = (1 + abs(ratio[i])) * c
				// => numEntries = (1 + abs(ratio[i])) * c^2
				// => c = sqrt(numEntries / (1 + abs(ratio[i])))
				dims[1][i] = Math.round((float)Math.sqrt(numEntries / (1 - ratio[i])));
				dims[0][i] = Math.round(numEntries / (float)dims[1][i]);
			} else {
				dims[0][i] = Math.round((float)Math.sqrt(numEntries / (1 + ratio[i])));
				dims[1][i] = Math.round(numEntries / (float)dims[0][i]);
			}
		}

		return dims;
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

	private static String printAsPythonList(int[] list) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");

		for (long el : list)
			sb.append(el + ",");

		if (list.length > 0)
			sb.deleteCharAt(sb.length() - 1);

		sb.append("]");
		return sb.toString();
	}

	private static String printAsPythonList(long[] list) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");

		for (long el : list)
			sb.append(el + ",");

		if (list.length > 0)
			sb.deleteCharAt(sb.length() - 1);

		sb.append("]");
		return sb.toString();
	}

	public void testSparseFormat(SparseBlock.Type btype, int rl, int cl, int repetitions) {
		float[] sparsities = sparsityProvider();
		long[][] results = new long[repetitions][sparsities.length];

		for (int repetition = 0; repetition < repetitions; repetition++)
			for (int sparsityIndex = 0; sparsityIndex < sparsities.length; sparsityIndex++)
				results[repetition][sparsityIndex] = evaluateMemoryConsumption(btype, sparsities[sparsityIndex], rl, cl);


		System.out.println("sparsities" + (btype == null ? "Dense" : btype.name()) + " = " + printAsPythonList(sparsities));
		System.out.println("memory" + (btype == null ? "Dense" : btype.name()) + " =  " + printAsPythonList(buildAverage(results)));
	}

	public void testChangingDims(SparseBlock.Type btype, double sparsity, int rl, int minCl, int maxCl, int resolution, int repetitions) {
		int[][] dims = dimsProvider(rl, minCl, maxCl, resolution);
		long[][] results = new long[repetitions][resolution];

		for (int repetition = 0; repetition < repetitions; repetition++)
			for (int dimIndex = 0; dimIndex < resolution; dimIndex++)
				results[repetition][dimIndex] = evaluateMemoryConsumption(btype, sparsity, dims[0][dimIndex], dims[1][dimIndex]);


		System.out.println("dims" + (btype == null ? "Dense" : btype.name()) + " = " + printAsPythonList(dims[1]));
		System.out.println("dimMemory" + (btype == null ? "Dense" : btype.name()) + " =  " + printAsPythonList(buildAverage(results)));
	}

	public void testBalancedDims(SparseBlock.Type btype, double sparsity, int numEntries, int resolution, float qMax /*The maximum / minimum row-column ratio*/, int repetitions) {
		float[] ratios = new float[resolution];
		int[][] dims = balancedDimsProvider(numEntries, ratios, qMax);
		long[][] results = new long[repetitions][resolution];

		for (int repetition = 0; repetition < repetitions; repetition++)
			for (int ratioIndex = 0; ratioIndex < resolution; ratioIndex++)
				results[repetition][ratioIndex] = evaluateMemoryConsumption(btype, sparsity, dims[0][ratioIndex], dims[1][ratioIndex]);

		System.out.println("ratio" + (btype == null ? "Dense" : btype.name()) + " = " + printAsPythonList(ratios) + "");
		System.out.println("ratioMemory" + (btype == null ? "Dense" : btype.name()) + " =  " + printAsPythonList(buildAverage(results)) + "");
	}

	private long[] buildAverage(long[][] results) {
		long[] mResults = new long[results[0].length];
		for (int i = 0; i < results[0].length; i++) {
			for (int j = 0; j < results.length; j++)
				mResults[i] += results[j][i];
			mResults[i] /= results.length;
		}

		return mResults;
	}

	private long evaluateMemoryConsumption(SparseBlock.Type btype, double sparsity, int rl, int cl) {
		if (btype == null)
			return Math.min(Long.MAX_VALUE, (long) DenseBlockFP64.estimateMemory(rl, cl));

		double[][] A = TestUtils.generateTestMatrix(rl, cl, -10, 10, sparsity, 7654321);

		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);

		if (!mbtmp.isInSparseFormat())
			mbtmp.denseToSparse(true);

		SparseBlock srtmp = mbtmp.getSparseBlock();
		SparseBlock sb = SparseBlockFactory.copySparseBlock(btype, srtmp, true);
		return sb.getExactSizeInMemory();
	}
}
