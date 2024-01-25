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
import org.apache.sysds.runtime.data.SparseBlockCOO;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockDCSR;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

public class MatrixStorage extends AutomatedTestBase {

	private static final int resolution = 18;
	private static final float resolutionDivisor = 2f;
	private static final float maxSparsity = .2f;
	private static final float dimTestSparsity = .0001f;

	static float[] sparsityProvider() {
		float[] sparsities = new float[resolution];
		float currentValue = maxSparsity;

		for (int i = 0; i < resolution; i++) {
			sparsities[i] = currentValue;
			currentValue /= resolutionDivisor;
		}

		return sparsities;
	}

	static int[][] dimsProvider(int rl, int maxCl, int minCl, int resolution) {
		int[][] dims = new int[2][resolution];
		for (int i = 0; i < resolution; i++) {
			dims[0][i] = rl;
			dims[1][i] = (int)(minCl + i * ((maxCl-minCl)/((float)resolution)));
		}

		return dims;
	}

	static int[][] balancedDimsProvider(int numEntries, float[] ratio, float qMax) {
		int resolution = ratio.length;
		int[][] dims = new int[2][resolution];

		for (int i = 0; i < resolution; i++) {
			ratio[i] = -qMax + 2 * qMax * (i / ((float)resolution));
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

	static String printAsPythonList(float[] list) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");

		for (float el : list)
			sb.append(el + ",");

		if (list.length > 0)
			sb.deleteCharAt(sb.length() - 1);

		sb.append("]");
		return sb.toString();
	}

	static String printAsPythonList(int[] list) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");

		for (long el : list)
			sb.append(el + ",");

		if (list.length > 0)
			sb.deleteCharAt(sb.length() - 1);

		sb.append("]");
		return sb.toString();
	}

	static String printAsPythonList(long[] list) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");

		for (long el : list)
			sb.append(el + ",");

		if (list.length > 0)
			sb.deleteCharAt(sb.length() - 1);

		sb.append("]");
		return sb.toString();
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	/*@Test
	public void testDense() {
		testSparseFormat(null, 1024, 1024);
	}

	@Test
	public void testMCSR() {
		testSparseFormat(SparseBlock.Type.MCSR, 1024, 1024);
	}

	@Test
	public void testCSR() {
		testSparseFormat(SparseBlock.Type.CSR, 1024, 1024);
	}

	@Test
	public void testCOO() {
		testSparseFormat(SparseBlock.Type.COO, 1024, 1024);
	}

	@Test
	public void testDCSR() {
		testSparseFormat(SparseBlock.Type.DCSR, 1024, 1024);
	}*/

	/*@Test
	public void testChangingDimsDense() {
		testChangingDims(null, dimTestSparsity, 1024, 10, 3000, 30);
	}

	@Test
	public void testChangingDimsMCSR() {
		testChangingDims(SparseBlock.Type.MCSR, dimTestSparsity, 1024, 10, 3000, 30);
	}

	@Test
	public void testChangingDimsCSR() {
		testChangingDims(SparseBlock.Type.CSR, dimTestSparsity, 1024, 10, 3000, 30);
	}

	@Test
	public void testChangingDimsCOO() {
		testChangingDims(SparseBlock.Type.COO, dimTestSparsity, 1024, 10, 3000, 30);
	}

	@Test
	public void testChangingDimsDCSR() {
		testChangingDims(SparseBlock.Type.DCSR, dimTestSparsity, 1024, 10, 3000, 30);
	}

	@Test
	public void testBalancedDimsDense() {
		testBalancedDims(null, dimTestSparsity, 1024*1024, 30, 10, 10);
	}

	@Test
	public void testBalancedDimsMCSR() {
		testBalancedDims(SparseBlock.Type.MCSR, dimTestSparsity, 1024*1024, 30, 10, 10);
	}

	@Test
	public void testBalancedDimsCSR() {
		testBalancedDims(SparseBlock.Type.CSR, dimTestSparsity, 1024*1024, 30, 10, 10);
	}

	@Test
	public void testBalancedDimsCOO() {
		testBalancedDims(SparseBlock.Type.COO, dimTestSparsity, 1024*1024, 30, 10, 10);
	}

	@Test
	public void testBalancedDimsDCSR() {
		testBalancedDims(SparseBlock.Type.DCSR, dimTestSparsity, 1024*1024, 30, 10, 10);
	}*/

	private void testSparseFormat(SparseBlock.Type btype, int rl, int cl, int repetitions) {
		float[] sparsities = MatrixStorage.sparsityProvider();
		long[][] results = new long[repetitions][sparsities.length];

		for (int repetition = 0; repetition < repetitions; repetition++)
			for (int sparsityIndex = 0; sparsityIndex < sparsities.length; sparsityIndex++)
				results[repetition][sparsityIndex] = evaluateMemoryConsumption(btype, sparsities[sparsityIndex], rl, cl);


		System.out.println("sparsities" + (btype == null ? "Dense" : btype.name()) + " = " + printAsPythonList(sparsities));
		System.out.println("memory" + (btype == null ? "Dense" : btype.name()) + " =  " + printAsPythonList(buildAverage(results)));
	}

	private void testChangingDims(SparseBlock.Type btype, double sparsity, int rl, int minCl, int maxCl, int resolution, int repetitions) {
		int[][] dims = MatrixStorage.dimsProvider(rl, minCl, maxCl, resolution);
		long[][] results = new long[repetitions][resolution];

		for (int repetition = 0; repetition < repetitions; repetition++)
			for (int dimIndex = 0; dimIndex < resolution; dimIndex++)
				results[repetition][dimIndex] = evaluateMemoryConsumption(btype, sparsity, dims[0][dimIndex], dims[1][dimIndex]);


		System.out.println("dims" + (btype == null ? "Dense" : btype.name()) + " = " + printAsPythonList(dims[1]));
		System.out.println("dimMemory" + (btype == null ? "Dense" : btype.name()) + " =  " + printAsPythonList(buildAverage(results)));
	}

	private void testBalancedDims(SparseBlock.Type btype, double sparsity, int numEntries, int resolution, float qMax /*The maximum / minimum row-column ratio*/, int repetitions) {
		float[] ratios = new float[resolution];
		int[][] dims = MatrixStorage.balancedDimsProvider(numEntries, ratios, qMax);
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
		try
		{
			if (btype == null)
				return Math.min(Long.MAX_VALUE, (long) DenseBlockFP64.estimateMemory(rl, cl));

			double[][] A = getRandomMatrix(rl, cl, -10, 10, sparsity, 7654321);

			MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);

			if (!mbtmp.isInSparseFormat())
				mbtmp.denseToSparse(true);

			SparseBlock srtmp = mbtmp.getSparseBlock();
			switch (btype) {
				case MCSR:
					SparseBlockMCSR mcsr = new SparseBlockMCSR(srtmp);
					return mcsr.getExactSizeInMemory();
				case CSR:
					SparseBlockCSR csr = new SparseBlockCSR(srtmp);
					return csr.getExactSizeInMemory();
				case COO:
					SparseBlockCOO coo = new SparseBlockCOO(srtmp);
					return coo.getExactSizeInMemory();
				case DCSR:
					SparseBlockDCSR dcsr = new SparseBlockDCSR(srtmp);
					return dcsr.getExactSizeInMemory();
			}
		} catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		throw new IllegalArgumentException();
	}
}
