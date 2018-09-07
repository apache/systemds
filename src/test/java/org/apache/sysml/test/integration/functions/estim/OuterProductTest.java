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

package org.apache.sysml.test.integration.functions.estim;

import org.junit.Test;
import org.apache.sysml.hops.estim.EstimatorBasicAvg;
import org.apache.sysml.hops.estim.EstimatorBasicWorst;
import org.apache.sysml.hops.estim.EstimatorBitsetMM;
import org.apache.sysml.hops.estim.EstimatorDensityMap;
import org.apache.sysml.hops.estim.EstimatorMatrixHistogram;
import org.apache.sysml.hops.estim.EstimatorSample;
import org.apache.sysml.hops.estim.SparsityEstimator;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;

/**
 * This is a basic sanity check for all estimator, which need
 * to compute the exact sparsity for the special case of outer products.
 */
public class OuterProductTest extends AutomatedTestBase 
{
	private final static int m = 1154;
	private final static int k = 1;
	private final static int n = 900;
	private final static double[] case1 = new double[]{0.1, 0.7};
	private final static double[] case2 = new double[]{0.6, 0.7};

	@Override
	public void setUp() {
		//do  nothing
	}
	
	@Test
	public void testBasicAvgCase1() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, n, case1);
	}
	
	@Test
	public void testBasicAvgCase2() {
		runSparsityEstimateTest(new EstimatorBasicAvg(), m, k, n, case2);
	}
	
	@Test
	public void testBasicWorstCase1() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, n, case1);
	}
	
	@Test
	public void testBasicWorstCase2() {
		runSparsityEstimateTest(new EstimatorBasicWorst(), m, k, n, case2);
	}
	
	@Test
	public void testDensityMapCase1() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, n, case1);
	}
	
	@Test
	public void testDensityMapCase2() {
		runSparsityEstimateTest(new EstimatorDensityMap(), m, k, n, case2);
	}
	
	@Test
	public void testDensityMap7Case1() {
		runSparsityEstimateTest(new EstimatorDensityMap(7), m, k, n, case1);
	}
	
	@Test
	public void testDensityMap7Case2() {
		runSparsityEstimateTest(new EstimatorDensityMap(7), m, k, n, case2);
	}
	
	@Test
	public void testBitsetMatrixCase1() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, k, n, case1);
	}
	
	@Test
	public void testBitsetMatrixCase2() {
		runSparsityEstimateTest(new EstimatorBitsetMM(), m, k, n, case2);
	}
	
	@Test
	public void testMatrixHistogramCase1() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(false), m, k, n, case1);
	}
	
	@Test
	public void testMatrixHistogramCase2() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(false), m, k, n, case2);
	}
	
	@Test
	public void testMatrixHistogramExceptCase1() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(true), m, k, n, case1);
	}
	
	@Test
	public void testMatrixHistogramExceptCase2() {
		runSparsityEstimateTest(new EstimatorMatrixHistogram(true), m, k, n, case2);
	}
	
	@Test
	public void testSamplingDefCase1() {
		runSparsityEstimateTest(new EstimatorSample(), m, k, n, case1);
	}
	
	@Test
	public void testSamplingDefCase2() {
		runSparsityEstimateTest(new EstimatorSample(), m, k, n, case2);
	}
	
	@Test
	public void testSampling20Case1() {
		runSparsityEstimateTest(new EstimatorSample(0.2), m, k, n, case1);
	}
	
	@Test
	public void testSampling20Case2() {
		runSparsityEstimateTest(new EstimatorSample(0.2), m, k, n, case2);
	}
	
	private void runSparsityEstimateTest(SparsityEstimator estim, int m, int k, int n, double[] sp) {
		MatrixBlock m1 = MatrixBlock.randOperations(m, k, sp[0], 1, 1, "uniform", 3);
		MatrixBlock m2 = MatrixBlock.randOperations(k, n, sp[1], 1, 1, "uniform", 3);
		MatrixBlock m3 = m1.aggregateBinaryOperations(m1, m2, 
			new MatrixBlock(), InstructionUtils.getMatMultOperator(1));
		
		//compare estimated and real sparsity
		double est = estim.estim(m1, m2);
		TestUtils.compareScalars(est, m3.getSparsity(), 1e-16);
	}
}
